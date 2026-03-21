import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import ale_py
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from pettingzoo.utils.env import ParallelEnv

gym.register_envs(ale_py)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--env-id", type=str, default="pong_v5")
    parser.add_argument("--total-timesteps", type=int, default=20000000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=16) # 8 Games * 2 Agents
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs): # Add 'envs' here
        super().__init__()
        # Your existing network code...
        c, h, w = 6, 84, 84
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 6), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        # Handle flattened minibatch [BatchSize, 84, 84, 6] 
        # or collection step [NumGames, NumAgents, 84, 84, 6]
        if x.dim() == 4:
            # Flattened update: B = BatchSize, N = 1
            x_flat = x
            B, N = x.shape[0], 1
        else:
            # Collection step: B = Games, N = Agents
            B, N = x.shape[0], x.shape[1]
            x_flat = x.reshape(B * N, *x.shape[2:])
            
        return self.critic(self.network(x_flat.permute((0, 3, 1, 2)) / 255.0)).reshape(B, N)

    def get_action_and_value(self, x, action=None):
        if x.dim() == 4:
            x_flat = x
            B, N = x.shape[0], 1
        else:
            B, N = x.shape[0], x.shape[1]
            x_flat = x.reshape(B * N, *x.shape[2:])
            
        hidden = self.network(x_flat.permute((0, 3, 1, 2)) / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        else:
            # Ensure action matches the flattened batch size of logits
            action = action.reshape(-1)
            
        return (action.reshape(B, N), 
                probs.log_prob(action).reshape(B, N), 
                probs.entropy().reshape(B, N), 
                self.critic(hidden).reshape(B, N))

class PongDirectEnv(ParallelEnv):
    def __init__(self, render_mode=None):
        self.env = gym.make("ALE/Pong-v5", full_action_space=False, render_mode=render_mode)
        self.render_mode = render_mode
        self.np_random = self.env.unwrapped.np_random # Fix missing np_random

        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
        self.agents = ["first_0", "second_0"]
        self.possible_agents = self.agents[:]
        self.observation_spaces = {a: self.env.observation_space for a in self.agents}
        self.action_spaces = {a: self.env.action_space for a in self.agents}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {a: obs for a in self.agents}, {a: info for a in self.agents}

    def step(self, actions):
        obs, reward, term, trunc, info = self.env.step(int(actions.get(self.agents[0], 0)))
        return ({a: obs for a in self.agents}, {self.agents[0]: reward, self.agents[1]: -reward},
                {a: term for a in self.agents}, {a: trunc for a in self.agents}, {a: info for a in self.agents})

    def render(self): return self.env.render()
    def close(self): self.env.close()

def make_env(render_mode="rgb_array"):
    def thunk():
        env = PongDirectEnv(render_mode=render_mode)
        # 1. Apply all SuperSuit preprocessing
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="full")
        env = ss.resize_v1(env, 84, 84) # <--- This makes it 84x84
        env = ss.frame_stack_v1(env, 4) # <--- This affects channels
        env = ss.agent_indicator_v0(env, type_only=False) # <--- This adds channels to get to 6
        
        # 2. Vectorize
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="gymnasium")

        class InfoDictify(gym.vector.VectorWrapper):
            def reset(self, seed=None, options=None):
                obs, info = self.env.reset(seed=seed, options=options)
                return obs, (info if isinstance(info, dict) else {"data": info})
            def step(self, action):
                obs, rew, term, trunc, info = self.env.step(action)
                return obs, rew, term, trunc, (info if isinstance(info, dict) else {"data": info})
        
        env = InfoDictify(env)
        # 3. Stats and the SHIELD wrapper
        env = gym.wrappers.vector.RecordEpisodeStatistics(env)

        # 4. THE SHIELD (Place this LAST)
        class WorkerSpaceFix(gym.vector.VectorWrapper):
            def __init__(self, env):
                super().__init__(env)
                # Ensure we have the right shape for AsyncVectorEnv buffers
                temp_obs, _ = self.env.reset()
                self.observation_space = gym.spaces.Box(
                    low=0, high=255, shape=temp_obs.shape, dtype=np.uint8
                )
                self.action_space = gym.spaces.MultiDiscrete([6, 6])
                self._render_mode = render_mode

            @property
            def render_mode(self):
                return self._render_mode

            def get_wrapper_attr(self, name):
                # This catches the call before it hits RecordEpisodeStatistics
                if name in self.__dict__:
                    return getattr(self, name)
                # Manually dig into the stack to avoid the AttributeError
                if hasattr(self.env, name):
                    return getattr(self.env, name)
                # Final fallback
                try:
                    return self.env.get_wrapper_attr(name)
                except AttributeError:
                    return None

            def reset(self, seed=None, options=None):
                obs, info = self.env.reset(seed=seed, options=options)
                return obs, (info if isinstance(info, dict) else {"data": info})

            def step(self, action):
                # 1. Get data from the SuperSuit/PettingZoo wrapper
                obs, rew, term, trunc, info = self.env.step(action)
                
                # 2. Store the multi-agent data in a custom key that won't confuse Gymnasium
                # We use a single NumPy array for all agents in this game (shape [2])
                custom_info = {
                    "ma_rew": np.array(rew, dtype=np.float32),
                    "ma_term": np.array(term, dtype=np.bool_),
                    "ma_trunc": np.array(trunc, dtype=np.bool_)
                }
                
                # 3. If RecordEpisodeStatistics added data, preserve it but flatten it
                if "episode" in info:
                    custom_info["episode"] = {
                        "r": np.array(info["episode"]["r"]).flatten(),
                        "l": np.array(info["episode"]["l"]).flatten()
                    }

                # 4. Return scalar rewards/dones to the AsyncVectorEnv
                # We return EMPTY info to the vectorizer to avoid the "sequence" error
                return obs, 0.0, bool(np.all(term)), bool(np.all(trunc)), custom_info
        return WorkerSpaceFix(env)
    return thunk

class ManualDictInfoProxy(gym.vector.VectorWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 8 games * 2 agents = 16 agent streams
        self._num_envs = env.num_envs * 2
        self.single_observation_space = gym.spaces.Box(0, 255, (84, 84, 6), np.uint8)
        self.single_action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self._num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self._num_envs)

    @property
    def num_envs(self):
        return self._num_envs

    def render(self):
        # Duplicate frames so RecordVideo sees 16 frames (one per agent)
        frames = self.env.render()
        if isinstance(frames, (list, tuple)):
            return [f for frame in frames for f in (frame, frame)]
        return frames

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.reshape(self.num_envs, 84, 84, 6), info

    def step(self, action):
        # Action (16,) -> (8, 2)
        obs, _, _, _, info = self.env.step(action.reshape(-1, 2))
        
        # info["ma_rew"] is now (8, 2) because AsyncVectorEnv stacked the (2,) arrays
        real_rew = info["ma_rew"].flatten()
        real_term = info["ma_term"].flatten()
        real_trunc = info["ma_trunc"].flatten()
        
        new_info = {"_episode": np.zeros(self.num_envs, dtype=bool)}
        
        # Extract episodic stats if they exist
        if "episode" in info:
            # info["episode"]["r"] will be (8, 2)
            new_info["episode"] = {
                "r": info["episode"]["r"].flatten(),
                "l": info["episode"]["l"].flatten()
            }
            # Gymnasium's RecordEpisodeStatistics uses _episode as a mask
            # We recreate it for 16 agents
            new_info["_episode"] = np.repeat(np.any(info["ma_term"] | info["ma_trunc"], axis=1), 2)
            
        return obs.reshape(self.num_envs, 84, 84, 6), real_rew, real_term, real_trunc, new_info

if __name__ == "__main__":
    args = parse_args(); run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    num_games = 8
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_games)], context="spawn", shared_memory=False)
    envs = ManualDictInfoProxy(envs)
    if args.capture_video:
        from gymnasium.wrappers.vector import RecordVideo
        envs = RecordVideo(envs, f"videos/{run_name}", episode_trigger=lambda ep: ep % 50 == 0)

    envs.single_observation_space = gym.spaces.Box(0, 255, (84, 84, 6), np.uint8)
    envs.single_action_space = gym.spaces.Discrete(6)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Simplified Buffers: All use envs.num_envs (16)
    obs = torch.zeros((args.num_steps, envs.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, envs.num_envs)).to(device)
    values = torch.zeros((args.num_steps, envs.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_raw, _ = envs.reset()
    next_obs = torch.Tensor(next_obs_raw).to(device)
    next_done = torch.zeros(envs.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    # Also make next_done dynamic instead of hardcoded (approx line 225)

    for update in range(1, (args.total_timesteps // args.batch_size) + 1):
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step], dones[step] = next_obs, next_done
            with torch.no_grad():
                # Agent expects (B, N, ...), we provide (16, 1, 84, 84, 6)
                action, logprob, _, value = agent.get_action_and_value(next_obs) # Returns [16, 1]
                values[step] = value.squeeze(1) # Store as [16]
                actions[step] = action.squeeze(1)
                logprobs[step] = logprob.squeeze(1)
                values[step] = value.squeeze(1)
            actions[step], logprobs[step] = action.squeeze(1), logprob.squeeze(1)

            next_obs_raw, reward, terms, truncs, info = envs.step(action.cpu().numpy().flatten())
            next_obs = torch.Tensor(next_obs_raw).to(device)
            rewards[step] = torch.Tensor(reward).to(device)
            next_done = torch.Tensor(np.logical_or(terms, truncs)).to(device)

            if "episode" in info:
                for i in range(16):
                    if info["_episode"][i]:
                        writer.add_scalar(f"charts/episodic_return-player{i % 2}", info["episode"]["r"][i], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(1) # Shape [16]
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                nonterm = 1.0 - (next_done if t == args.num_steps-1 else dones[t+1])
                next_v = next_value if t == args.num_steps-1 else values[t+1]
                delta = rewards[t] + args.gamma * next_v * nonterm - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nonterm * lastgaelam
            returns = advantages + values

        # Use -1 to let PyTorch handle the flattening of (num_steps * num_games * num_agents)
        b_obs = obs.reshape((-1, 84, 84, 6)) # (128*8*2) = 2048
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # Pass b_obs[mb_inds] and b_actions[mb_inds] directly as flat tensors
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                # Since newlogprob and newvalue come back as (BatchSize, 1), flatten them for the loss
                logratio = newlogprob.reshape(-1) - b_logprobs[mb_inds]

                ratio = logratio.exp()
                mb_adv = b_advantages[mb_inds]
                if args.norm_adv: mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((newvalue.reshape(-1) - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm); optimizer.step()

        print(f"global_step={global_step}, SPS: {int(global_step / (time.time() - start_time))}")
    envs.close(); writer.close()
