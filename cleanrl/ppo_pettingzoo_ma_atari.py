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
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--env-id", type=str, default="pong_v5")
    parser.add_argument("--total-timesteps", type=int, default=2000000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--target-kl", type=float, default=None)
    return parser.parse_args()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        # 6 channels: 4 stacked frames + 2 agent indicators
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 6), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

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

    def _process_obs(self, obs, agent_id):
        obs = obs.astype(np.float32) / 255.0
        # If second agent, flip horizontally so it thinks it is the left player
        if agent_id == "second_0":
            obs = np.flip(obs, axis=1).copy()
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {a: self._process_obs(obs, a) for a in self.agents}, {a: info for a in self.agents}

    def step(self, actions):
        # Only 'first_0' actually moves the paddle in this single-player mock
        primary_action = int(actions.get("first_0", 0))
        obs, reward, term, trunc, info = self.env.step(primary_action)
        
        return ({a: self._process_obs(obs, a) for a in self.agents}, 
                {"first_0": reward, "second_0": -reward},
                {a: term for a in self.agents}, 
                {a: trunc for a in self.agents}, 
                {a: info for a in self.agents})

    def render(self): return self.env.render()
    def close(self): self.env.close()

class VectorInfoDict(gym.vector.VectorWrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not isinstance(info, dict): info = {"data": info}
        return obs, info
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if not isinstance(info, dict): info = {"data": info}
        return obs, reward, term, trunc, info

class FixVectorRender(gym.vector.VectorWrapper):
    def render(self):
        frame = self.env.render()
        return [frame] * self.num_envs

def make_env(render_mode="rgb_array"):
    env = PongDirectEnv(render_mode=render_mode)
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="full")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    return env

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_env(render_mode="rgb_array")
    envs = VectorInfoDict(envs)
    envs = FixVectorRender(envs)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    
    if args.capture_video:
        from gymnasium.wrappers.vector import RecordVideo
        envs = RecordVideo(envs, f"videos/{run_name}", episode_trigger=lambda ep: ep % 10 == 0)

    args.num_envs = envs.num_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, 6, 84, 84)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs_raw, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs_raw).to(device).permute(0, 3, 1, 2)
    next_done = torch.zeros(args.num_envs).to(device)

    # Creating a mask: first_0 is index 0, second_0 is index 1
    # We only want to learn from the agent that actually has agency (index 0)
    agent_mask = torch.tensor([1, 0] * (args.num_envs // 2)).to(device).float()

    for update in range(1, (args.total_timesteps // args.batch_size) + 1):
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step], dones[step] = next_obs, next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action.flatten()
                logprobs[step] = logprob.flatten()

            next_obs_raw, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs_raw).to(device).permute(0, 3, 1, 2)
            rewards[step] = torch.Tensor(reward).to(device)
            next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)

            if "episode" in infos:
                for i in range(args.num_envs):
                    if infos["_episode"][i]:
                        r = infos["episode"]["r"][i]
                        writer.add_scalar(f"charts/episodic_return_agent_{i%2}", r, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, 6, 84, 84))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Flattened mask to match batch size
        b_mask = agent_mask.repeat(args.num_steps)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start:start + args.minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss with agency mask
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2) * b_mask[mb_inds]).mean()

                # Value loss with agency mask
                newvalue = newvalue.view(-1)
                v_loss = (0.5 * ((newvalue - b_returns[mb_inds]) ** 2) * b_mask[mb_inds]).mean()

                entropy_loss = (entropy * b_mask[mb_inds]).mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        print(f"Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")

    envs.close()
    writer.close()