# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
import sys
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cleanrl_utils.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    anneal_ent: bool = True
    """Toggle entropy coefficient annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 16
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.015
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # Define render_mode for all envs to ensure compatibility, 
        # but we only record on the first one.
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        
        # Force Pong-v5 to behave like a standard 'NoFrameskip' environment
        env = gym.make(env_id, render_mode=render_mode, frameskip=1, repeat_action_probability=0.25, full_action_space=False)
        # env = gym.make(env_id, render_mode=render_mode)
        
        # Record video only on the first environment
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, 
                f"videos/{run_name}",
                episode_trigger=lambda x: x % 10 == 0  # Records every 10th episode
            )
            
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        new_repo_space = gym.spaces.Box(
                low=0, 
                high=255, 
                shape=(110, 160, 3), 
                dtype=env.observation_space.dtype
            )

        env = gym.wrappers.TransformObservation(
            env, 
            lambda obs: obs[34:-16, :, :], 
            observation_space=new_repo_space
        )
        # env = gym.wrappers.NormalizeReward(env)  
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = ClipRewardEnv(env)
        # env = PongEnhancedRewardWrapper(env, contact_reward=0.02, time_penalty=-0.002)
        # env = PongAggressionWrapper(env, penalty=-0.001)
        env = gym.wrappers.ResizeObservation(env, (110, 110))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        # env = PongActionWrapper(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PongActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 0: Stay (0), 1: Up (2), 2: Down (3)
        self.mapping = {0: 0, 1: 2, 2: 3}
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return self.mapping[action]

class PongEnhancedRewardWrapper(gym.Wrapper):
    def __init__(self, env, contact_reward=0.1, time_penalty=-0.001):
        super().__init__(env)
        self.contact_reward = contact_reward
        self.time_penalty = time_penalty
        self.last_ball_x = 0
        self.was_moving_towards_player = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 1. Aggression Bonus: Small penalty every frame the ball is in play
        if reward == 0 and not (terminated or truncated):
            reward += self.time_penalty
            
        # 2. Contact Reward: Use RAM index 12 (Ball X-position)
        # Player is on the right side in Pong. Ball X increases as it moves right.
        ram = self.env.unwrapped.ale.getRAM()
        ball_x = int(ram[12])
        
        moving_towards_player = (ball_x > self.last_ball_x)
        
        # Check for direction change (hit) on the player's side (X > 180)
        if self.was_moving_towards_player and not moving_towards_player and ball_x > 180:
            if reward <= 0: # Only add if we didn't just score a point
                reward += self.contact_reward
                
        self.last_ball_x = ball_x
        self.was_moving_towards_player = moving_towards_player
        
        return obs, reward, terminated, truncated, info

class PongAggressionWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.001):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Apply penalty ONLY if it's a standard frame (reward is 0)
        # This prevents the penalty from being turned into -1 by ClipRewardEnv
        if reward == 0 and not (terminated or truncated):
            reward = self.penalty
        return obs, reward, terminated, truncated, info


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 10 * 10, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.num_actions = envs.single_action_space.n
        
#         # Shared CNN: The Actor's exploration "seeds" features for the Critic
#         self.network = nn.Sequential(
#             layer_init(nn.Conv2d(4, 32, 8, stride=4)), nn.ReLU(),
#             layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
#             layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
#             nn.Flatten(),
#         )
        
#         # Separate heads start here to prevent gradient interference
#         self.actor_fc = nn.Sequential(layer_init(nn.Linear(64 * 10 * 10, 512)), nn.ReLU())
#         self.critic_fc = nn.Sequential(layer_init(nn.Linear(64 * 10 * 10, 512)), nn.ReLU())

#         input_dim = 512
#         self.actor_head = layer_init(nn.Linear(input_dim, self.num_actions), std=0.01)
#         self.critic_head = layer_init(nn.Linear(input_dim, 1), std=1)

#     def get_action_and_value(self, x, action=None):
#         shared_features = self.network(x / 255.0)
        
#         # Split logic
#         a_hidden = self.actor_fc(shared_features)
#         c_hidden = self.critic_fc(shared_features)
                
#         logits = self.actor_head(a_hidden)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
            
#         value = self.critic_head(c_hidden)
#         return action, probs.log_prob(action), probs.entropy(), value

#     def get_value(self, x):
#         shared_features = self.network(x / 255.0)
#         c_hidden = self.critic_fc(shared_features)
#         return self.critic_head(c_hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            # Calculate the annealed rate but ensure it doesn't drop below the floor
            lrnow = max(5e-5, frac * args.learning_rate)
            optimizer.param_groups[0]["lr"] = lrnow
        # Updated annealing logic for the final push
        if args.anneal_ent:
            # Scale from start_ent down to 0 over the first 80% of training
            # Then hold at 0 for the final 20% to "harden" the policy
            progress = (iteration - 1.0) / args.num_iterations
            if progress < 0.8:
                ent_coef_now = args.ent_coef * (1.0 - progress / 0.8)
            else:
                ent_coef_now = 0.0
        else:
            ent_coef_now = args.ent_coef


        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Check for finished episodes in the vectorized info dictionary
            if "_episode" in infos and any(infos["_episode"]):
                for idx, d in enumerate(infos["_episode"]):
                    if d:
                        # Pull the scalars from the batched arrays for this specific env
                        r = infos["episode"]["r"][idx]
                        l = infos["episode"]["l"][idx]
                        print(f"global_step={global_step}, episodic_return={r}")
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

        # bootstrap value if not done
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

        # Add global advantage normalization here
        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
