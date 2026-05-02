import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
import gymnasium as gym

# Import your environment builder and Agent class from your main script
from ppo_pettingzoo_ma_atari_mappo import build_environments, Agent, parse_args

def harvest_dataset(N, num_trajectories=1000):
    args = parse_args()
    args.num_landmarks = N
    args.reward_cheat = False # Turn off the cheat; we just want the physical state
    args.env_id = "simple_spread_v3"
    
    # Disable the default video wrapper in build_environments so we can customize it here
    args.capture_video = False 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Build strict, un-hacked environment
    run_name = f"harvest_N{N}"
    envs, num_agents_per_game, num_games = build_environments(args, run_name, args.seed, current_local_ratio=0.5)
    
    # --- CUSTOM VIDEO RECORDER ---
    # Wrap the environment to record a video every 100 episodes
    envs = gym.wrappers.vector.RecordVideo(
        envs, 
        f"expert_videos/{run_name}", 
        episode_trigger=lambda x: x % 100 == 0, # Change this number to record more/less often
        name_prefix=f"expert_behavior_N{N}"
    )
    
    # 2. Load Oracle
    state_dim = (num_agents_per_game * np.array(envs.single_observation_space.shape).prod()) + args.num_landmarks
    oracle = Agent(envs, num_agents_per_game, state_dim).to(device)
    oracle.load_state_dict(torch.load(f"models/simple_spread_v3__ppo_pettingzoo_ma_atari_mappo__1__1777572871/1068_model.pth")) # Ensure path points to your saved model
    oracle.eval() # Lock batchnorm/dropout

    expert_obs = []
    expert_actions = []

    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
    
    print(f"Harvesting {num_trajectories} steps for N={N}...")
    with torch.no_grad():
        for step in tqdm(range(num_trajectories)):
            obs_reshaped = next_obs.view(-1, num_agents_per_game, oracle.obs_dim)
            
            actions_list = []
            for i in range(num_agents_per_game):
                logits = oracle.actors[i](obs_reshaped[:, i, :])
                # PURE EXPLOITATION: No categorical sampling
                deterministic_action = torch.argmax(logits, dim=-1) 
                actions_list.append(deterministic_action)
            
            action = torch.stack(actions_list, dim=1).view(-1)
            
            # Save raw numpy arrays for the dataset
            expert_obs.append(next_obs.cpu().numpy())
            expert_actions.append(action.cpu().numpy())
            
            step_data = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(step_data[0] if len(step_data) == 5 else step_data[0]).to(device)

        if step % 1000 == 0:
            print(step)

    # 3. Save to disk
    dataset = {
        "observations": np.vstack(expert_obs), 
        "actions": np.concatenate(expert_actions) 
    }
    
    os.makedirs("expert_data", exist_ok=True)
    with open(f"expert_data/expert_N{N}.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print(f"Successfully saved N={N} dataset!")
    envs.close()

if __name__ == "__main__":
    # Ensure you update the model path inside the function before running
    for N in [4]:
        harvest_dataset(N, num_trajectories=500000)