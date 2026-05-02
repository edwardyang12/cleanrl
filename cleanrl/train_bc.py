import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm

class ExpertDataset(Dataset):
    def __init__(self, n_values=[3]):
        all_obs = []
        all_actions = []
        
        for N in n_values:
            with open(f"expert_data/expert_N{N}.pkl", "rb") as f:
                data = pickle.load(f)
                # Reshape so each observation is an individual row
                # We want the student to learn a single shared policy regardless of N or agent ID
                obs_reshaped = data["observations"]
                all_obs.append(obs_reshaped)
                all_actions.append(data["actions"].flatten())
                
        self.obs = torch.Tensor(np.vstack(all_obs))
        self.actions = torch.LongTensor(np.concatenate(all_actions))
        
    def __len__(self):
        return len(self.actions)
        
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

# This is your Student Network. Notice it is just ONE Actor, not a ModuleList.
# We want a Shared Actor that can control any agent.
class StudentActor(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, x):
        return self.network(x)

def train_behavioral_cloning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Expert Datasets...")
    dataset = ExpertDataset(n_values=[3])
    dataloader = DataLoader(dataset, batch_size=8196, shuffle=True)
    
    # Get dimensions dynamically from dataset
    obs_dim = dataset.obs.shape[1]
    num_actions = len(torch.unique(dataset.actions))
    
    student = StudentActor(obs_dim, num_actions).to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 50
    print(f"Starting Behavioral Cloning for {epochs} epochs...")
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_obs, batch_actions in tqdm(dataloader):
            batch_obs, batch_actions = batch_obs.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            logits = student(batch_obs)
            
            # Cross-Entropy for discrete action mapping
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy for tracking
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_actions).sum().item()
            total += batch_actions.size(0)
            
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {accuracy:.2f}%")
        
    torch.save(student.state_dict(), "student_bc_pretrained.pt")
    print("Pre-training complete. Saved to student_bc_pretrained.pt")

if __name__ == "__main__":
    train_behavioral_cloning()