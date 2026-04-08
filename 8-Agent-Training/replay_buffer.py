import random
import numpy as np
import torch

class SequentialReplayBuffer:
    def __init__(self, capacity=10000, sequence_length=5):
        """
        capacity: How many full episodes the memory can hold before forgetting old ones.
        sequence_length: How many frames the LSTM looks at in one "movie clip" (e.g., 5 frames).
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = [] # This will store lists of complete episodes
        self.position = 0

    def push_episode(self, episode):
        """
        Takes a full episode (from spawn to delivery/crash) and saves it.
        episode format: list of tuples (obs, actions, reward, next_obs, done)
        """
        # We can't train on an episode if it's shorter than our required movie length!
        if len(episode) < self.sequence_length:
            return
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Pulls a batch of random "movie clips" out of the memory for the neural network to study.
        """
        # 1. Randomly pick 'batch_size' episodes from memory
        episodes = random.sample(self.buffer, batch_size)
        
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = [], [], [], [], []
        
        for ep in episodes:
            # 2. Pick a random starting frame within the episode
            # We subtract sequence_length to ensure we don't accidentally grab a movie that runs off the end
            start_idx = random.randint(0, len(ep) - self.sequence_length)
            
            # 3. Slice out a chunk of frames (e.g., frame 10 to frame 15)
            sequence = ep[start_idx : start_idx + self.sequence_length]
            
            # 4. Unzip the data so we have a list of all obs, all actions, etc.
            obs, actions, rewards, next_obs, dones = zip(*sequence)
            
            batch_obs.append(np.array(obs))
            batch_actions.append(np.array(actions))
            batch_rewards.append(np.array(rewards))
            batch_next_obs.append(np.array(next_obs))
            batch_dones.append(np.array(dones))
            
        # 5. Convert everything to PyTorch Tensors and send them directly to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return (
            torch.FloatTensor(np.array(batch_obs)).to(device),
            torch.LongTensor(np.array(batch_actions)).to(device),
            torch.FloatTensor(np.array(batch_rewards)).to(device),
            torch.FloatTensor(np.array(batch_next_obs)).to(device),
            torch.FloatTensor(np.array(batch_dones)).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# Quick test to ensure it works
if __name__ == "__main__":
    buffer = SequentialReplayBuffer(capacity=100, sequence_length=5)
    
    # Fake an episode with 10 frames of data
    # format: (obs(6), actions(4), reward, next_obs(6), done)
    fake_episode = []
    for _ in range(10):
        fake_episode.append((
            np.random.rand(6), 
            np.array([1, 0, 0, 1]), 
            1.0, 
            np.random.rand(6), 
            False
        ))
        
    buffer.push_episode(fake_episode)
    print(f"Episodes in memory: {len(buffer)}")
    
    # Try sampling 1 batch of 5 frames
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size=1)
    print(f"Sampled Observation Shape: {obs.shape}") # Should be [1, 5, 6]