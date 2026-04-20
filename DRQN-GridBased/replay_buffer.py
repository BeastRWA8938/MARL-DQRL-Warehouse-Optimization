import numpy as np
import random
import torch
from collections import deque

class SequentialReplayBuffer:
    def __init__(self, capacity=10000, sequence_length=8):
        self.capacity = capacity
        self.sequence_length = sequence_length
        # We store whole episodes, not individual steps!
        self.buffer = deque(maxlen=capacity) 
        self.current_episode = [] 

    def store_transition(self, state, action, reward, next_state, done):
        # Append to the currently running episode
        self.current_episode.append((state, action, reward, next_state, done))
        
        # If the agent delivers the package or crashes (episode done), 
        # save the whole tape to the main vault and reset the camera.
        if done:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample_batch(self, batch_size):
        # 1. Randomly pick 'batch_size' episodes from the vault
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for episode in sampled_episodes:
            # 2. Pick a random starting frame within the episode
            # We must ensure there is enough room to grab 8 frames without going out of bounds
            max_start = max(0, len(episode) - self.sequence_length)
            start_idx = random.randint(0, max_start)
            
            # 3. Slice the 8-step sequence
            sequence = episode[start_idx : start_idx + self.sequence_length]
            
            # Pad the sequence with zeros if the episode was shorter than 8 steps
            # Inside sample_batch, change np.zeros(11) to np.zeros(13)
            while len(sequence) < self.sequence_length:
                sequence.append((np.zeros(13), 3, 0.0, np.zeros(13), True)) # Action 3 is now Wait
                
            # Unpack the sequence
            s, a, r, s_prime, d = zip(*sequence)
            
            states.append(np.array(s))
            actions.append(np.array(a))
            rewards.append(np.array(r))
            next_states.append(np.array(s_prime))
            dones.append(np.array(d))

        # Convert everything to PyTorch Tensors
        # Final Shape: (Batch_Size, Sequence_Length, Feature_Size) e.g., (32, 8, 11)
        return (torch.FloatTensor(np.array(states)), 
                torch.LongTensor(np.array(actions)), 
                torch.FloatTensor(np.array(rewards)), 
                torch.FloatTensor(np.array(next_states)), 
                torch.FloatTensor(np.array(dones)))
    
    def __len__(self):
        return len(self.buffer)