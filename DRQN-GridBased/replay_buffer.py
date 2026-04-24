import numpy as np
import random
import torch
from collections import deque

class SequentialReplayBuffer:
    def __init__(self, capacity=10000, sequence_length=8):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity) 
        self.current_episode = [] 

    def store_transition(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample_batch(self, batch_size):
        sampled_episodes = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for episode in sampled_episodes:
            max_start = max(0, len(episode) - self.sequence_length)
            start_idx = random.randint(0, max_start)
            
            sequence = episode[start_idx : start_idx + self.sequence_length]
            
            # Pad sequence to 13 dimensions if episode was shorter than 8 steps
            while len(sequence) < self.sequence_length:
                sequence.append((np.zeros(13), 3, 0.0, np.zeros(13), True)) 
                
            s, a, r, s_prime, d = zip(*sequence)
            
            states.append(np.array(s))
            actions.append(np.array(a))
            rewards.append(np.array(r))
            next_states.append(np.array(s_prime))
            dones.append(np.array(d))

        return (torch.FloatTensor(np.array(states)), 
                torch.LongTensor(np.array(actions)), 
                torch.FloatTensor(np.array(rewards)), 
                torch.FloatTensor(np.array(next_states)), 
                torch.FloatTensor(np.array(dones)))
    
    def __len__(self):
        return len(self.buffer)