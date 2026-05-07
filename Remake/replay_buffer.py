import random
import numpy as np
import torch

class EpisodicReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.memory = []
        self.current_episodes = {}
        self.total_frames_stored = 0

    def push_transition(self, agent_id, state, action, reward, next_state, done):
        if agent_id not in self.current_episodes:
            self.current_episodes[agent_id] = []

        self.current_episodes[agent_id].append((state, action, reward, next_state, done))
        self.total_frames_stored += 1

        if done:
            self.finish_episode(agent_id)

    def finish_episode(self, agent_id):
        episode = self.current_episodes.pop(agent_id, [])
        if not episode:
            return

        self.memory.append(episode)

        while len(self.memory) > self.capacity:
            dropped_ep = self.memory.pop(0)
            self.total_frames_stored -= len(dropped_ep)

    def clear_active_episode(self, agent_id):
        episode = self.current_episodes.pop(agent_id, [])
        self.total_frames_stored -= len(episode)

    def finish_all_active_episodes(self):
        for agent_id in list(self.current_episodes.keys()):
            self.finish_episode(agent_id)

    def sample(self, batch_size, seq_len):
        valid_episodes = [ep for ep in self.memory if len(ep) >= seq_len]
        if len(valid_episodes) < batch_size:
            return None

        sampled_episodes = random.sample(valid_episodes, batch_size)
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = [], [], [], [], []

        for ep in sampled_episodes:
            start_idx = random.randint(0, len(ep) - seq_len)
            sequence = ep[start_idx : start_idx + seq_len]

            states, actions, rewards, next_states, dones = zip(*sequence)

            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(dones)

        return (
            torch.tensor(np.array(states_batch), dtype=torch.float32),
            torch.tensor(np.array(actions_batch), dtype=torch.long),
            torch.tensor(np.array(rewards_batch), dtype=torch.float32),
            torch.tensor(np.array(next_states_batch), dtype=torch.float32),
            torch.tensor(np.array(dones_batch), dtype=torch.float32)
        )
