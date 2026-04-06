import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import random

from drqn_model import DRQN
from replay_buffer import SequentialReplayBuffer

# --- Hyperparameters ---
BATCH_SIZE = 16
SEQ_LENGTH = 5
MEMORY_CAPACITY = 2000
MAX_STEPS = 100000
GAMMA = 0.99           # How much the AI cares about future rewards
EPSILON_START = 1.0    # 100% Random at the beginning
EPSILON_END = 0.05     # 5% Random at the end
EPSILON_DECAY = 50000  # How many steps it takes to go from Start to End

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting DRQN Training on: {device}")

    print("Waiting for Unity Environment... Please press PLAY in the Editor!")
    env = UnityEnvironment(file_name=None, seed=42)
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    decision_steps, _ = env.get_steps(behavior_name)
    total_obs_size = sum([obs.shape[1] for obs in decision_steps.obs])
    print(f"Observation size (Vectors + Raycasts): {total_obs_size}")

    q_network = DRQN(input_size=total_obs_size).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    memory = SequentialReplayBuffer(capacity=MEMORY_CAPACITY, sequence_length=SEQ_LENGTH)

    active_episodes = {} 
    epsilon = EPSILON_START

    print("\n--- Real Training Started! ---")
    print("Watch the robots in Unity. They will start random, but slowly get smarter.")
    print("Press Ctrl+C to save and quit.\n")
    
    try:
        for step in range(MAX_STEPS):
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # --- A. RECORD OBSERVATIONS ---
            current_obs = {}
            for agent_id in decision_steps.agent_id:
                agent_idx = decision_steps.agent_id_to_index[agent_id]
                combined_obs = np.concatenate([obs[agent_idx] for obs in decision_steps.obs])
                current_obs[agent_id] = combined_obs
                
                if agent_id not in active_episodes:
                    active_episodes[agent_id] = [] 

            # --- B. EPSILON-GREEDY ACTION SELECTION ---
            actions_to_send = []
            agent_ids_taking_action = []

            for agent_id in decision_steps.agent_id:
                if random.random() < epsilon:
                    # EXPLORE: Take a random action
                    act = [random.randint(0,2), random.randint(0,2), random.randint(0,1), random.randint(0,2)]
                else:
                    # EXPLOIT: Ask the Neural Network!
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(current_obs[agent_id]).unsqueeze(0).unsqueeze(0).to(device)
                        q_vals, _ = q_network(obs_tensor)
                        act = [
                            torch.argmax(q_vals[0]).item(), # Move
                            torch.argmax(q_vals[1]).item(), # Turn
                            torch.argmax(q_vals[2]).item(), # Interact
                            torch.argmax(q_vals[3]).item()  # Forks
                        ]

                actions_to_send.append(act)
                agent_ids_taking_action.append(agent_id)

            if len(actions_to_send) > 0:
                action_tuple = ActionTuple(discrete=np.array(actions_to_send, dtype=np.int32))
                env.set_actions(behavior_name, action_tuple)

            # --- C. STEP ENVIRONMENT ---
            env.step()

            # --- D. GATHER REWARDS & SAVE ---
            new_decision_steps, new_terminal_steps = env.get_steps(behavior_name)

            for idx, agent_id in enumerate(agent_ids_taking_action):
                if agent_id in new_decision_steps.agent_id:
                    agent_idx = new_decision_steps.agent_id_to_index[agent_id]
                    next_o = np.concatenate([obs[agent_idx] for obs in new_decision_steps.obs])
                    reward = new_decision_steps.reward[agent_idx]
                    active_episodes[agent_id].append((current_obs[agent_id], actions_to_send[idx], reward, next_o, False))

            for agent_id in new_terminal_steps.agent_id:
                if agent_id in current_obs and agent_id in agent_ids_taking_action:
                    idx = agent_ids_taking_action.index(agent_id)
                    agent_idx = new_terminal_steps.agent_id_to_index[agent_id]
                    next_o = np.concatenate([obs[agent_idx] for obs in new_terminal_steps.obs])
                    reward = new_terminal_steps.reward[agent_idx]
                    
                    active_episodes[agent_id].append((current_obs[agent_id], actions_to_send[idx], reward, next_o, True))
                    memory.push_episode(active_episodes[agent_id])
                    active_episodes[agent_id] = [] 

            # --- E. DECAY EPSILON ---
            epsilon = max(EPSILON_END, EPSILON_START - step * ((EPSILON_START - EPSILON_END) / EPSILON_DECAY))

            # --- F. TRAIN THE NEURAL NETWORK (BELLMAN EQUATION) ---
            if len(memory) > BATCH_SIZE:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = memory.sample(BATCH_SIZE)

                # 1. What does the brain think right now?
                current_q_vals, _ = q_network(b_obs)
                
                # 2. What will the brain think in the next frame?
                with torch.no_grad():
                    next_q_vals, _ = q_network(b_next_obs)

                total_loss = 0

                # 3. Calculate the error (loss) for all 4 Action Branches separately
                for branch_idx in range(4):
                    # Get the Q-values for the specific actions the robot actually took
                    branch_actions = b_actions[:, :, branch_idx].unsqueeze(-1)
                    current_q = current_q_vals[branch_idx].gather(2, branch_actions).squeeze(-1)
                    
                    # Bellman Equation Math: Reward + (Gamma * Max Future Q)
                    max_next_q = next_q_vals[branch_idx].max(dim=2)[0]
                    target_q = b_rewards + GAMMA * max_next_q * (1 - b_dones)

                    # Compare what it thought with what actually happened
                    total_loss += F.mse_loss(current_q, target_q)

                # 4. Backpropagation! Adjust the weights to make the brain smarter.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Print updates
            if step % 500 == 0 and step > 0:
                print(f"Step: {step} | Mem: {len(memory)} | Epsilon: {epsilon:.2f} | Loss: {total_loss.item():.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
        torch.save(q_network.state_dict(), "drqn_warehouse_model.pth")
        print("Model saved as 'drqn_warehouse_model.pth'.")
    finally:
        env.close()

if __name__ == '__main__':
    train()