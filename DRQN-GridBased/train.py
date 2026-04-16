import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter

import os

# Create a secure vault for your brain files
os.makedirs("saved_models", exist_ok=True)

def save_checkpoint(model, optimizer, epsilon, episode, filename="marl_checkpoint.pth"):
    filepath = os.path.join("saved_models", filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'episode': episode
    }
    torch.save(checkpoint, filepath)
    print(f"💾 [SAVE STATE] Checkpoint secured at Episode {episode}!")

def load_checkpoint(model, optimizer, filename="marl_checkpoint.pth"):
    filepath = os.path.join("saved_models", filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"🔄 [LOAD STATE] Neural link restored! Resuming from Episode {start_episode}...")
        return epsilon, start_episode
    else:
        print("⚠️ [WARNING] No prior save state found. Booting fresh brain.")
        return EPSILON_START, 0

# Import the weapons we forged in Phase 4 and 5
from drqn_model import DRQN
from replay_buffer import SequentialReplayBuffer

# --- HYPERPARAMETERS ---
MAX_EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99           # Discount factor for future rewards
LR = 0.001             # Learning Rate
EPSILON_START = 1.0    # 100% random actions to start
EPSILON_MIN = 0.05     # Always keep 5% randomness to prevent getting stuck
EPSILON_DECAY = 0.95   # FAST DECAY: Multiplied every episode

# --- INITIALIZATION ---
print("Booting up CTDE Training Engine...")
env = UnityEnvironment(file_name=None, seed=42, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
# Create a writer. It will automatically create a 'runs' folder in your directory
writer = SummaryWriter(log_dir="runs/MARL_Grid_Experiment_01")
print("📊 TensorBoard initialized! Logging to /runs/...")


# Init Brain, Memory, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DRQN(input_size=11, hidden_size=128, output_size=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
buffer = SequentialReplayBuffer(capacity=5000, sequence_length=8)

# 🚀 NEW: Attempt to load a previous brain before starting
epsilon, start_episode = load_checkpoint(model, optimizer)

epsilon = EPSILON_START

# --- TRAINING LOOP ---
for episode in range(start_episode, MAX_EPISODES):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    # 🚀 NEW: Multi-Agent Tracking
    num_agents = len(decision_steps)
    agent_rewards = np.zeros(num_agents) 
    step_count = 0 
    
    hidden_state = model.init_hidden(batch_size=1, device=device)
    done = False
    
    state = decision_steps.obs[0][0] # Assuming Agent 0 for the main state tracking right now
    
    while not done:
        step_count += 1
        
        # [Epsilon-Greedy Action & Environment Step Code remains exactly the same as Phase 5...]
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 5)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, hidden_state = model(state_tensor, hidden_state)
            action = torch.argmax(q_values).item()

        action_tuple = ActionTuple(discrete=np.array([[action]]))
        env.set_actions(behavior_name, action_tuple)
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        # 🚀 NEW: Track individual rewards dynamically based on who finished
        if len(terminal_steps) > 0: 
            next_state = terminal_steps.obs[0][0]
            # Accumulate rewards for agents that terminated this step
            for agent_id in terminal_steps.agent_id:
                agent_idx = list(terminal_steps.agent_id).index(agent_id)
                agent_rewards[agent_id] += terminal_steps.reward[agent_idx]
            done = True
        else: 
            next_state = decision_steps.obs[0][0]
            # Accumulate rewards for agents still moving
            for agent_id in decision_steps.agent_id:
                agent_idx = list(decision_steps.agent_id).index(agent_id)
                agent_rewards[agent_id] += decision_steps.reward[agent_idx]
        
        buffer.store_transition(state, action, np.sum(agent_rewards), next_state, done) # Store team reward in buffer
        state = next_state 

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # [Neural Network Backprop code remains exactly the same as Phase 5...]
    current_loss = 0.0 # Track loss for TensorBoard
    if len(buffer) > BATCH_SIZE:
        s_batch, a_batch, r_batch, s_prime_batch, d_batch = buffer.sample_batch(BATCH_SIZE)
        s_batch = s_batch.to(device)
        train_hidden = model.init_hidden(batch_size=BATCH_SIZE, device=device)
        q_values, _ = model(s_batch, train_hidden)
        loss = loss_fn(q_values, torch.randn_like(q_values)) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item() # Save loss value

    # 🚀 NEW: TENSORBOARD LOGGING END-OF-EPISODE
    team_total_reward = np.sum(agent_rewards)
    
    # Log Team Metrics
    writer.add_scalar('Reward/Team_Total', team_total_reward, episode)
    writer.add_scalar('Metrics/Episode_Length', step_count, episode)
    writer.add_scalar('Hyperparameters/Epsilon', epsilon, episode)
    
    # Log Loss only if we actually trained this episode
    if current_loss > 0:
        writer.add_scalar('Loss/DRQN_Loss', current_loss, episode)

    # Log Individual Agent Rewards (Dynamically scales to however many agents are in Unity!)
    for i in range(num_agents):
        writer.add_scalar(f'Reward/Individual_Agent_{i}', agent_rewards[i], episode)

    print(f"Episode: {episode} | Team Reward: {team_total_reward:.2f} | Steps: {step_count} | Epsilon: {epsilon:.3f}")
    
    # 🚀 NEW: Auto-Save every 50 episodes AND on the very last episode
    if episode > 0 and episode % 50 == 0:
        save_checkpoint(model, optimizer, epsilon, episode, filename=f"marl_backup_{episode}.pth")
        
    if episode == MAX_EPISODES - 1:
        save_checkpoint(model, optimizer, epsilon, episode, filename="marl_FINAL.pth")

env.close()
writer.close() # Always close the writer when done