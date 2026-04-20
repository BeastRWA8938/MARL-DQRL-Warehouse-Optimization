import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime # 🚀 NEW: Import time tracking

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
MAX_EPISODES = 500000
BATCH_SIZE = 32
GAMMA = 0.99           
LR = 0.001             
EPSILON_START = 1.0    
EPSILON_MIN = 0.05     
# ❌ Delete EPSILON_DECAY completely!

# 🚀 NEW: Generate a unique signature for this entire training run
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
RUN_NAME = f"MARL_B{BATCH_SIZE}_G{GAMMA}_LR{LR}_E{EPSILON_START}-{EPSILON_MIN}_{current_time}"
print(f"🏷️ Run Signature Generated: {RUN_NAME}")


# --- INITIALIZATION ---
print("Booting up CTDE Training Engine...")
env = UnityEnvironment(file_name=None, seed=42, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]

# Create a writer. It will automatically create a 'runs' folder in your directory
writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}") # 🚀 Sync TensorBoard name
print(f"📊 TensorBoard initialized! Logging to /runs/{RUN_NAME}/...")

# Init Brain, Memory, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DRQN(input_size=13, hidden_size=128, output_size=5).to(device)
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
    
    num_agents = len(decision_steps)
    if num_agents == 0:
        continue # Safety catch in case Unity needs a frame to wake up

    agent_rewards = np.zeros(num_agents) 
    step_count = 0 
    
    # 🚀 FIX 1: Initialize LSTM Memory for ALL agents at once! Shape: (1, 3, 128)
    hidden_state = model.init_hidden(batch_size=num_agents, device=device)
    
    # 🚀 FIX 2: Create a separate DVR timeline for each agent's sequence
    agent_episodes = [[] for _ in range(num_agents)]
    
    done = False
    
    while not done:
        step_count += 1
        
        # 🚀 FIX 3: Grab the observation matrix for ALL agents. Shape: (3, 11)
        states = decision_steps.obs[0] 
        
        # --- ACTION SELECTION (BATCHED) ---
        if np.random.rand() < epsilon:
            # Generate a random action for EVERY agent. Shape: (3, 1)
            actions = np.random.randint(0, 5, size=(num_agents, 1))
        else:
            # Pass all 3 agents through the Neural Network at the exact same time!
            state_tensor = torch.FloatTensor(states).to(device)
            with torch.no_grad():
                q_values, hidden_state = model(state_tensor, hidden_state)
            # Pick the best action for each agent
            actions = torch.argmax(q_values, dim=1).cpu().numpy().reshape(num_agents, 1)

        # Send the batch of 3 actions back to Unity
        action_tuple = ActionTuple(discrete=actions)
        env.set_actions(behavior_name, action_tuple)
        env.step() 

        # --- OBSERVE RESULTS ---
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        next_states = np.zeros((num_agents, 13))
        step_rewards = np.zeros(num_agents)

        # Map the new data securely to the correct agent ID
        for agent_id in range(num_agents):
            if agent_id in terminal_steps.agent_id:
                idx = list(terminal_steps.agent_id).index(agent_id)
                next_states[agent_id] = terminal_steps.obs[0][idx]
                step_rewards[agent_id] = terminal_steps.reward[idx]
                done = True # If ANY agent delivers or crashes, we end the episode
            elif agent_id in decision_steps.agent_id:
                idx = list(decision_steps.agent_id).index(agent_id)
                next_states[agent_id] = decision_steps.obs[0][idx]
                step_rewards[agent_id] = decision_steps.reward[idx]
            
            # Track total for TensorBoard
            agent_rewards[agent_id] += step_rewards[agent_id]

        # 🚀 FIX 4: Save the individual frames to each agent's specific timeline
        for i in range(num_agents):
            # We bypass the buffer.store_transition method and build the sequences manually
            agent_episodes[i].append((states[i], actions[i][0], step_rewards[i], next_states[i], done))

    # 🚀 FIX 5: The episode is over. Push all completed timelines into the central CTDE Vault!
    for i in range(num_agents):
        buffer.buffer.append(agent_episodes[i])

    # 🚀 NEW: Dynamic 80% Linear Epsilon Schedule
    decay_cutoff = int(MAX_EPISODES * 0.8) # Find the exact 80% mark
    
    if episode < decay_cutoff:
        # Linearly interpolate between Start and Min based on current progress
        progress = episode / decay_cutoff
        epsilon = EPSILON_START - (progress * (EPSILON_START - EPSILON_MIN))
    else:
        # Lock to minimum for the final 20% (The Exploitation Phase)
        epsilon = EPSILON_MIN
    # ---------------------------------------------------------
    # [KEEP YOUR EXISTING NEURAL NETWORK BACKPROPAGATION CODE HERE]
    # (The block starting with: current_loss = 0.0, if len(buffer) > BATCH_SIZE...)
    # ---------------------------------------------------------    
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
    
# 🚀 FIX: Auto-Save using the dynamic Run Name
    if episode > 0 and episode % 50 == 0:
        save_checkpoint(model, optimizer, epsilon, episode, filename=f"{RUN_NAME}_backup_{episode}.pth")
        
    if episode == MAX_EPISODES - 1:
        save_checkpoint(model, optimizer, epsilon, episode, filename=f"{RUN_NAME}_FINAL.pth")

env.close()
writer.close() # Always close the writer when done