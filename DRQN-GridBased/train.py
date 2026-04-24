import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from torch.utils.tensorboard import SummaryWriter
import os
import copy 
from datetime import datetime

os.makedirs("saved_models", exist_ok=True)

def save_checkpoint(model, target_model, optimizer, epsilon, episode, filename="marl_checkpoint.pth"):
    filepath = os.path.join("saved_models", filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'episode': episode
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, target_model, optimizer, filename="marl_checkpoint.pth"):
    filepath = os.path.join("saved_models", filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        target_model.load_state_dict(checkpoint['target_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode']
        print(f"🔄 [LOAD STATE] Neural link restored! Resuming from Episode {start_episode}...")
        return epsilon, start_episode
    else:
        print("⚠️ [WARNING] No prior save state found. Booting fresh brain.")
        return EPSILON_START, 0

from drqn_model import DRQN
from replay_buffer import SequentialReplayBuffer

# --- HYPERPARAMETERS ---
MAX_EPISODES = 10000
BATCH_SIZE = 32
GAMMA = 0.99           
LR = 0.001             
EPSILON_START = 0.05    
EPSILON_MIN = 0.05     
TARGET_UPDATE_FREQ = 10 

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
RUN_NAME = f"FINAL_DDQN_B{BATCH_SIZE}_G{GAMMA}_LR{LR}_E{EPSILON_START}-{EPSILON_MIN}_{current_time}"
print(f"🏷️ Run Signature Generated: {RUN_NAME}")

print("Booting up Stabilized Single-Agent DDQN Engine...")
env = UnityEnvironment(file_name=None, seed=42, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]

writer = SummaryWriter(log_dir=f"runs/{RUN_NAME}") 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DRQN(input_size=13, hidden_size=128, output_size=5).to(device)
target_model = copy.deepcopy(model).to(device)
target_model.eval() 

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
buffer = SequentialReplayBuffer(capacity=5000, sequence_length=8)

# ⚠️ FORCING A FRESH START
epsilon, start_episode = load_checkpoint(model, target_model, optimizer, filename=r"FINAL_DDQN_B32_G0.99_LR0.001_E1.0-0.05_2026-04-24_17-54_FINAL.pth")

epsilon = EPSILON_START
start_episode = 0

# --- TRAINING LOOP ---
for episode in range(start_episode, MAX_EPISODES):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    while len(decision_steps) == 0 and len(terminal_steps) == 0:
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

    state = decision_steps.obs[0][0] 
    hidden_state = model.init_hidden(batch_size=1, device=device)
    
    done = False
    step_count = 0 
    episode_reward = 0.0
    
    while not done:
        step_count += 1
        
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 5)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) 
            with torch.no_grad():
                q_values_seq, hidden_state = model(state_tensor, hidden_state)
            
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            
            # Extract just the last frame to make the decision
            q_values = q_values_seq[:, -1, :]
            action = torch.argmax(q_values, dim=1).item()

        action_tuple = ActionTuple(discrete=np.array([[action]]))
        env.set_actions(behavior_name, action_tuple)
        env.step() 

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        if len(terminal_steps) > 0:
            next_state = terminal_steps.obs[0][0]
            reward = terminal_steps.reward[0]
            done = True
        elif len(decision_steps) > 0:
            next_state = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]
            done = False
        else:
            continue 
            
        episode_reward += reward

        buffer.store_transition(state, action, reward, next_state, done)
        state = next_state

    decay_cutoff = int(MAX_EPISODES * 0.8) 
    if episode < decay_cutoff:
        progress = episode / decay_cutoff
        epsilon = EPSILON_START - (progress * (EPSILON_START - EPSILON_MIN))
    else:
        epsilon = EPSILON_MIN

    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.load_state_dict(model.state_dict())

    current_loss = 0.0 
    if len(buffer.buffer) > BATCH_SIZE: 
        s_batch, a_batch, r_batch, s_prime_batch, d_batch = buffer.sample_batch(BATCH_SIZE)
        s_batch = s_batch.to(device)
        a_batch = a_batch.to(device)
        r_batch = r_batch.to(device)
        s_prime_batch = s_prime_batch.to(device)
        d_batch = d_batch.to(device)

        h1 = model.init_hidden(batch_size=BATCH_SIZE, device=device)
        q_values_seq, _ = model(s_batch, h1) 
        
        # 🚀 Sequence Unrolling: Gather Q-values for ALL 8 frames!
        current_q = q_values_seq.gather(2, a_batch.unsqueeze(2)).squeeze(2) 
        
        with torch.no_grad():
            h_main = model.init_hidden(batch_size=BATCH_SIZE, device=device)
            next_q_main, _ = model(s_prime_batch, h_main)
            best_next_actions = next_q_main.argmax(dim=2).unsqueeze(2) 

            h_target = target_model.init_hidden(batch_size=BATCH_SIZE, device=device)
            next_q_target, _ = target_model(s_prime_batch, h_target) 
            max_next_q = next_q_target.gather(2, best_next_actions).squeeze(2) 
            
            last_dones = d_batch.float()
            
            # 🚀 Apply Bellman over the whole sequence
            target_q = r_batch + GAMMA * max_next_q * (1 - last_dones)

        loss = loss_fn(current_q, target_q) 
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        current_loss = loss.item() 

    writer.add_scalar('Reward/Total_Reward', episode_reward, episode)
    writer.add_scalar('Metrics/Episode_Length', step_count, episode)
    writer.add_scalar('Hyperparameters/Epsilon', epsilon, episode)
    if current_loss > 0:
        writer.add_scalar('Loss/DRQN_Loss', current_loss, episode)

    print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Steps: {step_count} | Epsilon: {epsilon:.3f}")
    
    if episode > 0 and episode % 50 == 0:
        save_checkpoint(model, target_model, optimizer, epsilon, episode, filename=f"{RUN_NAME}_backup.pth")
        
    if episode == MAX_EPISODES - 1:
        save_checkpoint(model, target_model, optimizer, epsilon, episode, filename=f"{RUN_NAME}_FINAL.pth")

env.close()
writer.close()