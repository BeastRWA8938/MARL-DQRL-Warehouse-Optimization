import os
import random
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from drqn_model import DRQN
from replay_buffer import EpisodicReplayBuffer

if __name__ == '__main__':
    # ==========================================
    # 1. RUN MODE SETTINGS
    # ==========================================
    # Modes: "train" (start fresh), "resume" (continue training), "test" (watch inference)
    MODE = "train" 
    
    # Path to the .pth file you want to load for resuming or testing
    LOAD_MODEL_PATH = "checkpoints/drqn_ep10000_gamma0.99_eps0.05_mem40000.pth" 

    # ==========================================
    # 2. HYPERPARAMETERS
    # ==========================================
    GAMMA = 0.99
    LR = 1e-5
    BATCH_SIZE = 32
    SEQ_LEN = 10
    TOTAL_EPISODES = 300000
    TARGET_UPDATE_FREQ = 10 

    # --- Dynamic Epsilon Settings ---
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY_PERCENTAGE = 0.9 # Epsilon hits minimum exactly at 80% of total episodes

    # Calculate exact exponential decay factor to hit minimum at the target episode
    decay_target_episodes = TOTAL_EPISODES * EPSILON_DECAY_PERCENTAGE
    EPSILON_DECAY = (EPSILON_MIN / EPSILON_START) ** (1.0 / decay_target_episodes)
    
    # ==========================================
    # 3. ENVIRONMENT & COMPUTE SETUP
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device} | Mode: {MODE.upper()}")
    os.makedirs("checkpoints", exist_ok=True)
    
    policy_net = DRQN().to(device)
    target_net = DRQN().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = EpisodicReplayBuffer(capacity=2000)

    # Variables that might be overridden by resuming
    start_episode = 1
    current_epsilon = EPSILON_START

    # --- Model Loading Logic ---
    if MODE == "resume" or MODE == "test":
        if os.path.exists(LOAD_MODEL_PATH):
            print(f"Loading Model: {LOAD_MODEL_PATH}")
            checkpoint = torch.load(LOAD_MODEL_PATH, map_location=device)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            
            if MODE == "resume":
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_episode = checkpoint['episode'] + 1
                current_epsilon = checkpoint['epsilon']
                print(f"Resuming training from Episode {start_episode}")
            elif MODE == "test":
                policy_net.eval() # Set network to evaluation mode
                current_epsilon = 0.0 # Force 100% greedy actions (no random exploration)
                print("Testing mode: Epsilon forced to 0.0")
        else:
            print(f"ERROR: Could not find checkpoint file at {LOAD_MODEL_PATH}")
            exit()

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() 

    # --- TensorBoard Setup (Only if Training) ---
    writer = None
    if MODE in ["train", "resume"]:
        run_name = f"DRQN_{MODE}_gamma{GAMMA}_{datetime.now().strftime('%m%d-%H%M')}"
        writer = SummaryWriter(f"runs/{run_name}")

    # ==========================================
    # 4. EXECUTION LOOP
    # ==========================================
    print("Waiting for Unity Environment... Please press PLAY in the Unity Editor or wait for file to load.")
    # !!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!
    # env = UnityEnvironment(file_name=None, seed=42) # if you do not have environment compiled
    # if you have your environment built then do the below
    # env_path
    env_path = "Build/Warehouse.exe"
    env = UnityEnvironment(file_name=env_path, seed=42, no_graphics=True)
    # handle the above env first before running.
    # !!!!!!!!!!!!!!! END IMPORTANT !!!!!!!!!!!!
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    try:
        # Loop accounts for starting mid-way through resumed training
        for episode in range(start_episode, TOTAL_EPISODES + 1):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            tracked_agent = decision_steps.agent_id[0]
            
            state = decision_steps.obs[0][0]
            hidden_state = None 
            done = False
            episode_reward = 0
            step_count = 0
            loss_sum = 0
            train_steps = 0

            while not done:
                # --- ACTION SELECTION ---
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values, new_hidden_state = policy_net(state_tensor, hidden_state)
                
                if random.random() < current_epsilon:
                    action_int = random.randint(0, 2)
                else:
                    action_int = torch.argmax(q_values).item()
                
                hidden_state = new_hidden_state

                # --- STEP ENVIRONMENT ---
                action_array = np.array([[action_int]], dtype=np.int32)
                env.set_action_for_agent(behavior_name, tracked_agent, ActionTuple(discrete=action_array))
                env.step()
                
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                if tracked_agent in terminal_steps:
                    next_state = terminal_steps.obs[0][0]
                    reward = terminal_steps.reward[0]
                    done = True
                else:
                    next_state = decision_steps.obs[0][0]
                    reward = decision_steps.reward[0]
                
                episode_reward += reward
                step_count += 1

                # --- TRAINING LOGIC (Skipped in Test Mode) ---
                if MODE in ["train", "resume"]:
                    buffer.push_transition(state, action_int, reward, next_state, done)
                    
                    batch = buffer.sample(BATCH_SIZE, SEQ_LEN)
                    if batch:
                        b_states, b_actions, b_rewards, b_next_states, b_dones = [b.to(device) for b in batch]
                        
                        curr_q, _ = policy_net(b_states)
                        last_actions = b_actions[:, -1].unsqueeze(-1)
                        curr_q_taken = curr_q.gather(1, last_actions).squeeze(-1)
                        
                        with torch.no_grad():
                            next_q, _ = target_net(b_next_states)
                            max_next_q = next_q.max(1)[0]
                            
                        last_rewards = b_rewards[:, -1]
                        last_dones = b_dones[:, -1]
                        target_q = last_rewards + GAMMA * max_next_q * (1 - last_dones)
                        
                        loss = F.smooth_l1_loss(curr_q_taken, target_q)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        loss_sum += loss.item()
                        train_steps += 1
                
                state = next_state

            # --- END OF EPISODE ---
            print(f"Ep {episode:4d} | Reward: {episode_reward:6.2f} | Steps: {step_count:3d} | Epsilon: {current_epsilon:.2f}")

            if MODE in ["train", "resume"]:
                # Log to Tensorboard
                avg_loss = loss_sum / train_steps if train_steps > 0 else 0
                writer.add_scalar("Training/Episode_Reward", episode_reward, episode)
                writer.add_scalar("Training/Episode_Length", step_count, episode)
                writer.add_scalar("Training/Avg_Loss", avg_loss, episode)
                writer.add_scalar("Hyperparameters/Epsilon", current_epsilon, episode)
                
                # Epsilon Decay Logic
                current_epsilon = max(EPSILON_MIN, current_epsilon * EPSILON_DECAY)

                # Target Network Update
                if episode % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Dynamic Checkpointing
                if episode % 10000 == 0:
                    model_name = f"drqn_ep{episode}_gamma{GAMMA}_eps{current_epsilon:.2f}_mem{buffer.total_frames_stored}.pth"
                    save_path = os.path.join("checkpoints", model_name)
                    torch.save({
                        'episode': episode,
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epsilon': current_epsilon
                    }, save_path)
                    print(f"--> Saved Checkpoint: {model_name}")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Closing environment.")
    finally:
        env.close()
        if writer: writer.close()