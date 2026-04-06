import os
import sys
import argparse
import glob
from datetime import datetime
import questionary # The interactive menu library


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
from torch.utils.tensorboard import SummaryWriter # <--- NEW: Import TensorBoard

SAVE_DIR = "models"
VERSION_NAME = "v2_dense_rewards"

def get_available_models():
    """Scans the models folder for existing brains"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    models = glob.glob(os.path.join(SAVE_DIR, "*.pth"))
    models.sort(key=os.path.getmtime, reverse=True) # Newest first
    return ["None (Start Fresh)"] + models

def interactive_setup():
    """Runs the interactive Up/Down arrow menu if no CLI args are provided"""
    print("\n" + "="*50)
    print("🤖 WAREHOUSE DRQN TRAINING DASHBOARD 🤖")
    print("="*50 + "\n")

    model_choices = get_available_models()
    
    selected_model = questionary.select(
        "Which Brain do you want to load?",
        choices=model_choices
    ).ask()

    # Prompt for key hyper-parameters
    epsilon_start = float(questionary.text("Starting Epsilon (1.0 = 100% Random, 0.05 = Mostly Smart):", default="1.0").ask())
    learning_rate = float(questionary.text("Learning Rate (e.g., 0.0001):", default="0.0001").ask())
    batch_size = int(questionary.text("Batch Size (Movies to study at once):", default="16").ask())
    gamma = float(questionary.text("Gamma (Future Reward Discount 0.0-0.99):", default="0.99").ask())
    max_steps = int(questionary.text("Max Training Steps:", default="100000").ask())

    if selected_model == "None (Start Fresh)":
        selected_model = None

    return selected_model, epsilon_start, learning_rate, batch_size, gamma, max_steps

def parse_cli_args():
    """Handles standard command line arguments like -e 0.5"""
    parser = argparse.ArgumentParser(description="Train the Warehouse DRQN")
    parser.add_argument('-m', '--model', type=str, default=None, help="Path to a .pth file to load")
    parser.add_argument('-e', '--epsilon', type=float, default=1.0, help="Starting Epsilon (0.0 to 1.0)")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help="Gamma value")
    parser.add_argument('-s', '--steps', type=int, default=100000, help="Max Training Steps")
    return parser.parse_args()

# --- THE MAIN TRAINING LOOP ---
def run_training(load_model_path, epsilon_start, learning_rate, batch_size, gamma, max_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Booting up DRQN Training on: {device} 🚀")

    print("Waiting for Unity Environment... Please press PLAY in the Editor!")
    env = UnityEnvironment(file_name=None, seed=42)
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    decision_steps, _ = env.get_steps(behavior_name)
    total_obs_size = sum([obs.shape[1] for obs in decision_steps.obs])
    print(f"Observation size (Vectors + Raycasts): {total_obs_size}")

    q_network = DRQN(input_size=total_obs_size).to(device)
    
    # LOAD THE MODEL
    if load_model_path and os.path.exists(load_model_path):
        q_network.load_state_dict(torch.load(load_model_path))
        print(f"\n✅ SUCCESS: Loaded existing brain from {load_model_path}!")
    else:
        print("\n🌱 Starting with a brand new random brain.")

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # Hardcoded structural parameters
    seq_length = 10
    memory_capacity = 5000
    epsilon_end = 0.05
    epsilon_decay = int(max_steps * 0.8) # Decay over 80% of the training time

    memory = SequentialReplayBuffer(capacity=memory_capacity, sequence_length=seq_length)
    active_episodes = {} 
    # ... your existing setup code ...
    epsilon = epsilon_start

    # --- NEW: START TENSORBOARD WRITER ---
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    log_dir = os.path.join("runs", f"drqn_{VERSION_NAME}_lr{learning_rate}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard started! Logs saving to: {log_dir}")

    print("\n--- Training Loop Started! ---")
    print("Press Ctrl+C to save and quit.\n")
    
    try:
        for step in range(max_steps):
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # A. RECORD OBSERVATIONS
            current_obs = {}
            for agent_id in decision_steps.agent_id:
                agent_idx = decision_steps.agent_id_to_index[agent_id]
                combined_obs = np.concatenate([obs[agent_idx] for obs in decision_steps.obs])
                current_obs[agent_id] = combined_obs
                
                if agent_id not in active_episodes:
                    active_episodes[agent_id] = [] 

            # B. EPSILON-GREEDY ACTION
            actions_to_send = []
            agent_ids_taking_action = []

            for agent_id in decision_steps.agent_id:
                if random.random() < epsilon:
                    act = [random.randint(0,2), random.randint(0,2), random.randint(0,1), random.randint(0,2)]
                else:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(current_obs[agent_id]).unsqueeze(0).unsqueeze(0).to(device)
                        q_vals, _ = q_network(obs_tensor)
                        act = [
                            torch.argmax(q_vals[0]).item(),
                            torch.argmax(q_vals[1]).item(),
                            torch.argmax(q_vals[2]).item(),
                            torch.argmax(q_vals[3]).item() 
                        ]

                actions_to_send.append(act)
                agent_ids_taking_action.append(agent_id)

            if len(actions_to_send) > 0:
                action_tuple = ActionTuple(discrete=np.array(actions_to_send, dtype=np.int32))
                env.set_actions(behavior_name, action_tuple)

            # C. STEP ENVIRONMENT
            env.step()

            # D. GATHER REWARDS
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

            # E. DECAY EPSILON
            epsilon = max(epsilon_end, epsilon_start - step * ((epsilon_start - epsilon_end) / epsilon_decay))

            # F. TRAIN NEURAL NETWORK
            if len(memory) > batch_size:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = memory.sample(batch_size)

                current_q_vals, _ = q_network(b_obs)
                with torch.no_grad():
                    next_q_vals, _ = q_network(b_next_obs)

                total_loss = 0
                for branch_idx in range(4):
                    branch_actions = b_actions[:, :, branch_idx].unsqueeze(-1)
                    current_q = current_q_vals[branch_idx].gather(2, branch_actions).squeeze(-1)
                    max_next_q = next_q_vals[branch_idx].max(dim=2)[0]
                    target_q = b_rewards + gamma * max_next_q * (1 - b_dones)
                    total_loss += F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if step % 500 == 0 and step > 0:
                print(f"Step: {step} | Mem: {len(memory)} | Eps: {epsilon:.2f} | Loss: {total_loss.item():.4f}")
                
                # --- NEW: SEND DATA TO THE WEBPAGE ---
                writer.add_scalar("1_Training/Loss", total_loss.item(), step)
                writer.add_scalar("1_Training/Epsilon", epsilon, step)
                writer.add_scalar("2_System/Memory_Stored", len(memory), step)

    except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving model...")
            timestamp = datetime.now().strftime("%m-%d_%H-%M")
            
            # Build the ultimate parameter-packed filename
            param_string = f"lr{learning_rate}_b{batch_size}_g{gamma}_seq{seq_length}"
            save_filename = f"drqn_{VERSION_NAME}_{param_string}_{timestamp}.pth"
            full_save_path = os.path.join(SAVE_DIR, save_filename)
            
            torch.save(q_network.state_dict(), full_save_path)
            print(f"💾 Model successfully saved at: {full_save_path}")
    finally:
        writer.close()
        env.close()

if __name__ == '__main__':
    # Logic to decide between Interactive Menu or CLI flags
    if len(sys.argv) > 1:
        # User typed something like: python train.py -e 0.5 -b 32
        args = parse_cli_args()
        run_training(args.model, args.epsilon, args.learning_rate, args.batch_size, args.gamma, args.steps)
    else:
        # User just typed: python train.py
        model_path, eps, lr, batch, gamma, steps = interactive_setup()
        run_training(model_path, eps, lr, batch, gamma, steps)