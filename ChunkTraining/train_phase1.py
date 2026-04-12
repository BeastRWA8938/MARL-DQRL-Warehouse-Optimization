import os
import sys
import argparse
import glob
from datetime import datetime
import questionary 

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
from torch.utils.tensorboard import SummaryWriter

SAVE_DIR = "models"
VERSION_NAME = "Phase1_Toddler"

# --- INTERACTIVE DASHBOARD & CLI ---

def get_available_models():
    """Scans the models folder for existing brains"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    models = glob.glob(os.path.join(SAVE_DIR, "*.pth"))
    models.sort(key=os.path.getmtime, reverse=True) 
    return ["None (Start Fresh)"] + models

def interactive_setup():
    """Runs the interactive menu if no CLI args are provided"""
    print("\n" + "="*50)
    print("👶 WAREHOUSE DRQN: PHASE 1 DASHBOARD 👶")
    print("="*50 + "\n")

    model_choices = get_available_models()
    
    selected_model = questionary.select(
        "Which Brain do you want to load?",
        choices=model_choices
    ).ask()

    epsilon_start = float(questionary.text("Starting Epsilon (1.0 = 100% Random, 0.05 = Mostly Smart):", default="1.0").ask())
    learning_rate = float(questionary.text("Learning Rate (e.g., 0.0001):", default="0.0001").ask())
    batch_size = int(questionary.text("Batch Size (Movies to study at once):", default="16").ask())
    gamma = float(questionary.text("Gamma (Future Reward Discount 0.0-0.99):", default="0.99").ask())
    max_steps = int(questionary.text("Max Training Steps:", default="50000").ask())

    if selected_model == "None (Start Fresh)":
        selected_model = None

    return selected_model, epsilon_start, learning_rate, batch_size, gamma, max_steps

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train the Phase 1 DRQN")
    parser.add_argument('-m', '--model', type=str, default=None, help="Path to a .pth file to load")
    parser.add_argument('-e', '--epsilon', type=float, default=1.0, help="Starting Epsilon (0.0 to 1.0)")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help="Learning Rate")
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help="Gamma value")
    parser.add_argument('-s', '--steps', type=int, default=50000, help="Max Training Steps")
    return parser.parse_args()


# --- THE MAIN TRAINING LOOP ---

def run_phase1(load_model_path, epsilon_start, learning_rate, batch_size, gamma, max_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Booting Phase 1 DRQN on: {device} 🚀")

    # Fixed the Timeout bug! Extended wait time and added the print statement.
    print("Waiting for Unity Environment... Please press PLAY in the Editor!")
    env = UnityEnvironment(file_name=None, seed=42, timeout_wait=120)
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    decision_steps, _ = env.get_steps(behavior_name)
    
    # MAGIC FIX: Hide the 5 human keys BEFORE building the brain
    total_obs_size = sum([obs.shape[1] for obs in decision_steps.obs]) - 5
    print(f"True PyTorch Observation size: {total_obs_size}")

    q_network = DRQN(input_size=total_obs_size).to(device)
    
    # --- NEW: LOAD THE MODEL IF REQUESTED ---
    if load_model_path and os.path.exists(load_model_path):
        q_network.load_state_dict(torch.load(load_model_path))
        print(f"\n✅ SUCCESS: Loaded existing brain from {load_model_path}!")
    else:
        print("\n🌱 Starting with a brand new random brain.")

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    seq_length = 20
    burn_in = 10
    memory = SequentialReplayBuffer(capacity=5000, sequence_length=seq_length, burn_in=burn_in)
    active_episodes = {} 
    
    hidden_states = {} 
    
    epsilon = epsilon_start
    epsilon_end = 0.05
    epsilon_decay = int(max_steps * 0.8)
    
    total_loss = torch.tensor(0.0) 

    # START TENSORBOARD WRITER
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    log_dir = os.path.join("runs", f"drqn_{VERSION_NAME}_lr{learning_rate}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard started! Logs saving to: {log_dir}")
    print("Press Ctrl+C to save and quit.\n")

    try:
        for step in range(max_steps):
            
            # CLEAR THE CHAMBER
            actions_to_send = []
            agent_ids_taking_action = []
            current_obs = {}
            human_overrides = {} 
            
            # --- A. OBSERVE ---
            for agent_id in decision_steps.agent_id:
                agent_idx = decision_steps.agent_id_to_index[agent_id]
                raw_obs = np.concatenate([obs[agent_idx] for obs in decision_steps.obs])
                
                current_obs[agent_id] = raw_obs[:-5]   # True Vision
                human_overrides[agent_id] = raw_obs[-5:] # Human Input
                
                if agent_id not in active_episodes:
                    active_episodes[agent_id] = [] 

            # --- B. ACT ---
            for agent_id in decision_steps.agent_id:
                human_data = human_overrides[agent_id]
                
                if human_data[0] == 1.0: # Human Override
                    act = [int(human_data[1]), int(human_data[2]), int(human_data[3]), int(human_data[4])]
                elif random.random() < epsilon: # Explore
                    act = [random.randint(0,2), random.randint(0,2), random.randint(0,1), random.randint(0,2)]
                else: # Exploit
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(current_obs[agent_id]).unsqueeze(0).unsqueeze(0).to(device)
                        if agent_id not in hidden_states:
                            hidden_states[agent_id] = q_network.init_hidden(1, device)

                        q_vals, hidden_states[agent_id] = q_network(obs_tensor, hidden_states[agent_id])

                        # Detach hidden state to prevent backprop explosion
                        h, c = hidden_states[agent_id]
                        hidden_states[agent_id] = (h.detach(), c.detach())
                        act = [torch.argmax(q[0]).item() for q in q_vals]

                actions_to_send.append(act)
                agent_ids_taking_action.append(agent_id)

            if len(actions_to_send) > 0:
                action_tuple = ActionTuple(discrete=np.array(actions_to_send, dtype=np.int32))
                env.set_actions(behavior_name, action_tuple)

            # --- C. STEP ---
            env.step()
            new_decision_steps, new_terminal_steps = env.get_steps(behavior_name)

            # --- D. REWARD & SAVE ---
            for idx, agent_id in enumerate(agent_ids_taking_action):
                if agent_id in new_decision_steps.agent_id:
                    agent_idx = new_decision_steps.agent_id_to_index[agent_id]
                    next_o = np.concatenate([obs[agent_idx] for obs in new_decision_steps.obs])[:-5]
                    reward = new_decision_steps.reward[agent_idx]
                    active_episodes[agent_id].append((current_obs[agent_id], actions_to_send[idx], reward, next_o, False))

            for agent_id in new_terminal_steps.agent_id:
                if agent_id in current_obs and agent_id in agent_ids_taking_action:
                    idx = agent_ids_taking_action.index(agent_id)
                    agent_idx = new_terminal_steps.agent_id_to_index[agent_id]
                    next_o = np.concatenate([obs[agent_idx] for obs in new_terminal_steps.obs])[:-5]
                    reward = new_terminal_steps.reward[agent_idx]
                    
                    writer.add_scalar(f"Agent_Score", reward, step)
                    
                    active_episodes[agent_id].append((current_obs[agent_id], actions_to_send[idx], reward, next_o, True))
                    memory.push_episode(active_episodes[agent_id])
                    active_episodes[agent_id] = []
                    hidden_states[agent_id] = q_network.init_hidden(1, device)

            # --- E. TRAIN ---
            epsilon = max(epsilon_end, epsilon_start - step * ((epsilon_start - epsilon_end) / epsilon_decay))

            if len(memory) > batch_size:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones, b_burn = memory.sample(batch_size)
                # Initialize hidden state for batch
                hidden = q_network.init_hidden(batch_size, device)

                # Burn-in phase (no gradients)
                with torch.no_grad():
                    _, hidden = q_network(b_burn, hidden)

                # Training phase
                current_q_vals, _ = q_network(b_obs, hidden)
                with torch.no_grad():
                    next_q_vals, _ = q_network(b_next_obs, hidden)

                total_loss = 0
                for branch_idx in range(4):
                    branch_actions = b_actions[:, :, branch_idx].unsqueeze(-1)
                    current_q = current_q_vals[branch_idx].gather(2, branch_actions).squeeze(-1)
                    max_next_q = next_q_vals[branch_idx].max(dim=2)[0]
                    target_q = b_rewards + gamma * max_next_q * (1 - b_dones)
                    total_loss += F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 5.0)
                optimizer.step()

            if step % 500 == 0 and step > 0:
                print(f"Step: {step} | Mem: {len(memory)} | Eps: {epsilon:.2f} | Loss: {total_loss.item():.4f}")
                writer.add_scalar("Training/Loss", total_loss.item(), step)
                writer.add_scalar("Training/Epsilon", epsilon, step)

            # THE MISSING LINK: Pass baton to next frame
            decision_steps = new_decision_steps

    except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving Phase 1 Model...")
            timestamp = datetime.now().strftime("%m-%d_%H-%M")
            param_string = f"lr{learning_rate}_b{batch_size}_g{gamma}"
            save_filename = f"{VERSION_NAME}_{param_string}_{timestamp}.pth"
            full_save_path = os.path.join(SAVE_DIR, save_filename)
            
            torch.save(q_network.state_dict(), full_save_path)
            print(f"💾 Model successfully saved at: {full_save_path}")
    finally:
        writer.close()
        env.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = parse_cli_args()
        run_phase1(args.model, args.epsilon, args.learning_rate, args.batch_size, args.gamma, args.steps)
    else:
        model_path, eps, lr, batch, gamma, steps = interactive_setup()
        run_phase1(model_path, eps, lr, batch, gamma, steps)