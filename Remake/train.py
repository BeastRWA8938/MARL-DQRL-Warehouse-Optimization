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
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from drqn_model import DRQN
from replay_buffer import EpisodicReplayBuffer

if __name__ == '__main__':
    # ==========================================
    # 1. RUN MODE SETTINGS
    # ==========================================
    # Modes: "train" (start fresh), "resume" (continue training), "test" (watch inference)
    MODE = "test" 
    
    # Path to the .pth file you want to load for resuming or testing
    LOAD_MODEL_PATH = "checkpoints/drqn_ep10000_gamma0.99_eps0.10_mem64527_10x10_Grid_Working_But_Reverse_pickup_and_delivery.pth" 

    # ==========================================
    # 2. HYPERPARAMETERS
    # ==========================================
    GAMMA = 0.99
    LR = 1e-4
    BATCH_SIZE = 32
    SEQ_LEN = 10
    TOTAL_EPISODES = 10000
    ROLLOUT_STEPS = 200
    TRAIN_EVERY_STEPS = 4
    TARGET_UPDATE_FREQ = 10 

    # --- Dynamic Epsilon Settings ---
    EPSILON_START = 1.0
    EPSILON_MIN = 0.10
    EPSILON_DECAY_PERCENTAGE = 1.0 # Keep exploration alive across the full run

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
    # env = UnityEnvironment(file_name=None, seed=42) # if you do not have environment compiled
    # !!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!
    if MODE == "test":
        env = UnityEnvironment(file_name=None, seed=42) # if you do not have environment compiled
    else:
        engine_channel = EngineConfigurationChannel()

        print("Launching Headless Unity Environment...")
        env_path = "Build/Warehouse.exe" # Ensure this matches your actual path

        # Pass the channel into the environment
        env = UnityEnvironment(
            file_name=env_path, 
            seed=42, 
            side_channels=[engine_channel], 
            no_graphics=True
        )

        # FORCE Unity to run at 100x speed and uncap the framerate (-1 means no limit)
        engine_channel.set_configuration_parameters(time_scale=100.0, target_frame_rate=-1)

    # !!!!!!!!!!!!!!! END IMPORTANT !!!!!!!!!!!!
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]

    def optimize_model():
        batch = buffer.sample(BATCH_SIZE, SEQ_LEN)
        if not batch:
            return None

        b_states, b_actions, b_rewards, b_next_states, b_dones = [b.to(device) for b in batch]

        curr_q_seq, _ = policy_net(b_states)
        curr_q = curr_q_seq[:, -1, :]
        last_actions = b_actions[:, -1].unsqueeze(-1)
        curr_q_taken = curr_q.gather(1, last_actions).squeeze(-1)

        with torch.no_grad():
            next_q_seq, _ = target_net(b_next_states)
            next_q = next_q_seq[:, -1, :]
            max_next_q = next_q.max(1)[0]

        last_rewards = b_rewards[:, -1]
        last_dones = b_dones[:, -1]
        target_q = last_rewards + GAMMA * max_next_q * (1 - last_dones)

        loss = F.smooth_l1_loss(curr_q_taken, target_q)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
        optimizer.step()

        return loss.item()

    def steps_to_dict(steps):
        return {int(agent_id): idx for idx, agent_id in enumerate(steps.agent_id)}

    try:
        for episode in range(start_episode, TOTAL_EPISODES + 1):
            env.reset()
            buffer.finish_all_active_episodes()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            states_by_agent = {}
            hidden_by_agent = {}
            rewards_by_agent = {}
            steps_by_agent = {}
            terminals_by_agent = {}
            loss_sum = 0
            train_steps = 0
            decision_count = 0

            for idx, agent_id in enumerate(decision_steps.agent_id):
                agent_id = int(agent_id)
                states_by_agent[agent_id] = decision_steps.obs[0][idx]
                hidden_by_agent[agent_id] = None
                rewards_by_agent[agent_id] = 0.0
                steps_by_agent[agent_id] = 0
                terminals_by_agent[agent_id] = 0

            for rollout_step in range(ROLLOUT_STEPS):
                actions_by_agent = {}
                prev_states_by_agent = {}

                for idx, agent_id in enumerate(decision_steps.agent_id):
                    agent_id = int(agent_id)
                    state = decision_steps.obs[0][idx]
                    states_by_agent[agent_id] = state

                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        q_values, new_hidden_state = policy_net(state_tensor, hidden_by_agent.get(agent_id))

                    if random.random() < current_epsilon:
                        action_int = random.randint(0, 3)
                    else:
                        action_int = torch.argmax(q_values[0, -1]).item()

                    hidden_by_agent[agent_id] = new_hidden_state
                    actions_by_agent[agent_id] = action_int
                    prev_states_by_agent[agent_id] = state

                    action_array = np.array([[action_int]], dtype=np.int32)
                    env.set_action_for_agent(behavior_name, agent_id, ActionTuple(discrete=action_array))

                if not actions_by_agent:
                    break

                env.step()
                next_decision_steps, terminal_steps = env.get_steps(behavior_name)
                next_decision_idx = steps_to_dict(next_decision_steps)
                terminal_idx = steps_to_dict(terminal_steps)

                for agent_id, action_int in actions_by_agent.items():
                    if agent_id in terminal_idx:
                        idx = terminal_idx[agent_id]
                        next_state = terminal_steps.obs[0][idx]
                        reward = float(terminal_steps.reward[idx])
                        done = True
                    elif agent_id in next_decision_idx:
                        idx = next_decision_idx[agent_id]
                        next_state = next_decision_steps.obs[0][idx]
                        reward = float(next_decision_steps.reward[idx])
                        done = False
                    else:
                        continue

                    rewards_by_agent[agent_id] = rewards_by_agent.get(agent_id, 0.0) + reward
                    steps_by_agent[agent_id] = steps_by_agent.get(agent_id, 0) + 1

                    if MODE in ["train", "resume"]:
                        buffer.push_transition(agent_id, prev_states_by_agent[agent_id], action_int, reward, next_state, done)
                        decision_count += 1
                        if decision_count % TRAIN_EVERY_STEPS == 0:
                            loss_value = optimize_model()
                            if loss_value is not None:
                                loss_sum += loss_value
                                train_steps += 1

                    if done:
                        terminals_by_agent[agent_id] = terminals_by_agent.get(agent_id, 0) + 1
                        states_by_agent.pop(agent_id, None)
                        hidden_by_agent.pop(agent_id, None)
                    else:
                        states_by_agent[agent_id] = next_state

                for idx, agent_id in enumerate(next_decision_steps.agent_id):
                    agent_id = int(agent_id)
                    if agent_id not in states_by_agent:
                        states_by_agent[agent_id] = next_decision_steps.obs[0][idx]
                        hidden_by_agent[agent_id] = None
                        rewards_by_agent.setdefault(agent_id, 0.0)
                        steps_by_agent.setdefault(agent_id, 0)
                        terminals_by_agent.setdefault(agent_id, 0)

                decision_steps = next_decision_steps

            buffer.finish_all_active_episodes()

            total_reward = sum(rewards_by_agent.values())
            total_steps = sum(steps_by_agent.values())
            avg_loss = loss_sum / train_steps if train_steps > 0 else 0
            agent_summary = " | ".join(
                f"A{agent_id}: R={rewards_by_agent.get(agent_id, 0.0):.1f}, T={terminals_by_agent.get(agent_id, 0)}"
                for agent_id in sorted(rewards_by_agent.keys())
            )
            print(f"Ep {episode:4d} | TotalR: {total_reward:7.2f} | Steps: {total_steps:4d} | Eps: {current_epsilon:.2f} | {agent_summary}")

            if MODE in ["train", "resume"]:
                writer.add_scalar("Training/Total_Reward", total_reward, episode)
                writer.add_scalar("Training/Total_Steps", total_steps, episode)
                writer.add_scalar("Training/Avg_Loss", avg_loss, episode)
                writer.add_scalar("Hyperparameters/Epsilon", current_epsilon, episode)

                for agent_id, reward in rewards_by_agent.items():
                    writer.add_scalar(f"Agents/{agent_id}_Reward", reward, episode)
                    writer.add_scalar(f"Agents/{agent_id}_Terminals", terminals_by_agent.get(agent_id, 0), episode)

                current_epsilon = max(EPSILON_MIN, current_epsilon * EPSILON_DECAY)

                if episode % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                checkpoint_interval = max(1, TOTAL_EPISODES // 10)
                if episode % checkpoint_interval == 0:
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
