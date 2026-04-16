from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

# 1. Connect to the Unity Editor (file_name=None means it waits for you to press Play in Unity)
print("Waiting for Unity Editor. Please press PLAY in Unity...")
env = UnityEnvironment(file_name=None, seed=42, side_channels=[])
env.reset()

# 2. Get the Behavior Name (e.g., 'GridAgent?team=0')
behavior_name = list(env.behavior_specs.keys())[0]
print(f"✅ Connection Established! Behavior Name: {behavior_name}")

# 3. Read the State
decision_steps, terminal_steps = env.get_steps(behavior_name)
num_agents = len(decision_steps)
print(f"🤖 Agents Detected in Warehouse: {num_agents}")

# Look at the observation of the very first agent
obs = decision_steps.obs[0][0] # First observation branch, first agent
print(f"👀 Observation Shape: {obs.shape}")
print(f"📊 Raw Observation Data: {obs}")

# Close the environment cleanly
env.close()