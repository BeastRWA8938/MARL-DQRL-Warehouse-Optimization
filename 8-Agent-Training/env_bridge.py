from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np

def test_environment():
    print("Waiting for Unity Environment... Please press PLAY in the Unity Editor!")
    env = UnityEnvironment(file_name=None, seed=42)
    env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    print(f"Successfully connected to behavior: {behavior_name}")
    
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    num_agents = len(decision_steps.agent_id)
    print(f"Number of agents detected: {num_agents}")
    
    for step in range(10):
        random_actions = np.random.randint(
            low=0, 
            high=[3, 3, 2, 3], 
            size=(num_agents, 4)
        )
        action_tuple = ActionTuple(discrete=random_actions)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        
        decision_steps, _ = env.get_steps(behavior_name)
        if len(decision_steps.obs) > 0:
             print(f"Step {step}: Agent 0 Observation shape: {decision_steps.obs[0].shape}")

    env.close()
    print("Test complete. Connection closed.")

if __name__ == '__main__':
    test_environment()