import numpy as np
from padm_env import create_env
import time

def test_agent(env, q_table_path, max_steps=100):
    # Load the trained Q-table
    q_table = np.load(q_table_path)
    
    # Reset the environment
    state, info = env.reset()
    
    total_reward = 0
    step_count = 0
    
    for _ in range(max_steps):
        # Select the best action based on the Q-table
        action = np.argmax(q_table[state[0], state[1]])
        
        # Take a step in the environment
        state, reward, done, info = env.step(action)
        
        # Render the environment
        env.render()
        time.sleep(0.5)  # Add a delay for better visualization
        
        total_reward += reward
        step_count += 1
        
        if done:
            break
    
    print(f"Test finished after {step_count} steps with total reward: {total_reward}")

def main():
    # Define the coordinates for the goal, obstacles, friendly states, and non-friendly states
    goal_coordinates = (6, 6)
    obstacle_coordinates = [(0, 5), (1, 2), (2, 6), (3, 1), (4, 4), (6, 2)]
    friendly_state_coordinates = [(1,0), (4,0), (2,2), (3,4), (5,3), (5,6), (6,5)]
    non_friendly_state_coordinates = [(0, 3), (2, 4), (4, 3), (6, 0)]

    # Create the environment
    env = create_env(goal_coordinates, obstacle_coordinates, friendly_state_coordinates, non_friendly_state_coordinates)
    
    # Test the agent with the loaded Q-table
    test_agent(env, q_table_path="q_table.npy")

if __name__ == "__main__":
    main()
