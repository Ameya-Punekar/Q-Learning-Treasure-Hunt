from padm_env import create_env
from q_learning import train_q_learning, visualize_q_table

def main():
    # Define the coordinates for the goal, obstacles, friendly states, and non-friendly states
    goal_coordinates = (6, 6)
    obstacle_coordinates = [(0, 5), (1, 2), (2, 6), (3, 1), (4, 4), (6, 2)]
    friendly_state_coordinates = [(1,0), (4,0), (2,2), (3,4), (5,3), (5,6), (6,5)]
    non_friendly_state_coordinates = [(0, 3), (2, 4), (4, 3), (6, 0)]

    # Create the environment
    env = create_env(goal_coordinates, obstacle_coordinates, friendly_state_coordinates, non_friendly_state_coordinates)

    # Train the Q-learning agent
    train_q_learning(env,
                     no_episodes=50000,    # Increase the number of episodes for more learning opportunities
                     epsilon=1.0,          # Increase the exploration rate
                     epsilon_min=0.01,
                     epsilon_decay=0.995,  # Slower decay rate for exploration
                     alpha=0.01, 
                     #0.01 #new2,           # Adjust the learning rate for balanced learning
                     gamma=0.99,          # Increase the discount factor to prioritize future rewards
                     q_table_save_path="q_table.npy")

    # Visualize the Q-table
    visualize_q_table(obstacle_coordinates=obstacle_coordinates,
                      goal_coordinates=goal_coordinates,
                      friendly_state_coordinates=friendly_state_coordinates,
                      non_friendly_state_coordinates=non_friendly_state_coordinates,
                      q_values_path="q_table.npy")

if __name__ == "__main__":
    main()