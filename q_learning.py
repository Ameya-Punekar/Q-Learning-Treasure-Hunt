import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = tuple(next_state)
            total_reward += reward

            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print("Training finished.\n")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")

#working
def visualize_q_table(obstacle_coordinates=[(0, 5), (1, 2), (2, 6), (3, 1), (4, 4), (6, 2)],
                      goal_coordinates=(6, 6),
                      friendly_state_coordinates=[(1,0), (4,0), (2,2), (3,4), (5,3), (5,6), (6,5)],
                      non_friendly_state_coordinates=[(0, 3), (2, 4), (4, 3), (6, 0)],
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):
    
    # Load the Q-table:
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal and obstacle states for visualization:
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for obs in obstacle_coordinates:
                mask[obs] = True

            # Plot the heatmap with masked areas and grid lines:
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9},
                        linewidths=0.5, linecolor='black')  # Add grid lines

            # Denote Goal and Obstacle states:
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            for obs in obstacle_coordinates:
                ax.text(obs[1] + 0.5, obs[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            # Add 'F' for friendly states in the top-right corner of the cell:
            for friendly in friendly_state_coordinates:
                ax.text(friendly[1] + 0.9, friendly[0] + 0.1, 'F', color='black',
                        ha='right', va='top', weight='bold', fontsize=8)

            # Add 'NF' for non-friendly states in the top-right corner of the cell:
            for non_friendly in non_friendly_state_coordinates:
                ax.text(non_friendly[1] + 0.9, non_friendly[0] + 0.1, 'NF', color='black',
                        ha='right', va='top', weight='bold', fontsize=8)

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

