import gymnasium as gym
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os

class PadmEnv(gym.Env):
    """
    Custom Environment for a Treasure Hunt game using OpenAI Gymnasium framework.
    The agent's goal is to reach the treasure while avoiding obstacles.
    """

    def __init__(self, grid_size=7):
        """
        Initialize the environment.

        Parameters:
        grid_size (int): The size of the grid (grid_size x grid_size).
        """
        super(PadmEnv, self).__init__()
        self.grid_size = grid_size
        self.state = None
        self.reward = 0
        self.info = {}
        self.treasure = np.array([6, 6])
        self.done = False
        self.obstacles = []
        self.friendly_states = []
        self.non_friendly_states = []
        self.visited_friendly_states = set()
        self.visited_non_friendly_states = set()

        # Define action space: 0 = up, 1 = down, 2 = right, 3 = left
        self.action_space = gym.spaces.Discrete(4)

        # Define observation space: position of the agent in the grid
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize Tkinter window for rendering
        self.root = tk.Tk()
        self.root.title("Padm Environment")
        self.canvas = tk.Canvas(self.root, width=self.grid_size * 100, height=self.grid_size * 100, bg="white")
        self.canvas.pack()

        # Load and resize the images
        current_dir = os.getcwd()
        self.images = {
            "treasure": ImageTk.PhotoImage(Image.open(os.path.join(current_dir, "treasure.png")).resize((80, 80), Image.LANCZOS)),
            "obstacle": ImageTk.PhotoImage(Image.open(os.path.join(current_dir, "guard.png")).resize((80, 80), Image.LANCZOS)),
            "explorer": ImageTk.PhotoImage(Image.open(os.path.join(current_dir, "prince.png")).resize((80, 80), Image.LANCZOS)),
            "friendly": ImageTk.PhotoImage(Image.open(os.path.join(current_dir, "hint.png")).resize((80, 80), Image.LANCZOS)),
            "non_friendly": ImageTk.PhotoImage(Image.open(os.path.join(current_dir, "trap.png")).resize((80, 80), Image.LANCZOS))
        }

        # Load background image
        try:
            self.bg_image = Image.open(os.path.join(current_dir, "background.jpg"))
            self.bg_image = self.bg_image.resize((self.grid_size * 100, self.grid_size * 100), Image.LANCZOS)
            self.background = ImageTk.PhotoImage(self.bg_image)
        except Exception as e:
            print("Error loading background image:", e)

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
        tuple: The initial state and the initial distance to the treasure.
        """
        #starting_positions = [(4, 1)]
        #self.state = np.array(starting_positions[np.random.choice(len(starting_positions))])
        self.state = np.array([0, 0])  # Starting position of the explorer
        self.reward = 0
        self.done = False
        self.info["Distance to treasure"] = np.sqrt((self.state[0] - self.treasure[0]) ** 2 + (self.state[1] - self.treasure[1]) ** 2)

        # Reset visited states
        self.visited_friendly_states = set()
        self.visited_non_friendly_states = set()

        return self.state, self.info

    def add_obstacles(self, obstacle_coordinates):
        """
        Add obstacles to the environment.

        Parameters:
        obstacle_coordinates (tuple): Coordinates of the obstacle to be added.
        """
        self.obstacles.append(np.array(obstacle_coordinates))

    def add_friendly_states(self, friendly_coordinates):
        """
        Add friendly states to the environment.

        Parameters:
        friendly_coordinates (tuple): Coordinates of the friendly state to be added.
        """
        self.friendly_states.append(np.array(friendly_coordinates))

    def add_non_friendly_states(self, non_friendly_coordinates):
        """
        Add non-friendly states to the environment.

        Parameters:
        non_friendly_coordinates (tuple): Coordinates of the non-friendly state to be added.
        """
        self.non_friendly_states.append(np.array(non_friendly_coordinates))

    def step(self, action):
        """
        Take a step in the environment based on the chosen action.

        Parameters:
        action (int): The action to be taken (c).

        Returns:
        tuple: The new state, the reward, whether the episode is done, and additional info.
        """
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1  # Move up
        if action == 1 and self.state[0] < self.grid_size - 1:
            self.state[0] += 1  # Move down
        if action == 2 and self.state[1] < self.grid_size - 1:
            self.state[1] += 1  # Move right
        if action == 3 and self.state[1] > 0:
            self.state[1] -= 1  # Move left

        if np.array_equal(self.state, self.treasure):
            self.reward += 100  # Found the treasure
            self.done = True
        elif any(np.array_equal(self.state, obstacle) for obstacle in self.obstacles):
            self.reward -= 50  # Encountered an obstacle
            self.done = True
        elif not np.array_str(self.state) in self.visited_friendly_states and any(np.array_equal(self.state, friendly) for friendly in self.friendly_states):
            self.reward += 5  # Entered a friendly state
            self.visited_friendly_states.add(np.array_str(self.state))
            self.done = False
        elif not np.array_str(self.state) in self.visited_non_friendly_states and any(np.array_equal(self.state, non_friendly) for non_friendly in self.non_friendly_states):
            self.reward -= 10  # Entered a non-friendly state
            self.visited_non_friendly_states.add(np.array_str(self.state))
            self.done = False
        else:
            self.reward -= 0.05 # Move penalty
            self.done = False

        self.info["Distance to treasure"] = np.sqrt((self.state[0] - self.treasure[0]) ** 2 + (self.state[1] - self.treasure[1]) ** 2)
        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        Render the environment using Tkinter.
        """
        self.canvas.delete("all")

        # Draw background image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.background)

        # Draw images at the center of each cell
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if np.array_equal([y, x], self.treasure):
                    image_type = "treasure"
                elif any(np.array_equal([y, x], obstacle) for obstacle in self.obstacles):
                    image_type = "obstacle"
                elif any(np.array_equal([y, x], friendly) for friendly in self.friendly_states):
                    image_type = "friendly"
                elif any(np.array_equal([y, x], non_friendly) for non_friendly in self.non_friendly_states):
                    image_type = "non_friendly"
                elif np.array_equal([y, x], self.state):
                    image_type = "explorer"
                else:
                    continue  # No image to draw for empty cells

                image = self.images[image_type]
                self.canvas.create_image(x * 100 + 50, y * 100 + 50, image=image)

        # Draw grid lines
        for i in range(self.grid_size + 1):
            self.canvas.create_line(0, i * 100, self.grid_size * 100, i * 100, fill="white", width=1)
            self.canvas.create_line(i * 100, 0, i * 100, self.grid_size * 100, fill="white", width=1)

        self.root.update()

    def close(self):
        """
        Close the Tkinter window.
        """
        self.root.destroy()
        self.root.quit()


def create_env(goal_coordinates, hell_state_coordinates, friendly_state_coordinates, non_friendly_state_coordinates):
    env = PadmEnv(grid_size=7)

    env.treasure = np.array(goal_coordinates)
    for coord in hell_state_coordinates:
        env.add_obstacles(coord)
    for coord in friendly_state_coordinates:
        env.add_friendly_states(coord)
    for coord in non_friendly_state_coordinates:
        env.add_non_friendly_states(coord)

    return env