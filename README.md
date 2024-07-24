# Q-Learning Agent for Custom Treasure-Hunt Environment

## Overview

This project involves the development and training of a Q-Learning agent within a custom static treasure-hunt environment. The environment, built using OpenAI Gymnasium, simulates a grid-based treasure hunt where an agent navigates through a maze to reach a treasure while avoiding pitfalls and gaining hints from friendly states. The agent learns to optimize its path through trial and error, guided by a Q-Learning algorithm.

## Project Components

### Custom Environment

- **Agent**: The entity that moves through the environment to find the treasure.
- **Treasure**: The goal state that the agent aims to reach to win the game.
- **Friendly States**: States that provide hints or guidance to the agent.
- **Non-Friendly States**: States that do not offer any additional information or assistance.
- **Hell States**: States that terminate the game and result in a loss.

### Reward Structure

- **Treasure State**: Reward and terminate the game.
- **Hell State**: Negative reward and terminate the game.
- **Friendly States**: Provide positive hints or minor rewards.
- **Non-Friendly States**: No additional rewards or penalties.

### Q-Learning Agent

- Implemented Q-Learning, a model-free reinforcement learning algorithm.
- Trained the agent to learn the optimal policy for navigating the environment.
- Visualized the learning process using a heat map of Q-values to analyze the agent’s behavior and decision-making.
