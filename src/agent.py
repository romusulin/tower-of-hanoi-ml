import os
import random
import numpy as np


class QLearningAgent:
    def __init__(self, table_row_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.table_row_size = table_row_size
        # Learning rate
        self.alpha = alpha
        # Discount factor
        self.gamma = gamma
        # Exploration rate
        self.epsilon = epsilon
        self.skipExploration = os.environ.get('SKIP_EXPLORATION', 0)

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.table_row_size)
        return self.q_table[state][action]

    def set_q_value(self, state, action, value):
        # Set the Q-value for a state-action pair
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.table_row_size)
        self.q_table[state][action] = value

    def choose_action(self, state, actions):
        if self.skipExploration and random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            q_values = [self.get_q_value(state, a) for a in range(len(actions))]
            max_q = max(q_values)
            best_actions = [actions[i] for i in range(len(actions)) if q_values[i] == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_actions):
        max_next_q = max([self.get_q_value(next_state, a) for a in range(len(next_actions))])
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.set_q_value(state, action, new_q)
