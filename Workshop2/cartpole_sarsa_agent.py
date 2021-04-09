import numpy as np
import math
from cartpole_base_agent import CartPoleAgent


class CartPoleSarsaAgent(CartPoleAgent):
    def update_table(self, state, action, reward, new_state, new_action):
        self.q_table[state][action] += self.learning_rate * (reward + self.discount * (self.q_table[new_state][new_action]) - self.q_table[state][action])


if __name__ == "__main__":
    print("Running Sarsa agent.")
    agent = CartPoleSarsaAgent()
    agent.train()
    t = agent.run()
    print("Time", t)
    agent.run_episodes()
