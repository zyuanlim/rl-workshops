import gym
import numpy as np
import math
from cartpole_base_agent import CartPoleAgent


####################### TO-DO #######################
# Replace None with your code
class CartPoleQAgent(CartPoleAgent):
    def update_table(self, state, action, reward, new_state, new_action):
        self.q_table[state][action] += None


if __name__ == "__main__":
    print("Running Q-learning agent.")
    agent = CartPoleQAgent()
    agent.train()
    t = agent.run()
    print("Time", t)
    agent.run_episodes()
