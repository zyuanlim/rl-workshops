import gym
import numpy as np
import math
from tqdm import tqdm


class MountainCarQAgent:
    def __init__(self, buckets, num_episodes=1000, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.env = gym.make('MountainCar-v0')

        # [car position, car velocity]
        self.upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        self.lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]

        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize_state(self, obs):
        discretized = []
        for i in range(len(obs)):
            scaling = (obs[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)

    def choose_action(self, state):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    ####################### TO-DO #######################
    # Replace None with your own code (q-learning or sarsa)
    def update_table(self, state, action, reward, new_state, new_action):
        self.q_table[state][action] += None

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        for e in range(self.num_episodes):
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                new_action = self.choose_action(new_state)
                self.update_table(current_state, action, reward, new_state, new_action)
                current_state = new_state
        print('Finished training!')

    def run(self):
        self.train()
        self.epsilon = 0
        env = gym.wrappers.Monitor(self.env, 'mountaincar', force=True)
        t = 0
        done = False
        current_state = self.discretize_state(env.reset())
        while not done:
            env.render()
            t = t+1
            action = self.choose_action(current_state)
            obs, reward, done, _ = env.step(action)
            new_state = self.discretize_state(obs)
            current_state = new_state
        return t

    def run_episodes(self, n_episodes=30):
        wins = 0
        env = gym.wrappers.Monitor(self.env, 'mountaincar', force=True)
        for _ in tqdm(range(n_episodes)):
            self.train()
            self.epsilon = 0
            t = 0
            done = False
            current_state = self.discretize_state(self.env.reset())
            while not done:
                env.render()
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                if done and t < 200:
                    wins += 1
                new_state = self.discretize_state(obs)
                current_state = new_state
        print(f"Total wins over {n_episodes} run: {wins}.")
        return wins


if __name__ == "__main__":
    print("Running Q-learning agent.")
    ####################### TO-DO #######################
    # Replace None with your own code, and change any parameters you want
    # Aim to get at least 3 wins over 10 runs
    agent = MountainCarQAgent(buckets=None)
    agent.run_episodes(10)
