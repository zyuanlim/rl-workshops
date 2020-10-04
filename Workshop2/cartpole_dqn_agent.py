import gym
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from statistics import median, mean
from collections import Counter

env = gym.make("CartPole-v0")
env.reset()


class DQNAgent:
    def __init__(self):
        self.training_data = []
        self.model = None

    def initial_experience(self, score_requirements=50, n_games=1000):
        scores = []
        accepted_scores = []
        for _ in range(n_games):
            observation = env.reset()
            t = 0
            score = 0
            game_memory = []
            for i in range(200):
                t = t + 1
                action = random.randrange(0, env.action_space.n)
                game_memory.append([observation, action])
                observation, reward, done, info = env.step(action)
                score = score + reward
                if done:
                    break
            if score >= score_requirements:
                accepted_scores.append(score)
                self.training_data.extend(game_memory)
            scores.append(score)
        env.close()

        training_data_save = np.array(self.training_data)
        np.save('training_data.npy', training_data_save)

        # some stats here, to further illustrate the neural network magic!
        # print('Average accepted score:', mean(accepted_scores))
        # print('Median score for accepted scores:', median(accepted_scores))
        # print(Counter(accepted_scores))

    ####################### TO-DO #######################
    # Replace None with your code
    def define_model(self, input_size):
        self.model = None

    def train_model(self, batch_size=64, n_epoch=5, lr=1e-3):
        X = torch.tensor([i[0] for i in self.training_data], dtype=torch.float)
        y = torch.tensor([i[1] for i in self.training_data], dtype=torch.long)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.define_model(X.shape[1])
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for e in range(n_epoch):
            running_loss = 0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # else:
            #     print(f"Training loss: {running_loss / len(dataloader)}")
        return self.model


if __name__ == "__main__":
    total_avg_reward = 0
    for run in tqdm(range(30)):
        cartpole_agent = DQNAgent()
        cartpole_agent.initial_experience(n_games=1000)
        model = cartpole_agent.train_model(n_epoch=10, lr=1e-3)
        scores = []
        choices = []
        for episode in range(100):
            score = 0
            game_memory = []
            prev_obs = []
            env.reset()
            done = False
            while not done:
                if len(prev_obs) == 0:
                    action = random.randrange(0, env.action_space.n)
                else:
                    with torch.no_grad():
                        action = np.argmax(model(torch.Tensor(prev_obs).view(1, len(prev_obs)))[0].numpy())
                choices.append(action)
                new_observation, reward, done, info = env.step(action)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score += reward
            scores.append(score)
        avg_score = sum(scores) / len(scores)
        total_avg_reward += avg_score
    print(f"Average reward over 30 runs over 100 episodes: {total_avg_reward / 30}.")
