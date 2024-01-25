import gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from dqn import DQN, ReplayBuffer
import random

MODEL_PATH = './models/checkpoint_ddqn.pt'

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class DLearningNetwork:
    """
        Double Deep Q-learning implementation

        :param gamma: reward discount
        :param batch_size: replay buffer batch sizes
        :param env: gym environment
        :param num_frames: number of frames per state (and number per action)
    """
    def __init__(self, gamma=.95, batch_size=64, env='MsPacmanDeterministic-v4', num_frames=4):
        self.num_frames = num_frames
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.memory = ReplayBuffer(50000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.score_history = []
        self.env = gym.make(env)
        self.start = self.get_start_state()
        _, height, width = self.start.size()
        self.num_actions = self.env.action_space.n
        self.render = lambda : plt.imshow(env.render(mode='rgb_array'))
        self.policy = DQN(height, width, num_frames, self.num_actions).to(self.device)
        self.target = DQN(height,width, num_frames, self.num_actions).to(self.device)
        self.loss = nn.SmoothL1Loss()
        self.opt = torch.optim.RMSprop(self.policy.parameters(), lr=.00025, alpha=.95, eps=0.01)
        self.policy.apply(init_weights)
        self.target.apply(init_weights)


    def get_start_state(self):
        self.lives = 3
        frame = self.preprocess(self.env.reset())
        frames = []
        for i in range(self.num_frames):
            frames.append(frame.clone())
        return torch.cat(frames)

    def preprocess(self, img):
        ds = img[::2,::2]
        grayscale = np.mean(ds, axis=-1).astype(np.uint8)
        return torch.tensor(grayscale).unsqueeze(0)

    def fit_buffer(self, sample):
        policy = self.policy
        target = self.target
        dev = self.device

        states, actions, next_states, rewards, non_terms = list(zip(*sample))
        states = torch.cat(states)
        actions = torch.tensor(actions).long().to(dev)
        next_states = torch.cat(next_states)
        rewards = torch.tensor(rewards).long().to(dev)
        non_terms = torch.tensor(non_terms).to(dev)
        next_mask = torch.ones((actions.size(0), self.num_actions))
        curr_mask = F.one_hot(actions, self.num_actions)

        policy.train()
        target.eval()
        next_Q_vals = policy(next_states.to(dev), next_mask.to(dev))
        next_actions = next_Q_vals.max(1)[1].to(dev)
        next_mask = F.one_hot(next_actions, self.num_actions).to(dev)
        next_Q_vals = target(next_states.to(dev), next_mask)
        next_Q_vals = next_Q_vals.gather(-1, next_actions.unsqueeze(1)).squeeze(-1) * non_terms
        next_Q_vals = (next_Q_vals * self.gamma) + rewards.to(dev)

        expected_Q_vals = policy(states.to(dev), curr_mask.to(dev))
        expected_Q_vals = expected_Q_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1)
        #print(next_Q_vals.size(), expected_Q_vals.size())

        self.opt.zero_grad()
        loss = self.loss(expected_Q_vals, next_Q_vals)
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        return loss.item()

    def get_epsilon_for_iteration(self, iteration):
        return max(.01, 1-(iteration*.99/300000))

    def choose_best_action(self, frames):
        model = self.policy
        dev = self.device
        model.eval()
        actions = model(frames.unsqueeze(0).to(dev),
                        torch.ones(1,self.num_actions).to(dev)).squeeze(0)
        return int(actions.max(0)[1])

    def q_iteration(self, frames, iteration):
        # Choose epsilon based on the iteration
        env = self.env
        epsilon = self.get_epsilon_for_iteration(iteration)

        # Choose the action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = self.choose_best_action(frames)


        is_done = False
        new_frames = []
        total_reward = 0
        total_score = 0
        # Play one game iteration (3 frames):
        for i in range(self.num_frames):
            if not is_done:
                new_frame, reward, is_done, lives = self.env.step(action)
                lives = lives['ale.lives']
                new_frame = self.preprocess(new_frame)
                total_score += reward
                if lives < self.lives:
                    self.lives = lives
                    #total_reward -= 10
                total_reward += reward
            else:
                reward = 0
            new_frames.append(new_frame)
        new_frames = torch.cat(new_frames)

        non_term = 1
        if is_done:
            non_term = 0

        mem = (frames.unsqueeze(0), action,
               new_frames.unsqueeze(0), total_reward, non_term)
        self.memory.push(mem)

        loss = None
        grad = None
        # Sample and fit
        if iteration > 64:
            batch = self.memory.sample(self.batch_size)
            loss= self.fit_buffer(batch)

        return is_done, total_reward, new_frames, loss

    def train(self, epochs=10000, start_iter=0, updates=500):
        self.score_history = []
        self.eps_history = []
        self.loss_hist = []
        self.updates = updates
        self.its_hist = []
        iteration = start_iter
        running_loss = 0
        running_count = 0
        running_score = 0
        running_its = 0
        for e in range(epochs):
            is_done = False
            e_reward = 0
            e_its = 0
            frames = self.get_start_state()
            while not is_done:
                is_done, reward, frames, loss = self.q_iteration(frames, iteration)
                iteration += 1
                e_reward += reward
                e_its += 1
                if loss is not None:
                    running_loss += loss
                    running_count += 1
            running_score += e_reward
            running_its += e_its
            if e%100 == 0:
                era_score = running_score / 100
                era_its = running_its / 100
                self.score_history.append(era_score)
                eps = self.get_epsilon_for_iteration(iteration)
                self.eps_history.append(eps)
                self.loss_hist.append(running_loss / running_count)
                self.its_hist.append(era_its)
                print(f'---> Epoch {e}/{epochs}, Score: {era_score}, eps: {eps}')
                print(f'-------->Loss: {running_loss / running_count}, Its: {era_its}')
                running_loss = 0
                running_count = 0
                running_score = 0
                running_its = 0
            if e%updates == 0:
                torch.save(self.policy.state_dict(), MODEL_PATH)
                self.load_target(MODEL_PATH)

    def load_target(self, path):
        self.target.load_state_dict(torch.load(path))
        self.target.eval()

    def play(self):
        #TODO, update this method
        state = self.get_start_state()
        im = plt.imshow(self.env.render(mode='rgb_array'))
        plt.ion()
        is_done = False
        while not is_done:
            action = self.choose_best_action(state)
            new_state = []
            for i in range(self.num_frames):
                new_frame, reward, is_done, lives = self.env.step(action)
                if is_done:
                    break
                im.set_data(self.env.render(mode='rgb_array'))
                plt.draw()
                plt.pause(.001)
                new_state.append(self.preprocess(new_frame))
            state = torch.cat(new_state)
        plt.show()

    def plot(self):
        fig, ax = plt.subplots()
        plt.title(f'Score During Training - {self.updates} Epoch Updates')
        ax.set_xlabel('100 Epoch')
        ax.set_ylabel('Score')
        ax2 = ax.twinx()
        ax2.set_ylabel('Loss/Epsilon')
        x = range(len(self.score_history))
        lns1 = ax.plot(x, self.score_history, label='score')
        lns2 = ax2.plot(x, self.eps_history, label='epsilon')
        lns3 = ax2.plot(x, self.loss_hist, label='loss')
        lns = lns1 + lns2 + lns3
        lbls = [l.get_label() for l in lns]
        ax.legend(lns, lbls)
        plt.show()
