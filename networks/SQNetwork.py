import gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from dqn import DQN, ReplayBuffer
import random

REWARD_PATH = './models/checkpoint_sqn_reward.pt'
PUNISH_PATH = './models/checkpoint_sqn_punish.pt'

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class SLearningNetwork:
    """
        Novel implementation of Lin et. al's split Q-learning for Deep RL

        :param gamma: reward discount
        :param batch_size: replay buffer batch sizes
        :param env: gym environment name
        :param num_frames: number of frames per state (and number per action)
        :param lam_r: instantaneous reward scaling [1,100] typically
        :param lam_p: instantaneous punishment scaling [1,100]
        :param a_r: gradient scaling for rewards [.1,1] typically
        :param a_l: gradient scaling for punishments [.1,1]
    """
    def __init__(self, gamma=.95, batch_size=64,
                env='MsPacmanDeterministic-v4', num_frames=4,
                lam_r=1, lam_p=1, a_r=1, a_p=1):
        self.num_frames = num_frames
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.memory = ReplayBuffer(30000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.lam_r = lam_r
        self.lam_p = lam_p
        self.env = gym.make(env)
        self.start = self.get_start_state()
        _, height, width = self.start.size()
        self.num_actions = self.env.action_space.n
        self.render = lambda : plt.imshow(env.render(mode='rgb_array'))
        self.reward_pol = DQN(height, width, num_frames, self.num_actions, a_r).to(self.device).train()
        self.reward_tar = DQN(height, width, num_frames, self.num_actions).to(self.device).eval()
        self.punish_pol = DQN(height,width, num_frames, self.num_actions, a_p).to(self.device).train()
        self.punish_tar = DQN(height, width, num_frames, self.num_actions).to(self.device).eval()
        self.loss = nn.SmoothL1Loss()
        self.opt_r = torch.optim.RMSprop(self.reward_pol.parameters(), lr=.00025, alpha=.95, eps=0.01)
        self.opt_p = torch.optim.RMSprop(self.punish_pol.parameters(), lr=.00025, alpha=.95, eps=.01)
        self.reward_pol.apply(init_weights)
        self.reward_tar.apply(init_weights)
        self.punish_pol.apply(init_weights)
        self.punish_tar.apply(init_weights)


    def get_start_state(self):
        self.lives = 3
        frame = self.preprocess(self.env.reset())
        frames = []
        for i in range(self.num_frames):
            frames.append(frame.clone())
        return torch.cat(frames)

    def preprocess(self, img):
        ds = img[::2,::2]
        grayscale = np.mean(ds, axis=-1).astype(np.uint8)#saves memory
        return torch.tensor(grayscale).unsqueeze(0)

    def fit_buffer(self, sample):
        rew_p = self.reward_pol.train()
        rew_t = self.reward_tar.eval()
        pun_p = self.punish_pol.train()
        pun_t = self.punish_tar.eval()
        dev = self.device

        states, actions, next_states, rewards, punishments, non_terms = list(zip(*sample))
        states = torch.cat(states).to(dev)
        actions = torch.tensor(actions).long().to(dev)
        next_states = torch.cat(next_states).to(dev)
        rewards = torch.tensor(rewards).long().to(dev)
        punishments = torch.tensor(punishments).long().to(dev)
        non_terms = torch.tensor(non_terms).to(dev)
        next_mask = torch.ones((actions.size(0), self.num_actions)).to(dev)
        curr_mask = F.one_hot(actions, self.num_actions).to(dev)

        #process rewards
        next_Qr_vals = rew_t(next_states, next_mask)
        next_Qr_vals = next_Qr_vals.max(1)[0] * non_terms
        next_Qr_vals = (next_Qr_vals * self.gamma) + (self.lam_r * rewards)

        expected_Qr_vals = rew_p(states, curr_mask)
        expected_Qr_vals = expected_Qr_vals.gather(-1, actions.unsqueeze(1))

        self.opt_r.zero_grad()
        r_loss = self.loss(expected_Qr_vals.squeeze(1), next_Qr_vals)
        r_loss.backward()
        for param in rew_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt_r.step()

        #process punishments
        next_Qp_vals = pun_t(next_states, next_mask)
        next_Qp_vals = next_Qp_vals.max(1)[0] * non_terms
        next_Qp_vals = (next_Qp_vals * self.gamma) + (self.lam_p * punishments)

        expected_Qp_vals = pun_p(states, curr_mask)
        expected_Qp_vals = expected_Qp_vals.gather(-1, actions.unsqueeze(1))

        self.opt_p.zero_grad()
        p_loss = self.loss(expected_Qp_vals.squeeze(1), next_Qp_vals)
        p_loss.backward()
        for param in pun_p.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt_p.step()

        return r_loss.item(), p_loss.item()

    def get_epsilon_for_iteration(self, iteration):
        #TODO provide scaling as parameter
        return max(.01, 1-(iteration*.99/500000))

    def choose_best_action(self, frames):
        pun = self.punish_pol
        rew = self.reward_pol
        dev = self.device
        frames = frames.unsqueeze(0).to(dev)
        mask = torch.ones(1,self.num_actions).to(dev).squeeze(0)
        pun.eval()
        rew.eval()
        r_Q = rew(frames, mask)
        p_Q = pun(frames, mask)
        actions = (r_Q + p_Q).squeeze(0)
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
        total_score = 0
        tot_reward = 0
        punishment = 0
        # Play one game iteration (num_frames):
        for i in range(self.num_frames):
            if not is_done:
                new_frame, reward, is_done, lives = self.env.step(action)
                score = reward
                #anxiety of dying
                punishment -= 1
                #excitment of staying alive
                tot_reward += 1
                if reward < 0:
                    punishment += reward
                else:
                    tot_reward += reward
                lives = lives['ale.lives']
                new_frame = self.preprocess(new_frame)
                #modify rewards if life lost
                if lives != self.lives:
                    if lives < self.lives:
                        punishment -= 10
                    else:
                        tot_reward += 10
                    self.lives = lives
                total_score += score
            else:
                tot_reward = 0
                punishment = 0
            new_frames.append(new_frame)
        new_frames = torch.cat(new_frames)

        non_term = 1
        if is_done:
            non_term = 0

        mem = (frames.unsqueeze(0), action,
               new_frames.unsqueeze(0), tot_reward, punishment, non_term)
        self.memory.push(mem)

        r_loss = None
        p_loss = None
        grad = None
        # Sample and fit
        if iteration > 64:
            batch = self.memory.sample(self.batch_size)
            r_loss, p_loss = self.fit_buffer(batch)

        return is_done, total_score, new_frames, r_loss, p_loss

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
                is_done, reward, frames, r_loss, p_loss = self.q_iteration(frames, iteration)
                iteration += 1
                e_reward += reward
                e_its += 1
                if r_loss is not None:
                    running_loss += (r_loss + p_loss)/2
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
            if e%updates == 0 and e > 0:
                torch.save(self.reward_pol.state_dict(), REWARD_PATH)
                self.load_reward_t(REWARD_PATH)
                torch.save(self.punish_pol.state_dict(), PUNISH_PATH)
                self.load_punish_t(PUNISH_PATH)

    def load_reward_t(self, path):
        self.reward_tar.load_state_dict(torch.load(path))
        self.reward_tar.eval()

    def load_punish_t(self, path):
        self.punish_tar.load_state_dict(torch.load(path))
        self.punish_tar.eval()

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
        #TODO cleanup this method
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
