import gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from networks.dqn import DQN, ReplayBuffer, init_weights
from envs.mspacman import MSPacmanQL
import random

class DDQ:
    """
        Double Deep Q-learning implementation

        :param name: name for checkpointing / data
        :param env: Environment object
        :param reward_fcn: reward function
        :param policy: policy network
        :param target: target network
        :param gamma: reward discount
        :param batch_size: replay buffer batch sizes
        :param env: gym environment
        :param num_frames: number of frames per state (and number per action)
    """
    def __init__(self, name, env, reward_fcn, policy, target
                 gamma=.95, batch_size=64, num_frames=4, **reward_kwargs):
        self.name = name
        self.path = f'./models/checkpoint_{self.name}.pt'
        self.reward_fcn = reward_fcn
        self.num_frames = num_frames
        self.device = torch.device('cuda') if torch.cuda.device_count() > 0 else torch.device('cpu')
        self.memory = ReplayBuffer(50000)
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.start = self.reset_env()
        _, height, width = self.start.size()
        self.num_actions = self.env.action_space
        self.policy = policy.to(self.device)
        self.target = target.to(self.device)
        self.loss = nn.SmoothL1Loss()
        self.opt = torch.optim.RMSprop(self.policy.parameters(), lr=.00025, alpha=.95, eps=0.01)
        self.policy.apply(init_weights)
        self.target.apply(init_weights)

        self.reward_kwargs = reward_kwargs

    def load_tensors(self, sample):
        """See prepare_update for descriptions"""
        dev = self.device
        states, actions, next_states, rewards, non_terms = list(zip(*sample))
        states = torch.cat(states)
        actions = torch.tensor(actions).long().to(dev)
        next_states = torch.cat(next_states)
        rewards = torch.tensor(rewards).long().to(dev)
        non_terms = torch.tensor(non_terms).to(dev)
        next_mask = torch.ones((actions.size(0), self.num_actions))
        curr_mask = F.one_hot(actions, self.num_actions)
        return states, next_states, rewards, non_terms, curr_mask, next_mask

    def prepare_update(self, states, next_states, rewards, non_terms, curr_mask, next_mask):
        """Uses target and policy network to prepare double deep-q update
            :param states: initial states for state transition
            :param next_states: end states for state transition
            :param rewards: reward gained from state transition
            :param non_terms: boolean for whether a transition ended a game
            :curr_mask: one-hot version of actions
            :next_mask: ONES vector for the next action mask
            :returns: end state q-values (chosen by policy and calculated by target)
                and expected_Q_values (calculated by policy with actual action taken)
        """
        dev = self.device
        policy = self.policy.train()
        target = self.target.eval()

        next_Q_vals = policy(next_states.to(dev), next_mask.to(dev))
        next_actions = next_Q_vals.max(1)[1].to(dev)
        next_mask = F.one_hot(next_actions, self.num_actions).to(dev)
        next_Q_vals = target(next_states.to(dev), next_mask)
        next_Q_vals = next_Q_vals.gather(-1, next_actions.unsqueeze(1)).squeeze(-1) * non_terms
        next_Q_vals = (next_Q_vals * self.gamma) + rewards.to(dev)

        expected_Q_vals = policy(states.to(dev), curr_mask.to(dev))
        expected_Q_vals = expected_Q_vals.gather(-1, actions.unsqueeze(1)).squeeze(-1)

    def fit_buffer(self, sample):
        """Takes a sample of frames from the buffer and does training"""
        policy = self.policy
        target = self.target
        dev = self.device

        #load tensors
        states, next_states, rewards, non_terms, curr_mask, next_mask = self.load_tensors(sample)
        
        #update rule: Q(s,a) -> gamma* max_{a'}Q(s',a') + r
        next_Q_vals, expected_Q_vals = self.prepare_update(states, next_states, rewards,
                                                           non_terms, curr_mask, next_mask)

        self.opt.zero_grad()
        loss = self.loss(expected_Q_vals, next_Q_vals)
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()

        return loss.item()

    def reset_env(self):
        """resets the environment"""
        state, score, terminal, lives, frame_number = self.env.reset()
        self.lives = lives
        return state

    def get_epsilon_for_iteration(self, iteration):
        """simple linear epsilon function for"""
        return max(.01, 1-(iteration*.99/300000))

    def infer_action(self, state):
        """Uses the policy model to choose best action for state"""
        dev = self.device
        state = state.unsqueeze(0).to(dev)
        mask = torch.ones(1,self.num_actions).to(dev).squeeze(0)
        self.policy.eval()
        rewards = self.policy(state, mask)
        return int(actions.max(0)[1])

    def q_iteration(self, state, iteration):
        """Processes a single state transition / action choice"""
        env = self.env
        epsilon = self.get_epsilon_for_iteration(iteration)

        # Choose the action
        if random.random() < epsilon:
            action = self.env.sample()
        else:
            action = self.infer_action(self, state)
        new_state, score, terminal, lives, frame_number = self.env.step(action)
        reward = self.reward_fcn(score, lives, self.lives, terminal, **self.reward_kwargs)

        mem = (state.unsqueeze(0), action,
               new_state.unsqueeze(0), reward, terminal)
        self.memory.push(mem)

        loss = None
        # Sample and fit
        if iteration > 64:
            batch = self.memory.sample(self.batch_size)
            loss= self.fit_buffer(batch)

        return new_state, reward, score, terminal, loss

    #TODO: Could probably abstract this to a logger class
    #Or use MLflow?
    def init_history(self):
        self.stats = {'score':[], 'epsilon':[], 'loss':[], 'iterations':[]}

    def init_stats(self):
        stats = {
            'loss':0,
            'count':0,
            'score':0,
            'iterations':0,
            'epoch_score':0,
            'epoch_iterations':0
        }
        return stats

    def init_epoch(self, stats):
        state = self.env.reset()[0]
        stats['epoch_score'] = 0
        stats['epoch_iterations'] = 0
        terminal = False
        return terimnal, stats, state

    def update_running_stats(stats, score, loss):
        stats['iterations'] = stats['iterations'] + 1
        stats['epoch_score'] = stats['epoch_score'] + score
        stats['epoch_iterations'] = stats['epoch_iterations'] + 1
        if loss is not None:
            stats['loss'] = stats['loss'] + loss
            stats['count'] = stats['count'] + 1
        return stats

    def update_overall_stats(stats, iteration):
        avg_score = stats['score'] / 100
        avg_its = stats['iterations'] / 100
        self.stats['score'] = self.stats['score'].append(avg_score)
        eps = self.get_epsilon_for_iteration(iteration)
        self.stats['epsilon'] = self.stats['epsilon'].append(eps)
        self.stats['loss'] = 

    def end_epoch(stats):
        stats['score'] = stats['score'] + stats['epoch_score']
        stats['iterations'] = stats['iterations'] + stats['epoch_iterations']

    def train(self, epochs=10000, start_iter=0, updates=500):
        """Main training loop, saves statistics to class during training
            :param epochs: number of games to play
            :param start_iter: starting iteration (default 0)
            :param udpates: number of epochs before updating target model
        """
        self.init_history()
        self.updates = updates    #when to save models / swap policy & target
        iteration = start_iter
        running_stats = self.init_stats()
        for e in range(epochs):
            terminal, running_stats, state = self.init_epoch(running_stats)
            while not terminal:
                new_state, reward, score, terminal, loss = self.q_iteration(state, 
                                                            iteration)
                running_stats = self.update_running_stats(running_stats, score, loss)
            running_stats = self.end_epoch(stats)
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
                torch.save(self.policy.state_dict(), self.path)
                self.load_target(self.path)

    def load_target(self, path):
        self.target.load_state_dict(torch.load(path))
        self.target.eval()

    def load(self):
        """Used before play() method to load policy network"""
        self.policy.load_state_dict(torch.load(self.path))
        self.policy.eval()

    def play(self):
        state = self.env.reset()[0]
        im = plt.imshow(self.env.render())
        plt.ion()
        terminal = False
        while not terminal:
            action = self.infer_action(self, state)
            new_frames = []
            for i in range(self.num_frames):
                frame, reward, terminal, truncated, info = self.env.step(action)
                lives = info['lives']
                if terminal:
                    break
                im.set_data(self.env.render())
                plt.draw()
                plt.pause(.001)
                new_frames.append(self.env.img_preprocess(new_frame))
            state = torch.cat(new_frames)
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