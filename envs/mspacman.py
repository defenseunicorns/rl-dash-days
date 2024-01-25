from envs.environment import Environment

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

class MsPacmanEnv(Environment):
    """
    Class to translate between a generic model and a MsPacman environment

    :param num_frames: number of frames to include in a state
    :param action_space: tuple of action space dimensions / shape
    """
    def __init__(self, num_frames=4, action_space=9, environment=None,
                 environment_name='MsPacmanDeterministic-v4',
                 info_args=['lives','frame_number'], render_mode='rgb_array', **kwargs):
        super().__init__(environment, environment_name,
                         info_args, render_mode, **kwargs)
        if action_space is None:
            self.action_space = self.env.action_space.shape
        self.lives = 0
        self.score = 0

    def image_preprocess(self, img):
        """ Preprocesses a MsPacmanDeterministic observation
            :param img: 3x210x160 rgb image
            :returns: grayscale 1x105x80 uint 8 image
        """
        down_sample = img[::2,::2]
        grayscale = np.mean(down_sample, axis=-1).astype(np.uint8)
        return torch.tensor(grayscale).unsqueeze(0)

    def bundle_update(self, frames, reward, terminal, info):
        """bundles an update including the info_args and a state Tensor
            :param frames: list of process frames
            :reward: integer reward
            :terminal: boolean whether terminal state reached
            :info: dict of amplifying information
            :returns: bundled update for processing
        """
        state = torch.cat(frames)
        update = [state, reward, terminal]
        for arg in self.info_args:
            update.append(self.info_args[arg])
        return update

    def render(self):
        """renders the current frame in the environment"""
        im = plt.imshow(self.env.render())
        plt.ion()
        plt.show()

    def sample(self):
        """samples from the action space"""
        return self.env.action_space.sample()

    def reset(self):
        """Resets environment
            :returns: (starting state (4 frame copies), reward, terminal, lives, frame_number)
        """
        obs, reward, terminal, truncated, info = self.env.reset()
        frame = self.image_preprocess(obs)
        frames = []
        for i in range(self.num_frames):
            frames.append(frame.clone())
        return self.bundle_update(frames, reward, terminal, info)

    def step(self, action, render=False):
        """Steps through num_frames with the associated action
            :param action: for MsPacman, integer from 0 to 8
            :render: boolean, whether to render the frames as they are stepped through
            :returns (state {4 frames given action}, reward, terminal, lives, frame_number)
        """
        frames = []
        total_reward = 0
        terminal = False
        for i in range(self.num_frames):
            if not terminal:
                obs, reward, terminal, truncated, info = self.env.step(action)
                frame = self.img_preprocess(obs)
                total_reward += reward
                if render:
                    self.render()
            else:
                total_reward = 0
                break
            frames.append(frame)
        return self.bundle_update(frames, reward, terminal, info)

class MsPacmanQL(MsPacmanEnv):
    """
        Special environment wrapper for use with Q-learning
        (model outputs Q-values of actions)

        Q learning update in form state, reward, score, terminal, lives, frame_number
        Split-Q in form state, reward, punishment, score, terminal, lives, frame_number

        :param reward_fcn: custom reward functions
        :param split_q: whether rewards and punishments are split
    """
    
    def __init__(self, reward_fcn, split_q, num_frames=4, action_space=9,
                 environment=None, environment_name='MsPacmanDeterministic-v4',
                 info_args=['lives','frame_number'], render_mode='rgb_array', **kwargs):
        super().__init__()
        self.reward_fcn = reward_fcn
        self.split_q = split_q

    def bundle_update(self, frames, score, terminal, info):
        """Custom updating for Q learning / Split Q-learning"""
        state = torch.cat(frames)
        update = [state]
        if self.split_q:
            reward, punishment = reward_fcn(score, terminal, info)
            update.append(reward)
            update.append(punishment)
        else:
            update.append(reward_fcn(score))

    def choose_best_action(self, rewards, punishments=None):
        """Chooses the best action given the q-values
            :param rewards: tensor of q-values shape (9)
            :param punishments: tensor of punishment q-values (if split q-learning)
            :returns: integer for the best discrete action
        """
        if punishments:
            rewards = rewards+punishments
        return int(q_vals.max(0)[1])

class MsPacmanPPO(MsPacmanEnv):
    """
        Special environment wrapper for use with PPO
        Simulates actual joystick manipulation
        
        update in form state, reward, score, terminal, lives, frame_number

        :param reward_fcn: custom reward function
    """
    
    def __init__(self, reward_fcn, num_frames=4, action_space=9,
                 environment=None, environment_name='MsPacmanDeterministic-v4',
                 info_args=['lives','frame_number'], render_mode='rgb_array', **kwargs):
        super().__init__()
        self.reward_fcn = reward_fcn

    def sample(self):
        """Custom sampling to create fake joystick positions"""
        

    def bundle_update(self, frames, score, terminal, info):
        """Custom updating for Q learning / Split Q-learning"""
        state = torch.cat(frames)
        reward = self.reward_fcn(score, terminal, info)
        

    def interpret_action(self, x_value, y_value):
        """Translates stick position to discrete action space
            :param x_value: x-axis position of joystick
            :param y_value: y-axis position of joystick
            :returns: integer for the discrete action
        """
        key = ""
        if y_value > .5:
            key += "up"
        elif y_value < -.5:
            key += "down"
        if x_value > .5:
            key += "right"
        elif x_value < -.5:
            key += "left"
        return JOYSTICK_TRANSLATION[key]