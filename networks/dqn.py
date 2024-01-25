import torch.nn.functional as F
from torch import nn
import torch
import random

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class GradMultiply(torch.autograd.Function):
    """
        Used for scaling the gradient in the split Q-learning implementation
        originally from fairseq module under MIT license.
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class ReplayBuffer(object):
    """
        Ring ReplayBuffer based on pytorch documentation
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, buffer):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = buffer
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch from the memory buffer"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ResNet(nn.Module):
    """
        CNN / resnet architecture

        :param height: height of image
        :param width: width of image
        :param frames: number of frames per state
        :param num_actions: action space dimension
        :param q-learn: whether model is used with q-learning
        :param alpha: for use with split-qlearning to scale the gradient
    """
    def __init__(self, height, width, frames, num_actions, q_learn, alpha=1):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.alpha = alpha
        new_h = self.get_output_dim(height)
        new_w = self.get_output_dim(width)
        flat_dim = 64*new_h*new_w
        self.conv1 = nn.Conv2d(frames, 16, 8, 4)
        self.conv2 = nn.Conv2d(16,32,4,2)
        self.conv3 = nn.Conv2d(32,64,3,1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(flat_dim, 256)
        self.out = nn.Linear(256, num_actions)

    def get_output_dim(self, dim):
        new_dim = (((dim - 8)//4 + 1) - 4) // 2 + 1
        new_dim = (new_dim - 3) + 1
        return new_dim

    def forward(self, X, actions):
        X = X/255.0 #casts from uint8 to float for processing
        res = X
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = self.flat(X)
        X = F.relu(self.fc(X))
        X = self.out(X)
        if self.alpha!= 1:
            X = GradMultiply.apply(X, self.alpha)
        if self.q_learn:
            return X*actions
        else:
            return F.tanh(X)
