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
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class CNN(nn.Module):
    """
        CNN / architecture

        :param height: height of image
        :param width: width of image
        :param frames: number of frames per state
        :param num_actions: action space dimension
        :param q-learn: whether model is used with q-learning
        :param alpha: for use with split-qlearning to scale the gradient
    """
    def __init__(self, height, width, num_frames, num_actions, q_learn=True, alpha=1):
        super().__init__()
        self.num_actions = num_actions
        self.q_learn = q_learn
        self.alpha = alpha
        new_h = self.get_output_dim(height)
        new_w = self.get_output_dim(width)
        flat_dim = 64*new_h*new_w
        self.conv1 = nn.Conv2d(num_frames, 16, 8, 4)
        self.conv2 = nn.Conv2d(16,32,4,2)
        self.conv3 = nn.Conv2d(32,64,3,1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(flat_dim, 256)
        self.out = nn.Linear(256, num_actions)

    #TODO parameterize this magic with variable names
    def get_output_dim(self, dim):
        new_dim = (((dim - 8)//4 + 1) - 4) // 2 + 1
        new_dim = (new_dim - 3) + 1
        return new_dim

    def forward(self, X, actions=None):
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
        elif self.num_actions > 1:
            return F.tanh(X)
        else:
            return X
