import random

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

class PPOBuffer(object):
    """Buffer that makes garbage collection easier for PPO"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.terms = []
        self.logprobs = []

    def push_state(self, state):
        self.states.append(state)

    def push(self, action, reward, terminal, logprobs):
        self.actions.append(action)
        self.rewards.append(reward)
        self.terms.append(terminal)
        self.logprobs.append(logprobs)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.terms[:]
        del self.logprobs[:]