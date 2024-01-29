import gym

class Environment:
    """
    Wrapper class to translate between Deep learning framework and actual environment

    :param environment: gym environment object, if initialized
    :param environment_name: if no environment object, create the gym environement
    :param info_args: arguments for data to collect from the 'info' field of the state
    :param render_mode: mode for gym to render
    :param kwargs: any additional args that may be needed for gym environment setup
    """
    def __init__(self, environment, environment_name,
               info_args, render_mode, **kwargs):
        if environment:
            self.env = environment
        else:
            self.env = gym.make(environment_name, render_mode=render_mode)
        self.info_args = info_args
    
    def reset(self):
        """resets the environment"""
        pass
    
    def step(self, action):
        """steps the environment"""
        pass
    
    def render(self):
        """renders the environment"""
        pass

    def sample(self):
        """samples from the action space"""
        pass