from envs.mspacman import MsPacmanEnv
from test import build_runner
import matplotlib.pyplot as plt

def lives_combiner(state, reward, curr_lives, prev_lives, terminal, **kwargs):
    """simple combiner that returns 0 if lives == 3 else 1"""
    if curr_lives == 3:
        return 0
    return 1

class ModelCombiner:
    def __init__(self, model0_params, model1_params, combination_fcn):
        """ Combines two models via a combination function
            :param model0_params: list of 'name' and 'algorithm' to build model
            :param model1_params: list of 'name' and 'agortihm' to build model
            :combination_fcn: fcn that takes state information and chooses model
        """
        self.env = MsPacmanEnv()
        self.model0 = build_runner(model0_params[0], model0_params[1], self.env)
        self.model1 = build_runner(model1_params[0], model1_params[1], self.env)
        self.combine = combination_fcn

    def eval(self):
        state, reward, terminal, lives, frame = self.env.reset()
        self.lives = lives
        self.model0.load_eval()
        self.model1.load_eval()
        im = plt.imshow(self.env.render())
        plt.ion()
        while not terminal:
            if self.combine(state, reward, lives, self.lives, terminal):
                state, reward, terminal, lives, frame = self.model1.advance(state, self.env, im)
            else:
                state, reward, terminal, lives, frame = self.model0.advance(state, self.env, im)

def eval_combination(m0_name, m0_algo, m1_name, m1_algo, combination_fcn=lives_combiner):
    runner = ModelCombiner([m0_name, m0_algo], [m1_name, m1_algo], combination_fcn)
    runner.eval()

if __name__ == '__main__':
    eval_combination('cp', 'DSQ', 'ppo_long', 'PPO')