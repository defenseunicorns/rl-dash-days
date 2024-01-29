from algos.deep_split_ql import DSQ
from algos.double_deep_ql import DDQ
from algos.prox_pol import PPO
from envs.mspacman import MsPacmanEnv
from test import build_runner

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
        self.model0 = build_runner(model0_params[0], model0_params[1], env)
        self.model1 = build_runner(model1_params[0], model1_params[1], env)
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