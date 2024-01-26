#import agent of choice
#from networks.SQNetwork import SLearningNetwork
from networks.DSQNetwork import DSLearningNetwork
#from netwroks.DDQNetwork import DLearningNetwork
#from networks.DQNetwork import LearningNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from algos.double_deep_ql import DDQ
from networks.cnn import CNN
from envs.mspacman import MsPacmanEnv, MsPacmanQL, MsPacmanPPO
from envs.mspacman_rewards import reward_fcns
from subprocess import Popen
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def build_runner(name, algo):
    function = reward_fcns.get('vanilla', None)
    if not function:
        function = lambda a,b,c,d : a
    if algo == 'PPO':
        env = MsPacmanPPO()
        runner = None
    elif algo == 'DDQ':
        env = MsPacmanEnv()
        _, height, width = env.reset()[0].shape
        policy = CNN(height, width, num_frames=4, num_actions=9, q_learn=True)
        target = CNN(height, width, num_frames=4, num_actions=9, q_learn=True)
        runner = DDQ(name, env, function, policy, target, training=False)
    return runner

def eval_model(name, algo):
    runner = build_runner(name, algo)
    runner.eval()
    save_and_plot(runner.name, runner.logger.stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Provide training arguments")
    parser.add_argument('-n', '--name')
    parser.add_argument('-a', '--algo', help='DDQ, DSQ, PPO')
    args = parser.parse_args()
    eval_model(args.name, args.algo)