import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from algos.double_deep_ql import DDQ
from algos.deep_split_ql import DSQ
from algos.prox_pol import PPO
from networks.cnn import CNN
from envs.mspacman import MsPacmanEnv
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
    env = MsPacmanEnv(ppo = (algo=='PPO')
    if algo == 'PPO':
        runner = PPO(name, env, function, training=False)
    elif algo == 'DDQ':
        runner = DDQ(name, env, function, training=False)
    elif algo == 'DSQ':
        runner = DSQ(name, env, function, training=False)
    return runner

def eval_model(name, algo):
    runner = build_runner(name, algo)
    runner.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Provide training arguments")
    parser.add_argument('-n', '--name')
    parser.add_argument('-a', '--algo', help='DDQ, DSQ, PPO')
    args = parser.parse_args()
    eval_model(args.name, args.algo)