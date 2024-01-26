from algos.double_deep_ql import DDQ
from algos.deep_split_ql import DSQ
from networks.cnn import CNN
from envs.mspacman import MsPacmanEnv, MsPacmanQL, MsPacmanPPO
from envs.mspacman_rewards import reward_fcns
from subprocess import Popen
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def build_runner(name, algo, reward_fcn, death_pen, ghost_mult,
                reward_weight, punishment_weight, lambda_r, lambda_p):
    kwargs = {}
    if death_pen:
        kwargs['death_pen'] = int(death_pen)
    if ghost_mult:
        kwargs['ghost_mult'] = int(ghost_mult)
    function = reward_fcns.get(reward_fcn, None)
    if not function:
        function = lambda a,b,c,d : a
    env = MsPacmanEnv()
    if algo == 'PPO':
        runner = PPO(name, env, function)
    elif algo == 'DDQ':
        runner = DDQ(name, env, function)
    elif algo == 'DSQ':
        runner = DSQ(name, env, function,
                     reward_weight=reward_weight,
                     punishment_weight=punishment_weight,
                     lambda_r=lambda_r, lambda_p=lambda_p)
    return runner

def save_and_plot(name, stats):
    scores = stats['score']
    its = stats['iterations']
    loss = stats['loss']
    eps = stats['epsilon']
    x = range(len(its))

    if not os.path.exists('./data'):
        os.mkdir('./data')

    path = f'./data/{name}.csv'
    with open(path, 'w') as fh:
        fh.write('score,actions,loss,eps\n')
        for i in range(len(scores)):
            fh.write(f'{scores[i]},{its[i]},{loss[i]},{eps[i]}\n')

    sns.set_style('dark')
    fig, ax1 = plt.subplots()
    lin1 = ax1.plot(x, scores, label='Score')
    ax1.set_xlabel('Epoch x 100')
    ax1.set_ylabel('Avg SPE')
    ax2 = ax1.twinx()
    ax2.plot([0],[0])
    lin2 = ax2.plot(x, its, label='Iterations')
    ax2.set_ylabel('Avg APE')
    plt.title('Average APE and SPE in 100 Epoch Intervals')
    lins = lin1 + lin2
    lbls = [l.get_label() for l in lins]
    ax1.legend(lins, lbls)
    plt.savefig(f'./data/{name}.png')
    plt.show()

def train_model(name, algo, reward_fcn, death_pen, ghost_mult, epochs,
               reward_weight=1, punish_weight=1, lambda_r=1, lambda_p=1):
    runner = build_runner(name, algo, reward_fcn, death_pen, ghost_mult,
                         reward_weight, punish_weight, lambda_r, lambda_p)
    if not epochs:
        epochs = 10000
    runner.train(epochs=epochs)
    save_and_plot(runner.name, runner.logger.stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Provide training arguments")
    parser.add_argument('-n', '--name')
    parser.add_argument('-a', '--algo', help='DDQ, DSQ, PPO')
    parser.add_argument('-r', '--reward_fcn', help='See envs/mspacman_rewards.py')
    parser.add_argument('-d', '--death_pen', type=int, help='penalty for dying')
    parser.add_argument('-g', '--ghost_mult', type=int, help='bounty for killing ghosts')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to run')
    parser.add_argument('--reward_weight', type=float, help='reward weight for splitQ learning')
    parser.add_argument('--punish_weight', type=float, help='punishment weight for splitQ learning')
    parser.add_argument('--reward_memory', type=float, help='lambda for rewards for splitQ learning')
    parser.add_argument('--punish_memory', type=float, help='lambda for punishments for splitQ learning')
    args = parser.parse_args()
    train_model(args.name, args.algo, args.reward_fcn,
                args.death_pen, args.ghost_mult, args.epochs,
                args.reward_weight, args.punish_weight, args.reward_memory,
                args.punish_memory)
