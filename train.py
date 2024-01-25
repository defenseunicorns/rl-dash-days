from algos import DDQ
from envs.mspacman import MsPacmanQL, MsPacmanPPO
from envs.mspacman_rewards import reward_fcns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import popen
import argparse

def build_runner(name, algo, reward_fcn, death_pen, ghost_mult):
    kwargs = {}
    if death_pen:
        kwargs['death_pen'] = int(deat_pen)
    if ghost_mult:
        kwargs['ghost_mult'] = int(ghost_mult)
    function = reward_fcns.get(reward_fcn, None)
    if not function:
        function = lamda a,b,c,d : a
    if algo == 'PPO':
        env = MsPacmanPPO()
        runner = None
    elif algo == 'DDQ':
        env = MsPacmanQL()
        runner = DDQ(name, env, function, **kwargs)
    return runner

def save_and_plot(runner):
    scores = runner.score_history
    its = runner.its_history
    loss = runner.loss_history
    eps = runner.eps_history
    x = range(len(its))

    path = f'./data/{runner.name}.csv'
    with open(path, 'w') as fh:
        fh.write('score,actions,loss,eps\n')
        for i range(len(scores)):
            fh.write(f'{scores[i]},{its[i]},{loss[i]},{eps[i]}\n'

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
    plt.savefig(f'./data/{runner.name}.png'
    plt.show()

def train_model(name, algo, reward_fcn, death_pen, ghost_mult):
    runner = build_runner(name, algo, reward_fcn, death_pen, ghost_mult)
    runner.train()
    save_and_plot(runner)

if __name__ == '__main__':
  env = 
  network.train(20000)

  data_path = './data/dsqn_vanilla.csv'

  scores = network.score_history
  its = network.its_hist
  loss = network.loss_hist
  eps = network.eps_history
  x = range(len(its))

  with open(data_path, 'w') as fh:
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
  plt.show()
