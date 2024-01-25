#import agent of choice
#from networks.SQNetwork import SLearningNetwork
from networks.DSQNetwork import DSLearningNetwork
#from networks.DDQNetwork import DLearningNetwork
#from networks.DQNetwork import LearningNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
  network = DSLearningNetwork()
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
