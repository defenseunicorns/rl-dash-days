#import agent of choice
#from networks.SQNetwork import SLearningNetwork
from networks.DSQNetwork import DSLearningNetwork
#from netwroks.DDQNetwork import DLearningNetwork
#from networks.DQNetwork import LearningNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
  network = DSLearningNetwork()
  network.load()
  network.play()
