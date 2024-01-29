# rl-dash-days
Dash Days project on Reinforcement Learning

## Getting started:
First, please `pip install -r requirements.txt` to ensure full functionality.

### Training CLI
To train a model, use the following CLI with `python train.py`
* -n, --name: Name the model for data storage purposes
* -a, --algorithm: Double-Deep Qlearning (DDQ), Deep Split Q-learning (DSQ), or PPO
* -r, --reward_fcn: custom reward functions (see `envs/mspacman_rewards` or create your own)
* -d, --death_pen: penalizes the agent for dying (integer punishment on death)
* -g, --ghost_mul: multiplies the rewards of killing a ghost (integer)
* -e, --epochs: How many games to play in total (2000 takes around an hour on my system)
* --reward_weight: For use in split-Q learning
* --punish_weight: For use in split-Q learning
* --reward_memory: For use in split-Q learning
* --punish_memory: For use in split-Q learning

### Testing CLI
To test a model, see the output, apply the `-n` name and `-a` algorithm tags to `python test.py` For a saved model.

### Combine models
There is some boilerplate code to combine two models into a kind of state machine of models with `combine.py` but I didn't really develop this capability, it was just an idea.

### Frontend dashboard
To see the frontend, run `python dashboard.py`, which will make the dashboard available at `localhost:5000`.

## Repo structure
### Backend
The backend comprises of the gym environmenmt wrapper `envs/mspacman.py`, and the reward functions, `envs/mspacman_rewards.py`.  The algorithms are found in the `algos/` folder and contain all the code needed to initialize a model, load a model, and run a training loop or test loop.  The `networks` folder has the actual Convolutional Neural Network architecture and the state transition buffer architectures.  Data for actual runs and metadata of saved models is stored in the `data/` folder, which the frontend interacts with.  Actual model weights are saved in the `models/` folder.

### Frontend
The frontend consists of a ploty Dash application.  The layout is stored in `layouts/layout.py` while most of the other functionality is stored in `callbacks/`, which includes the data loaders, the functionality, and the plotting functions.

## Algorithm Details
1. [Double-Deep Q-learning](https://towardsdatascience.com/double-deep-q-networks-905dd8325412) (DDQ): DDQ is a Q-learning algorithm:  Double deep q-learning works by attempting to learn the relative value of a state,action pair.  One hyperparameter that might need to be tuned that isn't available by the CLI is the epsilon parameter.  This parameter controls when the training algorithm performs a random action and when it uses the model to select an action.  This can be modified in the `algos/deep_split_ql.py` file in the `get_epsilon_for_iteration()` method.
2. [Split-Q learning](https://arxiv.org/pdf/1906.11286.pdf) (DSQ):  This is also a Q-learning algorithm that splits the reward into a positive and negative stream, with a model for each.  Epsilon is also applicable here and can be modified in `algos/deep_split_ql.py`.  See above paper for detailed description of hyperparameters, but the basic idea is you can modify the way rewards and punishemts are processed.
3. [Proximal Policy Optimization](https://medium.com/mlearning-ai/ppo-intuitive-guide-to-state-of-the-art-reinforcement-learning-410a41cb675b) (PPO).  Unlike Q-learning, PPO can use real-valued action spaces.  This algorithm uses an action representation that is a simlulated x-value and y-value for the atari joystick.  The conversion is done in `envs/mspacmman.py`, which is constructed with a boolean `ppo` variable if PPO is to be used.
