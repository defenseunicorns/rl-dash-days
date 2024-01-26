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
To test a model, see the output, apply the `-n` name and `-a` algorithm tags to `python test.py` For a saved model

### Frontend dashboard
To see the frontend, run `python dashboard.py`, which will make the dashboard available at `localhost:5000`.