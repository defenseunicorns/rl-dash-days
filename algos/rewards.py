def vanilla_reward(reward, terminal, info):
    if not terminal:
        return reward
    else:
        return 0

def split_q_baseline(reward, terminal, info):
    pos = reward
    neg = -1

    