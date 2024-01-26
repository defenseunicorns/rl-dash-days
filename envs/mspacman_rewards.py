def vanilla_reward(reward, curr_lives, prev_lives, terminal, **kwargs):
    """reward is difference in score"""
    if not terminal:
        return reward
    else:
        return 0

def life_penalty(reward, curr_lives, prev_lives, terminal, **kwargs):
    """subtracts 1 from reward for each state transition"""
    if not terminal:
        return reward - 1
    else:
        return 0

def death_tax(reward, curr_lives, prev_lives, terminal, **kwargs):
    """Don't you die on me"""
    death_pen = kwargs.get('death_pen', 500)
    if curr_lives < prev_lives:
        return reward - death_pen*(prev_lives - curr_lives)

def ghost_buster(reward, curr_lives, prev_lives, terminal, **kwargs):
    """extra rewards for killing ghosts"""
    ghost_mult = kwargs.get('ghost_mult', 1)
    if reward > 200:
        return reward*ghost_mult
    else:
        return reward

def split_q_parameterized(reward, curr_lives, prev_lives, terminal, **kwargs):
    """Multiple options parameterized with kwargs (split-q)"""
    death_pen = kwargs.get('death_pen', 500)
    ghost_mult = kwargs.get('ghost_mult', 1)
    if ghost_mult > 1 and reward > 200:
        reward = reward * ghost_mult
    pos = reward
    neg = -1
    if curr_lives < prev_lives:
        neg -= (death_pen*(prev_lives - curr_lives))
    return pos, neg

def regular_parameterized(reward, curr_lives, prev_lives, terminal, **kwargs):
    """Multiple options parameterized (normal)"""
    death_pen = kwargs.get('death_pen', 500)
    ghost_mult = kwargs.get('ghost_mult', 1)
    if ghost_mult > 1 and reward > 200:
        reward = reward * ghost_mult
    if curr_lives < prev_lives:
        reward -= (death_pen*(prev_lives - curr_lives))
    
    return reward


reward_fcns = {
    'vanilla':vanilla_reward,
    'life_penalty':life_penalty,
    'death_tax':death_tax,
    'ghostbusters':ghost_buster,
    'split_q':split_q_parameterized,
    'parameterized':regular_parameterized,
}