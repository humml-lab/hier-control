def compute_neg_discounted_expected_reward(discount_factor, reward):
    assert reward.dim() == 2, "Reward should be of shape (sequence_length, 1)"
    # compute discounted expected reward
    neg_expected_reward = 0
    for t in range(reward.size(0)):
        neg_expected_reward += (discount_factor ** t) * reward[t]
    neg_expected_reward = neg_expected_reward / reward.size(0)  # normalize by sequence length
    return - neg_expected_reward
