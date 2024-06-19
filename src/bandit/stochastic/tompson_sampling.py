import numpy as np

def thompson_sampling(bandit_probs, num_trials):
    num_bandits = len(bandit_probs)
    num_wins = np.zeros(num_bandits)
    num_losses = np.zeros(num_bandits)

    for _ in range(num_trials):
        samples = np.random.beta(num_wins + 1, num_losses + 1)
        chosen_bandit = np.argmax(samples)
        reward = np.random.binomial(1, bandit_probs[chosen_bandit])

        if reward == 1:
            num_wins[chosen_bandit] += 1
        else:
            num_losses[chosen_bandit] += 1

    return np.argmax(num_wins / (num_wins + num_losses))

# Example usage
bandit_probs = [0.3, 0.5, 0.7]  # Probabilities of success for each bandit
num_trials = 1000  # Number of trials

best_bandit = thompson_sampling(bandit_probs, num_trials)
print("Best bandit:", best_bandit)