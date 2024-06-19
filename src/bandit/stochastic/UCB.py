import numpy as np

from src.bandit import BasicBandit

class UCB(BasicBandit):

    def __init__(self, num_arms: int = 0, alpha: float = 2) -> None:
        super().__init__(num_arms)
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.t = 0
    
    def update(self, arm, reward):
        self.num_pulls[arm] += 1
        n = self.num_pulls[arm]
        self.sample_mean[arm] = ((n - 1) / float(n)) * self.sample_mean[arm] + (1 / float(n)) * reward
        self.t += 1
    
    def choose_action(self):
        for arm in range(self.num_arms):
            if self.num_pulls[arm] == 0:
                return arm
        ucb = [0.0 for _ in range(self.num_arms)]
        for arm in range(self.num_arms):
            radius = np.sqrt((2 * np.log(self.t)) / float(self.num_pulls[arm]))
            ucb[arm] = self.sample_mean[arm] + radius
        return np.argmax(ucb)
    
    def best_arm(self):
        return np.argmax(self.num_pulls)
    
    def reset(self):
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.t = 0
        
    def get_sample_mean(self):
        
        return self.sample_mean

