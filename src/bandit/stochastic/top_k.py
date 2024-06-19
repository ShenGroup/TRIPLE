import numpy as np

from src.bandit import BasicBandit

class TopK(BasicBandit):

    def __init__(self, num_arms: int = 0, T: int = 100, k: float = 0.05) -> None:
        super().__init__(num_arms)
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.individual_budget = T//num_arms
        self.t = 0
    
    def update(self, arm, reward):
        self.num_pulls[arm] += 1
        n = self.num_pulls[arm]
        self.sample_mean[arm] = ((n - 1) / float(n)) * self.sample_mean[arm] + (1 / float(n)) * reward
        self.t += 1
    
    def choose_action(self):
        for arm in range(self.num_arms):
            if self.num_pulls[arm] < self.individual_budget:
                return arm
        
        return np.argmax(self.sample_mean)
            
    def best_arm(self):
        return np.argmax(self.sample_mean)
    
    def reset(self):
        self.num_pulls = np.zeros(self.num_arms, dtype=np.int32)
        self.sample_mean = np.zeros(self.num_arms)
        self.t = 0
        
    def get_sample_mean(self):
        
        return self.sample_mean

