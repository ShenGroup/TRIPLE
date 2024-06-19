import numpy as np

from src.bandit import BasicBandit

class SequentialHalving(BasicBandit):

    def __init__(self, num_arms: int = 0, T: int = 0) -> None:
        super().__init__(num_arms)
        self.T = T
        self.L = np.ceil(np.log2(self.num_arms))

        self.active_arms = np.arange(self.num_arms)
        self.num_active_arms = len(self.active_arms)

        self.num_pulls = np.zeros(self.num_active_arms, dtype= float)
        self.total_pulls = np.zeros(num_arms, dtype= float)
        self.sample_mean = np.zeros(self.num_active_arms, dtype= float)
        self.pull_sequence = np.repeat(self.active_arms, 
                                       repeats = max(int(np.floor(self.T/(self.L * self.num_active_arms))), 1)
                                       )

        self.t = 0
    
    def update(self, arm, reward):
        index = np.where(self.active_arms == arm)[0][0]
        self.num_pulls[index] += 1
        self.total_pulls[arm] += 1
        n = self.total_pulls[arm]
        self.sample_mean[arm] = ((n - 1) / float(n)) * self.sample_mean[arm] + (1 / float(n)) * reward
        
        self.t += 1
    
    def choose_action(self):
        if self.t >= len(self.pull_sequence):
            sample_mean = self.sample_mean[self.active_arms]
            self.num_active_arms = int(np.ceil(self.num_active_arms/2))
            active_index = (-sample_mean).argsort()[:self.num_active_arms]
            self.active_arms = self.active_arms[active_index]
            self.pull_sequence = np.append(self.pull_sequence, np.repeat(self.active_arms, repeats = np.floor(self.T/(self.L * self.num_active_arms))))

            self.num_pulls = np.zeros(self.num_active_arms, dtype= float)
            # self.rewards = np.zeros(self.num_active_arms, dtype= float)
            
        if self.t >= len(self.pull_sequence):
            self.pull_sequence = np.append(self.pull_sequence, self.active_arms)

        return self.pull_sequence[self.t]

    
    def best_arm(self):
        return self.active_arms[-1]
    
    def reset(self):
        self.active_arms = np.arange(self.num_arms)
        self.num_active_arms = len(self.active_arms)

        self.num_pulls = np.zeros(self.num_active_arms, dtype= float)
        self.total_pulls = np.zeros(self.num_arms, dtype= float)
        self.sample_mean = np.zeros(self.num_active_arms, dtype= float)

        self.pull_sequence = np.repeat(self.active_arms, repeats = np.floor(self.T/(self.L * self.num_active_arms)))

        self.t = 0
        
    def get_sample_mean(self):
        
        return self.sample_mean 


if __name__ == "__main__":
    bandit = SequentialHalving(num_arms=10, T=10)
    print(bandit.choose_action())