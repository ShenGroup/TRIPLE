import numpy as np

from src.bandit import BasicBandit
import pdb

class ContinuousRejects(BasicBandit):

    def __init__(self, num_arms: int = 0, T: int = 0, target_arms: int = 1) -> None:
        super().__init__(num_arms)
        self.T = T
        self.L = np.ceil(np.log2(self.num_arms))
        self.target_arms = target_arms
        self.active_arms = np.arange(self.num_arms)
        self.non_active_arms = np.array([], dtype= int)
        self.num_active_arms = len(self.active_arms)
        self.num_pulls = np.zeros(self.num_active_arms, dtype= float)
        self.sample_mean = np.zeros(self.num_active_arms, dtype= float)
        self.cumulative_rewards = np.zeros(num_arms, dtype= float)
        # theta = 1/self.__log_bar(self.num_arms)
        theta = 0.0
        self.threshold = np.ceil(theta * T)

        self.t = 0
        
        
    def __log_bar(self, m: int = 30):
        
        temp_bar = [1/i for i in range(2, m+1)]
        return 1/2 + sum(temp_bar)
    
    def update(self, arm, reward):
        # index = np.where(self.active_arms == arm)[0][0]
        self.num_pulls[arm] += 1
        n = self.num_pulls[arm]
        self.sample_mean[arm] = ((n - 1) / float(n)) * self.sample_mean[arm] + (1 / float(n)) * reward
        self.cumulative_rewards[arm] += reward
        self.t += 1
        active_means = self.sample_mean[self.active_arms]
        if self.t >= self.threshold:
            l = int(self.active_arms[np.argmin(active_means)])
            condition1 = (self.num_active_arms >= 2)
            if len(self.non_active_arms) > 0:
                condition2 = self.num_pulls[l] > np.max(self.num_pulls[self.non_active_arms])
            else:
                condition2 = True
            if condition1:
                condition3 = (self.__CRA(l) or self.__CRC(l))
            else:
                condition3 = False
            
            if condition1 and condition2 and condition3:
                # reject arm l
                idx_l = np.where(self.active_arms == l)[0][0]
                self.active_arms = np.delete(self.active_arms, idx_l)
                self.non_active_arms = np.append(self.non_active_arms, int(l))
                self.num_active_arms -= 1
                print(f"Round {self.t}, Rejected arm {l}, current active arms: {self.active_arms}, current sample mean: {self.sample_mean}")
        
        
    def choose_action(self):
        
        # if never pulled, pull it
        if np.any(self.num_pulls == 0):
            return np.where(self.num_pulls == 0)[0][0]
        
        active_pulls = self.num_pulls[self.active_arms]
        return self.active_arms[np.argmin(active_pulls)]
       
    def __G_func(self, beta: float):
        return 1/np.sqrt(beta) - 1


    def __CRA(self, l:int) -> bool:
        """
        Check if to reject arm l or not
        """
        mu_l = self.sample_mean[l]
        mu_k = self.sample_mean[self.active_arms]
        idx_l = np.where(self.active_arms == l)[0][0]
        mu_k = np.delete(mu_k, idx_l)
        mean_k = np.mean(mu_k)
        min_diff = mean_k - mu_l
        beta = sum([self.num_pulls[i]* self.__log_bar(self.num_active_arms) for i in self.active_arms]) / (self.T - sum([self.num_pulls[i] for i in self.non_active_arms]))
        threshold = self.__G_func(beta)
        # print(f"CR-A check: Round {self.t}, beta: {beta}, threshold: {threshold}, min_diff: {min_diff}")
        
        return min_diff > threshold
    
    def __CRC(self, l:int) -> bool:
        """
        Check if to reject arm l or not via CR-A
        """
        mu_l = self.sample_mean[l]
        mu_k = self.sample_mean[self.active_arms]
        differences = mu_k - mu_l
        idx_l = np.where(self.active_arms == l)[0][0]
        differences = np.delete(differences, idx_l)
        min_diff = np.min(differences)
        beta = sum([self.num_pulls[i]* self.__log_bar(self.num_active_arms) for i in self.active_arms]) / (self.T - sum([self.num_pulls[i] for i in self.non_active_arms]))
        threshold = self.__G_func(beta)
        
        # print(f"CR-C check: Round {self.t}, beta: {beta}, threshold: {threshold}, diff: {min_diff}")
        
        return min_diff > threshold
    
    def best_arm(self):

        if len(self.active_arms) == 0:
            breakpoint()
        return self.active_arms[np.argmax(self.sample_mean[self.active_arms])]
    
    def reset(self):
        self.active_arms = np.arange(self.num_arms)
        self.num_active_arms = len(self.active_arms)
        self.non_active_arms = np.array([])

        self.num_pulls = np.zeros(self.num_active_arms, dtype= float)
        self.sample_mean = np.zeros(self.num_active_arms, dtype= float)
        self.cumulative_rewards = np.zeros(self.num_arms, dtype= float)

        self.t = 0
        
    def get_sample_mean(self):
        
        return self.sample_mean
if __name__ == "__main__":
    bandit = ContinuousRejects(num_arms=10, T=10)
    print(bandit.choose_action())