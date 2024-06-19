
from src.bandit import BasicBandit
import math
import numpy as np
# Code for bandit algorithm successive rejects to pull the bandit



class SuccessiveRejects(BasicBandit):
    def __init__(self, num_arms: int = 0, T: int = 0) -> None:
        super().__init__(num_arms)
        # Add any additional initialization code here
        self.num_arms = num_arms
        self.T = T
        self.log_bar_k = 1/2 + sum([1/i for i in range(2, num_arms+1)])
        self.cur_round = 1
        self.cur_arm = 0
        self.n_k = [0] * (num_arms)
        self.__init_n_k() 
        self.active_arms = [i for i in range(num_arms)]
        self.cumulative_rewards = [0 for _ in range(num_arms)]
        self.rewards = [0 for _ in range(num_arms)]
        self.num_pulls = [0 for _ in range(num_arms)]
        
        
    def __init_n_k(self):
        self.n_k[1] = math.ceil(self.T/(self.num_arms) * (1/self.log_bar_k)) 
        check = self.n_k[1] * self.num_arms
        for i in range(2, self.num_arms):
            # breakpoint()
            n_k_i = math.ceil((self.T - self.num_arms)/(self.num_arms + 1 - i) * (1/self.log_bar_k)) - math.ceil((self.T - self.num_arms)/(self.num_arms + 2 - i) * (1/self.log_bar_k))
            self.n_k[i] = max(0, n_k_i)  # Fix potential list overflow issue
            check += self.n_k[i] * (self.num_arms - i + 1)
            print(f"check: {check}")
        
    def choose_action(self):
        # Implement the arm selection logic here
        # if not at the last arm and the number of pulls is less than the number of desired pulls
        if self.num_pulls[self.cur_arm] < self.n_k[self.cur_round]:
            return self.active_arms[self.cur_arm]
        elif self.cur_arm < self.num_arms - 1:
            self.cur_arm += 1
            return self.active_arms[self.cur_arm]
        elif len(self.active_arms) > 1:
            self.cur_round += 1
            self.cur_arm = 0
            self.num_pulls = [0 for _ in range(self.num_arms)]
            self.__reject_arm()
            while self.n_k[self.cur_round] == 0:
                self.cur_round += 1
                self.__reject_arm()
            return self.active_arms[self.cur_arm]
        else:
            return self.active_arms[0]
            
            
            
    def __reject_arm(self):
        # Find the arm with the lowest cumulative reward
        # remove it from the active arms
        # reduce the number of arms by 1
        worst_arm = min(range(self.num_arms), key=lambda arm: self.cumulative_rewards[arm])
        print(f"Rejected arm: {self.active_arms[worst_arm]}, current active arms: {self.active_arms}, current cumulative rewards: {self.cumulative_rewards}")
        self.active_arms.pop(worst_arm)
        self.cumulative_rewards.pop(worst_arm)
        self.num_arms -= 1

    def best_arm(self):
        # Implement the logic to return the best arm
        return self.active_arms[-1]

    def update(self, arm, reward):
        # Implement the update logic here
        arm_pos = self.active_arms.index(arm)
        self.cumulative_rewards[arm_pos] += reward
        self.num_pulls[arm] += 1
        self.rewards[arm] += reward
    
    def get_sample_mean(self):
        rewards = np.array(self.rewards, dtype=float)
        num_pulls = np.array(self.num_pulls, dtype=float)
        sample_mean = np.divide(rewards, num_pulls, out=np.zeros_like(rewards), where=num_pulls!=0)
        return sample_mean
        
    def reset(self):
        # Implement the reset logic here
        self.cur_round = 1
        self.cur_arm = 0
        self.num_pulls = [0 for _ in range(self.num_arms)]
        self.active_arms = [i for i in range(self.num_arms)]
        self.cumulative_rewards = [0 for _ in range(self.num_arms)]
        
        

        

    # # Add any additional methods or overrides as needed
    # def __init__(self, num_arms: int = 0, T: int = 0) -> None:
    #     super().__init__(num_arms)
    #     # Add any additional initialization code here
    #     self.num_arms = num_arms
    #     self.T = T
    #     self.log_bar_k = 1/2 + sum([1/i for i in range(2, num_arms+1)])
    #     self.cur_round = 1
    #     self.cur_arm = 0
    #     self.n_k = [0] * num_arms
    #     self.__init_n_k() 
    #     self.active_arms = [i for i in range(num_arms)]
    #     self.cumulative_rewards = [0 for _ in range(num_arms)]
    #     self.num_pulls = [0 for _ in range(num_arms)]
        
        
    # def __init_n_k(self):
        
    #     self.n_k[1] = math.ceil(self.T/(self.num_arms) * (1/self.log_bar_k))
    #     for i in range(2, self.num_arms):
    #         self.n_k[i] = math.ceil((self.T - self.num_arms)/(self.num_arms + 1 - i) * (1/self.log_bar_k)) 
    #         - math.ceil((self.T - self.num_arms)/(self.num_arms + 1 - i + 1) * (1/self.log_bar_k)) 
        
    # def choose_action(self):
    #     # Implement the arm selection logic here
    #     # if not at the last arm and the number of pulls is less than the number of desired pulls
    #     if self.num_pulls[self.cur_arm] < self.n_k[self.cur_arm]:
    #         return self.active_arms[self.cur_arm]
    #     elif self.cur_arm < self.num_arms:
    #         self.cur_arm += 1
    #         return self.active_arms[self.cur_arm]
    #     else:
    #         self.cur_round += 1
    #         self.cur_arm = 0
    #         self.num_pulls = [0 for _ in range(self.num_arms)]
    #         self.__reject_arm()
    #         return self.active_arms[self.cur_arm]
            
            
            
    # def __reject_arm(self):
    #     # Find the arm with the lowest cumulative reward
    #     # remove it from the active arms
    #     # reduce the number of arms by 1
    #     worst_arm = min(range(self.num_arms), key=lambda arm: self.cumulative_rewards[arm])
    #     self.active_arms.remove(worst_arm)
    #     self.cumulative_rewards.pop(worst_arm)
    #     self.num_arms -= 1
        
        
    # def best_arm(self):
    #     # Implement the logic to return the best arm
    #     return self.active_arms[0]

    # def update(self, arm, reward):
    #     # Implement the update logic here
    #     self.cumulative_rewards[arm] += reward
    #     self.num_pulls[arm] += 1
        
    # def reset(self):
    #     # Implement the reset logic here
    #     self.cur_round = 1
    #     self.cur_arm = 0
    #     self.num_pulls = [0 for _ in range(self.num_arms)]
    #     self.active_arms = [i for i in range(self.num_arms)]
    #     self.cumulative_rewards = [0 for _ in range(self.num_arms)]
        

    # Add any additional methods or overrides as needed
