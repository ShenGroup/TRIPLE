from abc import ABC, abstractmethod

class BasicBandit(ABC):

    def __init__(self, num_arms) -> None:
        self.num_arms = num_arms
    
    @abstractmethod
    def updata_phase(self, play, reward):
        pass
    
    @abstractmethod
    def choose_actions(self):
        pass

    @abstractmethod
    def best_arm(self):
        pass

    @abstractmethod
    def reset(self):
        pass