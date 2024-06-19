from abc import ABC, abstractmethod

class BasicBandit(ABC):

    def __init__(self, num_arms) -> None:
        self.num_arms = num_arms
    
    @abstractmethod
    def update(self, play, reward):
        pass
    
    @abstractmethod
    def choose_action(self):
        pass

    @abstractmethod
    def best_arm(self):
        pass

    @abstractmethod
    def reset(self):
        pass