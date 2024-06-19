from abc import ABC, abstractmethod

class BasicLLM(ABC):

    @abstractmethod
    def get_response(self, prompt):
        pass

    @abstractmethod
    def reset(self):
        pass
