from abc import ABC, abstractmethod

class BasicEmbeddingExtractor(ABC):

    @abstractmethod
    def get_embedding(self, text_to_embed: str):
        pass

    @abstractmethod
    def reset(self):
        pass