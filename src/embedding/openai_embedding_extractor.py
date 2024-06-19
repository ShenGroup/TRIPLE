import openai
import time
from src.constants import DEFAULT_OPENAI
openai.api_key = DEFAULT_OPENAI.API_KEY

from src.embedding.basic_embedding_extractor import BasicEmbeddingExtractor

class OpenAIExtractor(BasicEmbeddingExtractor):

    def __init__(self, model: str = "text-embedding-ada-002", sleep_interval: int = 0) -> None:
        super().__init__()
        self.model = model
        self.sleep_interval = sleep_interval

    def get_embedding(self, text_to_embed):
        response = openai.Embedding.create(
            model= self.model,
            input=[text_to_embed]
        )
        
        embedding = response["data"][0]["embedding"]

        time.sleep(self.sleep_interval)
        
        return embedding
    
    def reset(self):
        pass