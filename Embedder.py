from abc import abstractmethod
import time
from sentence_transformers import SentenceTransformer

class EmbedderBase:
    def __init__(self):
        self.latency = 0.0

    @abstractmethod
    def _embed(self, input):
        '''
        Return the embedding
        '''
        pass

    def create_embedding(self, input):
        start = time.time()
        embedding = self._embed(input)
        end = time.time()
        self.latency += end - start
        return embedding
    
    def to_dict(self):
        return {
            "latency": self.latency
        }
    
class SentenceTransformerTextEmbedder(EmbedderBase):
    def __init__(self, model_name: str="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = 768
    
    def _embed(self, input):
        return self.model.encode(input).tolist()
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "embedder": "SentenceTransformer",
            "model_name": self.model_name,
        })