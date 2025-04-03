from abc import abstractmethod
import time
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from typing import List

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

class AutoTokenizerTextEmbedder(EmbedderBase):
    def __init__(self, model_name: str="intfloat/e5-small-v2"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dim = 384

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _embed(self, input):
        if isinstance(input, str):
            input_list = [input]
        elif isinstance(input, List):
            input_list = input
        else:
            raise ValueError(f"Unknown input type: {type(input)}")
        
        # Add prefix if not already present
        prefixed_input_list = []
        for t in input_list:
            if not (t.startswith("query: ") or t.startswith("passage: ")):
                prefixed_input_list.append("passage: " + t)
            else:
                prefixed_input_list.append(t)
        
        # Tokenize the inputs
        batch_dict = self.tokenizer(
            prefixed_input_list,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        # Generate embeddings
        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(
            last_hidden_states=outputs.last_hidden_state,
            attention_mask=batch_dict["attention_mask"]
        )

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).tolist()

        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return embeddings


