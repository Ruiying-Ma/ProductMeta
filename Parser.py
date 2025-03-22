import regex
from abc import ABC, abstractmethod
import json
import re

class BaseParser(ABC):
    def __init__(self):
        self.error = ""
        self.name = ""
    @abstractmethod
    def parse(self, input: str):
        '''
        Parse the input. If fail, return `None`.
        '''
        pass

    def to_dict(self) -> dict:
        return {
            "name": self.name
        }

    def print_error(self) -> str:
        return f"Parser-{self.name} Error: {self.error}"
    

class DefaultParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.name = "Default"

    def parse(self, input: str):
        return input
    
class JsonParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.name = "Json"
    def parse(self, input: str):
        self.error = ""
        try:
            json_pattern = r'\{(?:[^{}]|(?R))*\}'
            json_match = regex.search(json_pattern, input)
            if json_match == None:
                raise TypeError
            json_str = json_match.group()
            extracted_json = json.loads(json_str)
            return extracted_json
        except Exception:
            self.error = "[No JSON found]"
            return None
    
class ProductMetaSrcGenParser(BaseParser): # TODO: Or you may use JsonParser if you ask LLM to output JSON format
    def __init__(self):
        super().__init__()
        self.name = "ProductMetaSrcGen"

    def parse(self, input: str):
        if input == None:
            return None
        parts = input.split(":::")
        if len(parts) == 1:
            return input.strip().lower()
        return parts[-1].strip().lower()
    
class ProductMetaSrcEvalParse(BaseParser):
    def __init__(self):
        super().__init__()
        self.name = "ProductMetaSrcEvalParser"
    
    def parse(self, input: str):
        '''
        Args:
        - `input`: the raw response from LLM agent
        Return:
        - `result` (dict): {"rating_explanation": ..., "score": ....}
        '''
        if input == None:
            self.error = "[Null LLM Response]"
            return None
        # TODO
        # parse the score from `input`
        score = None # TODO
        # if parsing fails (i.e., score == None), return None
        if score == None:
            self.error = "[Invalid score]"
            return None
        # parse the explanation of the rating from `input`
        rating_explanation = None #TODO
        return {
            "rating_explanation": rating_explanation,
            "score": score
        }
