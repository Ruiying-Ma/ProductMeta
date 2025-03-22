import os
from openai import AzureOpenAI
import time
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from Parser import BaseParser, DefaultParser
from datetime import datetime
from utils import write_to_file
import base64
from mimetypes import guess_type

class AgentConfig:
    def __init__(self, agent_name: str="Default", env_path: str=None, trial_num: int=None, answer_parser: BaseParser=DefaultParser(), temperature: float=0.0, max_tokens: int=None, agent_type: str="text"):
        '''
        `env_path`: default as `.env` under the root directory
        `trial_num`: default as 1
        '''
        self.agent_name: str = agent_name
        if trial_num == None or trial_num <= 0:
            trial_num = 1
        self.trial_num = trial_num
        if env_path == None or (not os.path.exists(env_path)):
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        self.env_path = env_path
        self.answer_parser = answer_parser
        self.temperature = temperature
        self.max_tokens = max_tokens
        if agent_type not in ["text", "image"]:
            raise ValueError(f"Unknown agent type: {agent_type}")
        self.agent_type = agent_type

class Agent:
    def __init__(self, agent_config: AgentConfig):
        # basic
        self.name: str = agent_config.agent_name
        self.log_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "agent", self.name+".log")
        self.trial_num = agent_config.trial_num
        self.answer_parser = agent_config.answer_parser
        self.temperature = agent_config.temperature
        self.max_tokens = agent_config.max_tokens
        self.agent_type = agent_config.agent_type
        # statistics
        self.input_token_num: int = 0
        self.output_token_num: int = 0
        self.latency: float = 0.0
        # client
        load_dotenv(agent_config.env_path)
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        if self.agent_type == "text":
            self.model = os.getenv("AZURE_OPENAI_MODEL_GPT")
        else:
            assert self.agent_type == "image"
            self.model = os.getenv("AZURE_OPENAI_MODEL_DALLE")

    def _serialize_image(self, img_path: str):
        mime_type, _ = guess_type(img_path)
        if mime_type == None:
            mime_type = "application/octet-stream"
        with open(img_path, 'rb') as img_file:
            serialized_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{serialized_data}"

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _call(self, message):
        if self.agent_type == "text":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response
        else:
            assert self.agent_type == "image"
            response = self.client.images.generate(
                model=self.model,
                prompt=message,
                size="1024x1024",
                quality="standard",
                n=1, # number of images to generate
            )
            return response

    def answer(self, prompt: str, img_path_list: list=None):
        '''
        Return:  `None` if fail. Else, return the parsed answer according to function `parse_answer()`
        '''
        if img_path_list == None or len(img_path_list) == 0:
            if self.agent_type == "text":
                message = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            else:
                assert self.agent_type == "image"
                message = prompt
        else:
            assert self.agent_type == "text"
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
            message[0]["content"] += [{
                "type": "image_url",
                "image_url": {
                    "url": self._serialize_image(ip),
                    "detail": "auto"
                }
            }
            for ip in img_path_list]

        trial_num = self.trial_num
        while trial_num > 0: 
            start = time.time()
            try:
                response = self._call(message)
            except Exception:
                trial_num -= 1
                end = time.time()
                self.latency += end - start
                continue
            end = time.time()
            # update statistics
            self.latency += end - start
            if self.agent_type == "text":
                self.input_token_num += response.usage.prompt_tokens
                self.output_token_num += response.usage.completion_tokens
            else:
                assert self.agent_type == "image"
                self.input_token_num += len(message) # number of characters
                self.output_token_num += 1 # number of images
            # log this conversation
            log_response = str(response.choices[0].message.content) if self.agent_type == "text" else str(response.data[0].url)
            log_input = str(response.usage.prompt_tokens) if self.agent_type == "text" else str(len(prompt))
            log_output = str(response.usage.completion_tokens) if self.agent_type == "text" else str(1)
            write_to_file(
                dest_path=self.log_path,
                contents=(self._log()
                        .replace("<prompt>", str(prompt))
                        .replace("<response>", log_response)
                        .replace("<input>", log_input)
                        .replace("<output>", log_output)
                        .replace("<time>", str(end - start))),
                is_append=True,
                is_json=False
            )
            # parse the answer
            if self.agent_type == "text":
                answer = self.answer_parser.parse(response.choices[0].message.content)
            else:
                assert self.agent_type == "image"
                answer = response.data[0].url
            if answer != None:
                return answer
            # if fail, reduce trial_num
            trial_num -= 1
        return None
    
    def _log(self):
        return f"**************************************************{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**************************************************\n" + f"[Prompt]\n<prompt>\n[{self.name} Response]\n<response>\n[Model, Temperature, MaxTokens]{self.model}, {self.temperature}, {self.max_tokens}\n[Input, Output, Time]<input>, <output>, <time>" + "\n=======================================================================================================================\n\n"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trial_num": self.trial_num,
            "answer_parser": self.answer_parser.to_dict(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "input_token_num": self.input_token_num,
            "output_token_num": self.output_token_num,
            "latency": self.latency,
            "model": self.model
        }