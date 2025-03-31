import os
import json
import logging
import logging_config
from typing import List, Dict
import numpy as np
import random
random.seed(42)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Embedder import SentenceTransformerTextEmbedder
from utils import write_to_file
from Agent import Agent, AgentConfig
from Parser import ProductMetaSrcGenParser, ProductMetaSrcEvalParse
import numpy as np

sbert_embedder = SentenceTransformerTextEmbedder()

def radian_to_degree(rad: float) -> float:
    '''
    rad \in [-1, 1]
    degree \in [0, 360)
    '''
    degree = np.clip((rad + 1) * 180, 0, 360)
    if degree == 360:
        degree = 0
    return float(degree)

def degree_to_radian(deg: float) -> float:
    '''
    degree \in [0, 360)
    rad \in [-1, 1)
    '''
    rad = np.clip((deg / 180) - 1, -1, 1)
    return float(rad)

def radian_to_category(rad: float, cell_num: int) -> int:
    '''
    `rad`: [-1, 1]
    Return `category`: [0, `cell_num` - 1]
    '''
    cat = (rad + 1) * cell_num / 2 # [0, cell_num]
    return int(np.clip(int(cat), 0, cell_num - 1)) # [0, CELL_NUM -1], `CELL_NUM` cells in total

class Entry:
    def __init__(self, id: int, source: str, explanation: str, quality: int, quality_explanation: str):
        self.id = id
        self.source = source.strip().lower()
        self.explanation = explanation # for human analysis: expain how to map source to target
        self.quality = quality
        self.quality_explanation = quality_explanation # explain why rate the source with this quality score

        self.embedding = None # embedding = embedder.encode(self.source)
        self.theta = None
        self.category = None

    def __str__(self):
        return f"(id={self.id}, source={self.source})"
    def __repr__(self):
        return f"(id={self.id}, source={self.source})"
    
    @classmethod
    def from_jsonl(cls, jsonl: str, recover_embedding: bool, recover_theta_and_category: bool, base_ax_x: List[float], base_ax_y: List[float], cell_num):
        entry_dict = json.loads(jsonl)
        entry = Entry(
            id=entry_dict["id"],
            source=entry_dict["source"],
            explanation=entry_dict["explanation"],
            quality=entry_dict["quality"],
            quality_explanation=entry_dict["quality_explanation"]
        )
        if recover_embedding == True:
            entry.set_embedding()
            if recover_theta_and_category == True:
                entry.set_theta(base_ax_x, base_ax_y)
                entry.set_category(cell_num)
        return entry
    
    
    def set_embedding(self):
        assert self.source != None
        self.embedding = sbert_embedder.create_embedding(self.source)
        return self.embedding

    def set_theta(self, base_ax_x: List[float], base_ax_y: List[float]):
        '''
        theta \in [-1, 1]
        '''
        assert self.embedding != None
        if base_ax_x == None or base_ax_y == None:
            dim = sbert_embedder.dim
            base_ax_y = [0 if i != (dim - 1) else 1 for i in range(dim)]
            base_ax_x = [0 if i != (dim - 2) else 1 for i in range(dim)]
        y = np.dot(base_ax_y, self.embedding)
        x = np.dot(base_ax_x, self.embedding)
        phi = np.arctan2(y, x)
        self.theta = float(np.clip(phi/np.pi, -1, 1))
        return self.theta

    def set_category(self, cell_num: int):
        '''
        `theta`: [-1, 1]
        Return `category`: [0, `cell_num` - 1]
        '''
        assert self.theta != None
        cat = (self.theta + 1) * cell_num / 2 # [0, cell_num]
        self.category = int(np.clip(int(cat), 0, cell_num - 1)) # [0, CELL_NUM -1], `CELL_NUM` cells in total
        return self.category
    
    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "explanation": self.explanation,
            "quality": self.quality,
            "quality_explanation": self.quality_explanation
        }
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())
    
class Archive:
    def __init__(self):
        self.archive: Dict[int, List[Entry]] = dict() # category -> [entries in this category]
        # statistics
        self.archive_size: Dict[int, int] = dict() # category -> number of entries in this category
        self.sum_archive_size = 0

    def add(self, entry: Entry):
        assert entry.quality != None
        if entry.category not in self.archive:
            self.archive[entry.category] = []
            self.archive_size[entry.category] = 0
        self.archive[entry.category].append(entry)
        self.archive_size[entry.category] += 1
        self.sum_archive_size += 1

    def to_list(self, category=None) -> List[Entry]:
        '''
        Return the archive as a list of Entry sorted by `(category, entry.quality, entry.id)`
        '''
        archive_list = []
        if category != None:
            assert category in self.archive
            return sorted(self.archive[category], key=lambda e: (e.quality, e.id))
        sorted_category_list = sorted(list(self.archive.keys()))
        for cat in sorted_category_list:
            archive_list += sorted(self.archive[cat], key=lambda e: (-e.quality, e.id))
        assert len(archive_list) == self.sum_archive_size 
        return archive_list
    
    def to_dict(self):
        return {
            "archive": {
                int(cat): [int(e.id) for e in self.to_list(cat)]
                for cat in sorted(list(self.archive.keys()))
            },
            "archive_size": {int(k): v for k, v in self.archive_size.items()},
            "sum_archive_size": self.sum_archive_size,
        }
    
    def _debug_reshape(self, is_before: bool):
        dict_to_print = {
            cat: [f"(id={e.id}, category={e.category})" for e in self.to_list(cat)]
            for cat in sorted(self.archive.keys())
        }
        state = "before" if is_before == True else "after"
        logging.debug(f"Archive {state} reshaping: {dict_to_print}")
    
    def reshape(self):
        new_archive: Dict[int, List[Entry]] = dict()
        new_archive_size: Dict[int, int] = dict()
        for entry_list in self.archive.values():
            for entry in entry_list:
                new_cat = entry.category
                if new_cat not in new_archive:
                    new_archive[new_cat] = []
                    new_archive_size[new_cat] = 0
                new_archive[new_cat].append(entry)
                new_archive_size[new_cat] += 1
        self.archive = new_archive
        self.archive_size = new_archive_size
        assert sum([len(v) for v in list(self.archive.values())]) == self.sum_archive_size
        assert sum(list(self.archive_size.values())) == self.sum_archive_size


class ProductMetaSourceGen:
    record_jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "srcgen_result", "record.jsonl")
    statistics_json_path = record_jsonl_path.replace("record.jsonl", "statistics.json")

    def __init__(
            self,
            # prompt params
            target: str,
            environment: str,
            user: str,
            # budgets
            tot_llm_call_num: int,
            # architecture params
            cell_num: int,
            srcgen_example_num: int,
            update_interval: int,
    ):
        # prompt params
        self.target = target.strip().lower()
        self.environment = environment
        self.user = user
        # budgets
        self.tot_llm_call_num = tot_llm_call_num
        # architecture params
        self.cell_num = cell_num
        self.srcgen_example_num = srcgen_example_num
        self.update_interval = update_interval
        # architecture
        self.archive = Archive()
        self.entry_counter = 0
        self.base_ax_x = None
        self.base_ax_y = None

        # agents
        self.srcgen_agent = Agent(AgentConfig(
            agent_name="SrcGen",
            trial_num=3,
            answer_parser=ProductMetaSrcGenParser(),
            temperature=1.0
        ))
        self.srcexp_agent = Agent(AgentConfig(
           agent_name="SrcExp",
           trial_num=1,
        ))
        self.srceval_agent = Agent(AgentConfig(
            agent_name="SrcEval",
            trial_num=3,
            answer_parser=ProductMetaSrcEvalParse(), # TODO: write this parse in Parser.py
            temperature=0.0
        ))

        # prompts
        # self.SRCGEN_PROMPT_TEMPL = '''Provide a **noun** word or a **noun** phrase that is in the specified category. To achieve this, you should learn from the examples provided below. Each example is presented as a line in the following format:\n<category> ::: <word/phrase>\n- The first field, <category>, is an integer representing the category of the word or the phrase.\n- The second field is the word or the phrase.\n\n**Examples:**\n[[examples]]\n\nAfter studying these examples, provide a new noun word or a new noun phrase that is in the specified category and never appear in the examples:\n<[[category]]> ::: <word/phrase>\n\nInclude nothing else in your answer.''' # TODO: you need to add [[quality]] score to this prompt
        self.SRCGEN_PROMPT_TEMPL = '''Provide a **noun** word that is in the specified category. To achieve this, you should learn from the examples provided below. Each example is presented as a line in the following format:\n<category> ::: <word>\n- The first field, <category>, is an integer representing the category of the word.\n- The second field is the word.\n\n**Examples:**\n[[examples]]\n\nAfter studying these examples, provide a new noun word that is in the specified category and never appear in the examples:\n<[[category]]> ::: <word>\n\nInclude nothing else in your answer.'''
        # self.SRCEXP_PROMPT_TEMPL = f'''You are an expert in product design. Your task is to use the concepts inspired by the given word or phrase to design creative ideas for a [[target]].\n\nAnswer in the following format:\n<The given word or phrase> relates to the concept of <concept 1>.\n<Concept 1> relates to <Concept 2>.\n...\n<Concept n-1> relates to <Concept n>\nInspired by <Concept n>, <your creative ideas for cache replacement policies in a few sentences>.\n\n**The given word or phrase**: [[word]]\n\nDo not include any additional text or explanation in your answer.'''
        self.SRCEXP_PROMPT_TEMPL = f'''You are an expert in product design. Your task is to use the concepts inspired by the given word to design creative ideas for a [[target]].\n\nAnswer in the following format:\n<The given word> relates to the concept of <concept 1>.\n<Concept 1> relates to <Concept 2>.\n...\n<Concept n-1> relates to <Concept n>\nInspired by <Concept n>, <your creative ideas for cache replacement policies in a few sentences>.\n\n**The given word**: [[word]]\n\nDo not include any additional text or explanation in your answer.'''
        self.SRCEVAL_PROMPT_TEMPL = None # TODO: the LLM-as-a-judge rating agent that returns the quality and quality_explanation of a generated source

    def _set_source_quality(self, source: str):
        '''
        Return:
        - quality (int): the score
        - quality explanation (str): the explanation
        '''
        assert source != None
        # TODO: formulate your rating prompt using self.SRCEVAL_PROMPT_TEMPL
        # Then run the following line to get the rating result
        # quality_info = self.srceval_agent.answer(prompt)
        # reutrn quality_info["score"], quality_info["rating_explanation"]
        return 0, None
    
    def _set_source_explanation(self, source: str):
        assert source != None
        prompt = self.SRCEXP_PROMPT_TEMPL.replace("[[word]]", source).replace("[[target]]", self.target)
        return self.srcexp_agent.answer(prompt)

    def _create_entry(self, source: str):
        if source == None:
            return
        
        quality, quality_explanation = self._set_source_quality(source)
        if quality == None:
            return
        explanation = self._set_source_explanation(source)

        entry = Entry(
            id=self.entry_counter,
            source=source,
            explanation=explanation,
            quality=quality,
            quality_explanation=quality_explanation
        )
        logging.info(f"Create entry {str(entry)} with quality {quality}")

        entry.set_embedding()
        entry.set_theta(self.base_ax_x, self.base_ax_y)
        entry.set_category(self.cell_num)

        self.archive.add(entry)
        self.entry_counter += 1

        write_to_file(
            dest_path=self.record_jsonl_path,
            contents=entry.to_jsonl() + "\n",
            is_append=True,
            is_json=False
        )

    def _update(self):
        self.archive._debug_reshape(True)
        entry_list = self.archive.to_list()
        source_list = [e.source for e in entry_list]
        assert all([isinstance(s, str) for s in source_list])
        embedding_list = sbert_embedder.create_embedding(source_list)
        normalized_embedding_list = StandardScaler().fit_transform(embedding_list)
        pca = PCA(n_components=2)
        pca.fit_transform(normalized_embedding_list)
        self.base_ax_x = list(pca.components_[0])
        self.base_ax_y = list(pca.components_[1])
        # update archive
        for entry in entry_list:
            entry.set_theta(self.base_ax_x, self.base_ax_y)
            entry.set_category(self.cell_num)
        self.archive.reshape()
        self.archive._debug_reshape(False)

    def _select_example_from_one_cell(self, category: int, example_num: int) -> List[Entry]:
        '''
        Select a list of examplar entries from the given cell
        '''
        true_example_num = min(example_num, self.archive.archive_size[category] if category in self.archive.archive_size else 0)
        if true_example_num == 0:
            return []
        assert true_example_num > 0
        full_entry_list = self.archive.to_list(category)
        min_quality = full_entry_list[true_example_num - 1].quality
        chosen_entry_list = [e for e in full_entry_list if e.quality > min_quality]
        assert len(chosen_entry_list) < true_example_num
        candid_entry_list = [e for e in full_entry_list if e.quality == min_quality]
        assert len(candid_entry_list) >= true_example_num - len(chosen_entry_list)
        return chosen_entry_list + list(random.sample(candid_entry_list, true_example_num - len(chosen_entry_list)))

    def _select_srcgen_examples(self):
        examples: List[Entry] = []
        for cat in range(self.cell_num):
            examples += self._select_example_from_one_cell(
                            category=cat, 
                            example_num=int(self.srcgen_example_num / self.cell_num) + 1
                        )
        return examples
    
    def _select_explore_srcgen_target_theta(self):
        '''
        Return the target radian for srcgen_agent's prompt
        ''' 
        degree_list = [] # to calculate the cost function
        candid_degree_set = set() # candidates solution for the cost function's maximal
        entries_list = self.archive.to_list()
        for entry in entries_list:
            assert isinstance(entry, Entry) and entry.source != None and entry.theta != None
            degree = radian_to_degree(entry.theta)
            degree_list.append(degree)
            candid_degree_set.add(degree)
            if degree < 180:
                candid_degree_set.add(degree + 180)
            else:
                assert degree >= 180
                candid_degree_set.add(degree - 180)
        max_score = sum([min(abs(0 - dd), 360 - abs(0 - dd)) for dd in degree_list])
        opt_degree = 0
        for d in candid_degree_set:
            if d == 0:
                continue
            score = sum([min(abs(d - dd), 360 - abs(d - dd)) for dd in degree_list])
            if score > max_score:
                max_score = score
                opt_degree = d
        logging.info(f"\topt_degree: {opt_degree}")
        target_degree = np.random.normal(loc=opt_degree, scale=15) # scale = stdev = half_range / 3 = 45 / 3
        while target_degree >= 360 or target_degree < 0:
            if target_degree >= 360:
                target_degree -= 360
            else:
                assert target_degree < 0
                target_degree += 360
        assert target_degree >= 0 and target_degree < 360
        return degree_to_radian(target_degree)

    def optimize(self):
        while self.entry_counter < self.tot_llm_call_num:
            # select examples
            examples = self._select_srcgen_examples()
            # formulate prompt
            ## - target_category
            target_theta = self._select_explore_srcgen_target_theta()
            target_category = radian_to_category(target_theta, self.cell_num)
            ## - example_order
            sorted_examples = sorted(examples, key=lambda entry: (-min(abs(target_theta - entry.theta), 2 - abs(target_theta - entry.theta)), entry.id)) # TODO: you need to sort by quality in increasing order (as the 2nd order sorting)
            ## - formulate
            # TODO: This version contains only category. You need to add quality.
            example_str = "\n".join([
                f"{entry.category} ::: {entry.source}"
                for entry in sorted_examples
            ])
            prompt = (self.SRCGEN_PROMPT_TEMPL
                    .replace("[[examples]]", str(example_str))
                    .replace("[[category]]", str(target_category)))
            answer = self.srcgen_agent.answer(prompt)
            # create entry: update archive (if succeed), entry_counter, llm_call_counter, mutate_iter
            self._create_entry(answer)
            if self.entry_counter % self.update_interval == 0:
                self._update()
        
        write_to_file(
            dest_path=self.statistics_json_path,
            contents=self.to_dict(),
            is_append=False,
            is_json=True
        )

    
    def to_dict(self) -> dict:
        return {
            # prompt params
            "target": self.target,
            "environment": self.environment,
            "user": self.user,
            # budgets
            "tot_llm_call_num": self.tot_llm_call_num,
            # architecture params
            "cell_num": self.cell_num,
            "srcgen_example_num": self.srcgen_example_num,
            "update_interval": self.update_interval,
            # agents
            "srcgen_agent": self.srcgen_agent.to_dict(),
            "srcexp_agent": self.srcexp_agent.to_dict(),
            "srceval_agent": self.srceval_agent.to_dict(),
            "srcgen_prompt_templ": self.SRCGEN_PROMPT_TEMPL,
            "srcexp_prompt_templ": self.SRCEXP_PROMPT_TEMPL,
            "srceval_prompt_templ": self.SRCEVAL_PROMPT_TEMPL,
            # embedder
            "embedder": sbert_embedder.to_dict(),
            # architecture
            "archive": self.archive.to_dict(),
            "entry_counter": self.entry_counter,
            # static
            "record_jsonl_path": self.record_jsonl_path,
            "statistics_json_path": self.statistics_json_path,
        }


if __name__ == "__main__":
    src_generator = ProductMetaSourceGen(
        target="kettle",
        environment="",
        user="",
        tot_llm_call_num=400,
        cell_num=360,
        srcgen_example_num=360,
        update_interval=10 # MUST be at least 2
    )
    ### CAUTION: SRCGEN_TEMPL/SRCEXP_PROMPT_TEMPL allows word or not
    src_generator.optimize()
