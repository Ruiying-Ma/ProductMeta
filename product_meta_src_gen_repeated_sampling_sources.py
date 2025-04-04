from ProductMetaSourceGen import Entry, ProductMetaSourceGen
import logging
import logging_config
from utils import write_to_file

class ProductMetaSourceGen_RS_Sources(ProductMetaSourceGen):
    def __init__(self, target, environment, user, tot_llm_call_num, cell_num, srcgen_example_num, update_interval, init_jsonl_path):
        super().__init__(target, environment, user, tot_llm_call_num, cell_num, srcgen_example_num, update_interval, init_jsonl_path)
        
        # self.SRCGEN_PROMPT_TEMPL = '''Provide a **noun** word or a **noun** phrase that has never appeared in the following examples:\n[[examples]]\n\nInclude nothing else in your answer.\n<word/phrase>'''
        self.SRCGEN_PROMPT_TEMPL = '''Provide a **noun** word that has never appeared in the following examples:\n[[examples]]\n\nInclude nothing else in your answer.\n<word>'''

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

        entry.category = 0
        self.archive.add(entry)
        self.entry_counter += 1

        write_to_file(
            dest_path=self.record_jsonl_path,
            contents=entry.to_jsonl() + "\n",
            is_append=True,
            is_json=False
        )

    def _init(self):
        if self.init_jsonl_path != None:
            with open(self.init_jsonl_path, 'r') as file:
                for l in file:
                    entry = Entry.from_jsonl(jsonl=l, recover_embedding=False, recover_theta_and_category=False, base_ax_x=None, base_ax_y=None, cell_num=self.cell_num)
                    entry.category = 0
                    self.archive.add(entry)
                    self.entry_counter += 1
                    write_to_file(
                        dest_path=self.record_jsonl_path,
                        contents=entry.to_jsonl() + "\n",
                        is_append=True,
                        is_json=False
                    )

    def optimize(self):
        self._init()
        while self.entry_counter < self.tot_llm_call_num:
            # select examples
            examples = self.archive.to_list()
            # formulate prompt
            # TODO: This version contains only category. You need to add quality.
            example_str = ", ".join([
                f"{entry.source}"
                for entry in examples
            ])
            prompt = (self.SRCGEN_PROMPT_TEMPL
                    .replace("[[examples]]", str(example_str))
                    )
            answer = self.srcgen_agent.answer(prompt)
            # create entry: update archive (if succeed), entry_counter, llm_call_counter, mutate_iter
            self._create_entry(answer)
        
        write_to_file(
            dest_path=self.statistics_json_path,
            contents=self.to_dict(),
            is_append=False,
            is_json=True
        )

if __name__ == "__main__":
    src_generator = ProductMetaSourceGen_RS_Sources(
        target="kettle",
        environment="",
        user="",
        tot_llm_call_num=400,
        cell_num=360,
        srcgen_example_num=360,
        update_interval=10, # MUST be at least 2
        init_jsonl_path="/home/v-ruiyingma/ProductMeta/init_qd_20250331_125816.jsonl",
    )
    ### CAUTION: SRCGEN_TEMPL/SRCEXP_PROMPT_TEMPL allows phrase or not
    src_generator.optimize()