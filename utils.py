import os
import json
from datetime import datetime
import requests
import random

# DICTIONARY_URL = "https://python.sdv.u-paris.fr/data-files/english-common-words.txt"


def write_to_file(dest_path: str, contents, is_append=False, is_json=False):
    '''
    `dest_path`: absolute path
    '''
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if is_append:
        assert is_json == False
        with open(dest_path, 'a') as file:
            file.write(contents)
    else:
        if is_json:
            assert dest_path.endswith(".json")
            with open(dest_path, 'w') as file:
                json.dump(contents, file, indent=4)
        else:
            with open(dest_path, 'w') as file:
                file.write(contents)

def get_dictionary(size: int=None):
    word_list = []
    with open("/home/v-ruiyingma/dict_3000.txt", 'r') as file:
        for l in file:
            word_list.append(l.strip().lower())
    
    if size == None or size > 3000 or size < 0:
        return word_list
    else:
        return random.sample(word_list, int(size))
    
    