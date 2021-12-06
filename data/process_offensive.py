import json
import random
from typing import Text

from torch._C import Value

def convert_to_fever_style(origin_path: Text, fever_style_path: Text) -> None:
    """[summary]
    This is to covert the Mh-RIOT data to a FEVER style
    Args:
        origin_path (Text): path to the original Mh-RIOT data.
        fever_style_path (Text): path to the converted FEVER style data.
    """

    original_file = open(origin_path, 'r', encoding='utf-8')
    fever_style_file = open(fever_style_path, 'w', encoding='utf-8')

    original_data = json.load(original_file)
    id = 0

    # collect the claims and evidences as different sets
    claims = {}
    evidences = {}

    tmp_data = {
        "id": 0,
        "claim": "",
        "evidence": [],
        "label": ""
    }
    # get claims set and evidences set
    for example in original_data:
        claims[id] = example['explicit']
        evidences[id] = [[example['attribute'], random.randrange(10), example['implicit'], random.random()]]
        for step in example['chain']:
            step_tag = step.split('(')[1][:-1]
            step_content = step.split('(')[0][:-1]
            evidences[id].append([example['attribute'], random.randrange(10), step_content, random.random()])
        id += 1

    # generate test data
    # generate SUPPORTS examples
    for key, value in claims.items():
        tmp_data['id'] = key
        tmp_data['claim'] = claims[key]
        tmp_data['evidence'] = evidences[key]
        tmp_data['label'] = 'SUPPORTS'
        json.dump(tmp_data, fever_style_file)
        fever_style_file.write('\n')
    
    # generate REFUTES examples
    for key, value in claims.items():
        tmp_data['id'] = len(claims) + key
        tmp_data['claim'] = claims[key]
        random_evidence = random.randrange(len(claims))
        while random_evidence == key:
            random_evidence = random.randrange(len(claims))
        tmp_data['evidence'] = evidences[random_evidence]
        tmp_data['label'] = 'REFUTES'
        json.dump(tmp_data, fever_style_file)
        fever_style_file.write('\n')
    
    # generate NOT ENOUGH INFO examples
    for key, value in claims.items():
        tmp_data['id'] = 2 * len(claims) + key
        tmp_data['claim'] = claims[key]
        random_removed_evidence = random.randrange(len(evidences[key]))
        evidences[key].pop(random_removed_evidence)
        tmp_data['evidence'] = evidences[key]
        tmp_data['label'] = 'NOT ENOUGH INFO'
        json.dump(tmp_data, fever_style_file)
        fever_style_file.write('\n')

if __name__ == '__main__':
    convert_to_fever_style(r'./offensive_data/origin/non_offensive_grammar.json',
                           r'./offensive_data/FEVER_Style/bert_test.jsonl')