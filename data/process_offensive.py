import json
import random
from typing import Text, Tuple

import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, PreTrainedTokenizer, PreTrainedModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.openai.tokenization_openai import text_standardize

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

class ParaphrasingModel:
    """The class to initiate a paraphrasing model from the given name
    """

    def __init__(self, name) -> None:
        self.name = name
        self.model = None
        self.tokenizer = None

    def load_paraphrasing_model(self) -> None:
        """[summary]
        This is to load the paraphrasing models.
        Args:
            name (Text): name of the paraphrasing models.
                        current avaliable names: pegasus, t5_paws
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: Return the loaded model and tokenizer
        """
        # for pegasus
        if self.name == "pegasus":
            model_name = 'tuner007/pegasus_paraphrase'
            # torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        elif self.name == "t5_paws":
            model_name = 'Vamsi/T5_Paraphrase_Paws'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def get_paraphrase(self, input: Text) -> Text:
        """[summary]
        This is to get a paraphrase of the input text.
        Args:
            input (Text): the input text

        Returns:
            Text: the returned paraphrasing
        """
        tgt_text = ""
        self.load_paraphrasing_model()
        if self.name == "pegasus":
            batch = self.tokenizer([input],truncation=True,padding='longest',max_length=60, return_tensors="pt")
            translated = self.model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=1, temperature=1.5)
            tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        elif self.name == "t5_paws":
            source = "paraphrase: " + input + " </s>"
            encoding = self.tokenizer.encode_plus(source,pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=256,
                do_sample=True,
                top_k=120,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1
            )
            for output in outputs:
                tgt_text = self.tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        
        return tgt_text


if __name__ == '__main__':
    convert_to_fever_style(r'./offensive_data/origin/non_offensive_grammar.json',
                           r'./offensive_data/FEVER_Style/bert_test.jsonl')