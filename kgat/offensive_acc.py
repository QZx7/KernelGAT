import json
from typing import Text, List
from sklearn.metrics import classification_report

def read_jsonl(path: Text) -> List:
    """[summary]
    This is to read a jsonl file
    Args:
        path (Text): The path to the jsonl file.

    Returns:
        List: The data result in the jsonl.
    """

    result = []
    with open(path, 'r', encoding='utf-8') as file:
        for step, line in enumerate(file):
            instance = json.loads(line.strip())
            result.append(instance)
    return result

def get_accuracy(prediction_path: Text, truth_pah: Text) -> None:
    """[summary]
    This is to calcuate the model accuracy on the Mh-RIOT.
    Args:
        prediction_path (Text): the model output file.
        truth_pah (Text): the file of the ground truth.
    """

    label_list = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    predictions_data = read_jsonl(prediction_path)
    truth_data = read_jsonl(truth_pah)
    
    predictions = list()
    truths = list()

    for prediction in predictions_data:
        predictions.append(label_list.index(prediction['predicted_label']))
    
    for truth in truth_data:
        truths.append(label_list.index(truth['label']))

    # print(predictions, truths)
    # print(classification_report(truths[:393], predictions[:393], target_names=label_list))
    print(classification_report(truths, predictions, target_names=label_list))


if __name__ == '__main__':
    get_accuracy(r'./kgat/outputtest.json', r'./data/offensive_data/FEVER_Style/bert_test_non_rr_counter.jsonl')
    
