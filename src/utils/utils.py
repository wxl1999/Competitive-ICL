import json
import string

def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)

    return data

# ensure sentences end with punctuation.
def add_period_for_lmbff(sentence):
    sentence = sentence.rstrip()
    if sentence[-1] not in string.punctuation:
        sentence = sentence.rstrip() + "."
    return sentence