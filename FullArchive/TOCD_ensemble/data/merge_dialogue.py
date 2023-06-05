import json



def merge():
    dialogues_files = ["banking_dialogues.jsonl","finance_dialogues.jsonl","insurance_dialogues.jsonl"]
    all_dialogues = []
    for dialogues_file in dialogues_files:
        with open(dialogues_file) as fIn:
            dialogues = list(fIn)
            for i in range(len(dialogues)):
                dialogues[i] = json.loads(dialogues[i])
                all_dialogues.append(dialogues[i])
    with open("dialogues.jsonl","w") as fOut:
        for dialogue in all_dialogues:
            fOut.write(json.dumps(dialogue)+"\n")

def get_all_utterances():
    dialogues_files = ["banking_dialogues.jsonl","finance_dialogues.jsonl","insurance_dialogues.jsonl"]
    utts = []
    for file in dialogues_files:
        with open(file) as fIn:
            L = list(fIn)
            for i in range(len(L)):
                L[i] = json.loads(L[i])

        for dialogue in L:
            dialogue_id = dialogue["dialogue_id"]
            for turn in dialogue["turns"]:
                if turn["intents"]:
                    utterance = {"turn_id":turn["turn_id"],"intent":turn["intents"][0],"utterance":turn["utterance"],"dialogue_id":dialogue_id}
                    utts.append(utterance)
                else:
                    utterance = {"turn_id":turn["turn_id"],"intent":None,"utterance":turn["utterance"],"dialogue_id":dialogue_id}
                    utts.append(utterance)
    with open("all_utterances.jsonl","w") as fOut:
        for utt in utts:
            fOut.write(json.dumps(utt)+"\n")

get_all_utterances()
