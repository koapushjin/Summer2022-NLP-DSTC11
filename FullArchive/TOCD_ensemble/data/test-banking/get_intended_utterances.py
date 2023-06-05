import json

L = []

with open('dialogues.jsonl', 'r') as f:
  for dial in f:
    dial = json.loads(dial)
    for i in dial['turns']:
      if i['intents']:
          u = {'dialogue_id': dial['dialogue_id'],
               'turn_id': i['turn_id'],
               'intent': i['intents'],
               'utterance': i['utterance']
               }
          L.append(u)

with open('intended_utterances.jsonl', 'w') as f:
  for i in L:
    f.write(json.dumps(i)+'\n')
