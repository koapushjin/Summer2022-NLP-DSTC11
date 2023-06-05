import json
import random
import numpy as np
import torch
import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
import logging
#logger = logging.getLogger(__name__)
class Graph:
    def __init__(self,window_size,dialogue_file):
        self.window_size = window_size
        self.graph = {}
        self.L = []
        self.result_for_test = []
        self.dialogue_file = dialogue_file
    def generate_graph(self):
        N = self.window_size # window size
        with open (self.dialogue_file) as f:
            l = list(f)
        for dial in l:
            dial = json.loads(dial)
            dial_id = dial['dialogue_id']
            current_x = 0
            dial_dict = {}
            for turn in dial['turns']:
                turn_id = turn['turn_id']
                for i in range(N+1):
                    current_y = current_x - i
                    if current_y >= 0:
                        try:
                            dial_dict[turn_id].append(dial['turns'][current_y]['turn_id'])
                        except KeyError:
                            dial_dict[turn_id] = [dial['turns'][current_y]['turn_id']]
                for i in range(N+1):
                    current_y = current_x + i
                    if current_y < len(dial['turns']):
                        if dial['turns'][current_y]['turn_id'] not in dial_dict[turn_id]:
                            dial_dict[turn_id].append(dial['turns'][current_y]['turn_id'])

                current_x+=1
                utt = turn['utterance']
                role = turn['speaker_role']
                if len(turn['intents']) > 0:
                    intent = turn['intents'][0]
                    k = {"dialogue_id":dial_id,"turn_id":turn_id,"utterance":utt}
                    self.result_for_test.append(k)
                else:
                    intent = ''


                d = {"dialogue_id":dial_id,"turn_id":turn_id,"intent":intent,"utterance":utt,"role":role}
                self.L.append(d)

            self.graph[dial_id] = dial_dict
        return self.graph,self.L,self.result_for_test

class TaskDataLoader:
    def __init__(self,tokenizer,max_seq_length,batch_size,device):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self.train_data, self.train_dataloader = None,None

    def process_data_to_model_inputs(self,batch):
        # Tokenize the input and target data
        inputs = self.tokenizer(batch["utterance"], padding="max_length", truncation=True, max_length=self.max_seq_length)
        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        max_len = max([len(example) for example in batch['input_ids']])

        prepare_output = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in batch["input_ids"]]
        x = np.concatenate(([input_ids[1:] for input_ids in prepare_output],
                    [[-100 for i in range(max_len - len(example) + 1)] for example in batch["input_ids"]]),axis=1)

        batch["labels"] = x
        return batch

    def collate_fn(self,batch):
        input_ids = [seq['input_ids'] for seq in batch]
        attention_mask = [seq['attention_mask'] for seq in batch]
        labels = [seq['labels'] for seq in batch]
        dialogue_id = [seq['dialogue_id'] for seq in batch]
        turn_id = [seq['turn_id'] for seq in batch]
        utterance = [seq['utterance'] for seq in batch]

        input_ids = torch.from_numpy(np.asarray(input_ids))
        attention_mask = torch.from_numpy(np.asarray(attention_mask))
        labels = torch.from_numpy(np.asarray(labels))
        device = self.device
        return input_ids.to(device), attention_mask.to(device), labels.to(device), dialogue_id, turn_id, utterance

    def generate_dataloader(self,utterance_file):
        utt_dataset = datasets.load_dataset('json', data_files=utterance_file)['train']
        tokenizer= self.tokenizer
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token

        self.train_data = utt_dataset.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size
            )
        self.train_dataloader = DataLoader(
            self.train_data,
            sampler=RandomSampler(self.train_data),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
            )
        return self.train_dataloader
