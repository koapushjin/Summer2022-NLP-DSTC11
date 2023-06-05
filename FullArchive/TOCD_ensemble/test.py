from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from gatv2_conv_DGL import GATv2Conv
from transformers import BertForMaskedLM,BertGenerationDecoder
from tqdm import tqdm
from models import UE_GAT_UD
from data import Graph, TaskDataLoader
import logging
import numpy as np
import os
import torch
import dgl
import json
from metric import get_metrics
import argparse

##Paramters and Logger
parser = argparse.ArgumentParser()
parser.add_argument('--num_train_epochs', type=int, required=True)
parser.add_argument('--window_size',type=int,required=True)
parser.add_argument('--starting_epoch',type=int,required=True)
parser.add_argument('--domain',type=str,required=True)
parser.add_argument('--num_trials',type=int,required=True)
args = parser.parse_args()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# train parameters
batch_size = 16
num_train_epochs = args.num_train_epochs
starting_epoch = args.starting_epoch
domain = args.domain
num_trials_each_kmeans = args.num_trials
seed = 42
device = torch.device("cuda")

# ---------------------------- Get Graph -----------------------------
window_size = args.window_size
dialogue_file = 'data/test-{}/dialogues.jsonl'.format(domain)
graph_object = Graph(window_size, dialogue_file)
graph, L,result = graph_object.generate_graph()

# ------------------------------- prepare dataset -----------------------------
tokenizer = AutoTokenizer.from_pretrained('models/bert-base-uncased')
max_seq_length = 64
batch_size = batch_size
device = device
utterance_file = 'data/test-{}/intended_utterances.jsonl'.format(domain)
test_dataloader = TaskDataLoader(
    tokenizer, max_seq_length, batch_size, device).generate_dataloader(utterance_file)

# ------------------------------- prepare model -----------------------------
# sub models
# UE: sentence transformer
sentence_transformer_name = 'models/all-MiniLM-L6-v2'
UttEncoder = SentenceTransformer(sentence_transformer_name).to(device)
#UD:BertGeneration
decoderConfig = BertConfig()
decoderConfig.is_decoder, decoderConfig.add_cross_attention = True, True
decoderConfig.hidden_size, decoderConfig.intermediate_size = 384, 1536
UttDecoder = BertGenerationDecoder(config=decoderConfig)
#UD: BertMLM
# decoderConfig = BertConfig()
# decoderConfig.is_decoder, decoderConfig.add_cross_attention = True, True
# decoderConfig.hidden_size, decoderConfig.intermediate_size = 384, 1536
# UttDecoder = BertForMaskedLM(config=decoderConfig)
# load state dict for UD
bert_bin_path = "models/all-MiniLM-L6-v2-layer1/pytorch_model.bin"
state_dict = torch.load(bert_bin_path, map_location="cpu")
UttDecoder.bert.load_state_dict(state_dict, strict=False)
# init models
# model = UE_GAT_UD_MLM(UttEncoder, UttDecoder, graph, L, tokenizer).to(device)
model = UE_GAT_UD(UttEncoder, UttDecoder, graph, L, tokenizer).to(device)

model_output_path = 'model_output_L6/{}/tocd-window={}'.format(domain,window_size)
for epoch in range(num_train_epochs):
    logger.info("Epoch {} started...".format(epoch))
    model_bin_path = "{}/-epoch-{}/pytorch_model.bin".format(model_output_path,(starting_epoch+epoch))
    # load state dict
    state_dict = torch.load(model_bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    for step, batch in enumerate(test_dataloader):
        input_ids, attention_mask, labels,dialogue_id, turn_id, utterance = batch
        sudo_labels = input_ids
        output, g, turn_id,_ = model(input_ids, dialogue_id, turn_id, utterance, attention_mask,sudo_labels)
        logger.info(g.size())
        logger.info(turn_id)
        g = g.tolist()

        for i in range(len(turn_id)):
            for res in result:
                if res["turn_id"] == turn_id[i]:
                    res["hidden_state"] = g[i][0]

    logger.info(result)
    result_path = '{}/-epoch-{}/result.jsonl'.format(model_output_path,(starting_epoch+epoch))
    with open(result_path,'w') as f:
        for ex in result:
            f.write(json.dumps(ex)+'\n')
    get_metrics(result_path,domain,num_trials_each_kmeans)
    logger.info("Epoch {} finished...".format(epoch))
