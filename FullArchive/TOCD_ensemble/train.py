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
import argparse

##Paramters and Logger
parser = argparse.ArgumentParser()
parser.add_argument('--num_train_epochs', type=int, required=True)
parser.add_argument('--window_size',type=int,required=True)
parser.add_argument('--starting_epoch',type=int,required=True)
parser.add_argument('--domain',type=str,required=True)
args = parser.parse_args()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger.info(args)
logger.info(torch.version.cuda)
logger.info(torch.__version__)
# train parameters
batch_size = 16
gradient_accumulation_steps = 1
learning_rate = 3e-5
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = args.num_train_epochs
warmup_steps = 0
starting_epoch = args.starting_epoch
seed = 42
domain = args.domain
device = torch.device("cuda")
#device = torch.device('cpu')

# ---------------------------- Get Graph -----------------------------
window_size = args.window_size
dialogue_file = 'data/test-{}/dialogues.jsonl'.format(domain)
graph_object = Graph(window_size, dialogue_file)

graph,L,_ = graph_object.generate_graph()

# ------------------------------- prepare dataset -----------------------------
tokenizer = AutoTokenizer.from_pretrained('models/bert-base-uncased')
max_seq_length = 64
batch_size = batch_size
device = device
utterance_file = 'data/test-{}/all_utterances.jsonl'.format(domain)
train_dataloader = TaskDataLoader(
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
# load state dict for UD
bert_bin_path = "models/all-MiniLM-L6-v2-layer1/pytorch_model.bin"
state_dict = torch.load(bert_bin_path, map_location="cpu")
UttDecoder.bert.load_state_dict(state_dict, strict=False)
# init models
# model = UE_GAT_UD_MLM(UttEncoder, UttDecoder, graph, L, tokenizer).to(device)
model = UE_GAT_UD(UttEncoder, UttDecoder, graph, L, tokenizer).to(device)

# ------------------------------- training -----------------------------
# Loss and Optimizer
t_total = len(train_dataloader) // gradient_accumulation_steps * \
    num_train_epochs
logger.info(f't_total: {t_total}')
#criterion = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
)

model_output_path = 'model_output_L6/{}/tocd-window={}'.format(domain,window_size)
# start training
if starting_epoch != 0:
    state_dict = torch.load('{}/-epoch-{}/pytorch_model.bin'.format(model_output_path,starting_epoch-1))
    model.load_state_dict(state_dict,strict=True)
model.train()
global_step = 0
for epoch in tqdm(range(num_train_epochs)):
    tr_loss = 0.0
    last_tr_loss = 0.0
    local_step = 0
    logger.info(f"epoch: {epoch + starting_epoch}")
    for step, batch in enumerate(train_dataloader):
        global_step += 1
        local_step += 1
        input_ids, attention_mask, labels, dialogue_id, turn_id, utterance = batch
        output, g, turn_id, labels = model(
            input_ids, dialogue_id, turn_id, utterance, attention_mask, labels)
        optimizer.zero_grad()
        logits = output.logits
        B, N, V = logits.size()

        loss = criterion(logits.view(B*N, V), labels.view(B*N))
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        if local_step % 400 == 0 and local_step != 0:
            added_loss = tr_loss - last_tr_loss
            last_tr_loss = tr_loss
            logger.info(
                f"Average loss over Step {local_step-400} ~ Step {local_step} : {added_loss/400}")

    logger.info("Loss of training set: {}".format(tr_loss))
    state_dict = model.state_dict()
    output_dir = "{}/-epoch-{}".format(model_output_path,(epoch+starting_epoch))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    logger.info( "---------------------------Epoch {} finishes-------------------------".format(epoch+starting_epoch))
