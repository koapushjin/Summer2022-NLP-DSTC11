import datasets
import json
import numpy

from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertModel, BertConfig,AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from gatv2_conv_DGL import GATv2Conv
from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
from tqdm import tqdm

import logging
import random
import numpy as np
import os
import torch
import torch.nn as nn
import math
import dgl
import torch.nn.functional as F

device = torch.device('cuda')

class UE_GAT_UD(nn.Module):
  def __init__(self, UttEncoder, UttDecoder, graph, L,tokenizer):
    super().__init__()
    self.dim_in, self.dim_h, self.dim_out, self.num_heads = 384, 384, 384, 8
    self.ue = UttEncoder
    self.gat1 = GATv2Conv(self.dim_in, self.dim_h, num_heads=self.num_heads)
    self.gat2 = GATv2Conv(self.dim_h*self.num_heads, self.dim_out, num_heads=1)
    self.ud = UttDecoder
    self.graph = graph
    self.all_info = L

  def make_attention_mask(self,ids,tok):
    return torch.log(torch.ne(ids,tok.pad_token_id))

  def forward(self, input_ids, dialogue_id, turn_id, utterance, attention_mask,labels):
    #getting tensors' sizes
    B = input_ids.size()[0]
    M = input_ids.size()[1]

    # Get input for GAT
    edge_idx_x = []
    edge_idx_y = []
    tuple_l = [] # check replication
    for idx_x in range(len(dialogue_id)): # the index x of the sub graph here
      idx_y = len(turn_id) # the index y is corresponding to the newly appended nodes

      for t in self.graph[dialogue_id[idx_x]][turn_id[idx_x]]:
        if t in turn_id:
          tp = (idx_x, idx_x)
          if tp not in tuple_l:
              edge_idx_x.append(idx_x)
              edge_idx_y.append(idx_x)
              tuple_l.append(tp)

        if t not in turn_id:
          tp1 = (idx_x, idx_y)
          tp2 = (idx_y, idx_x)
          if (tp1 not in tuple_l) and (tp2 not in tuple_l):
              edge_idx_x.append(idx_x)
              edge_idx_y.append(idx_y)
              edge_idx_x.append(idx_y)
              edge_idx_y.append(idx_x)
              idx_y += 1

              tuple_l.append(tp1)
              tuple_l.append(tp2)

              # append new nodes
              turn_id.append(t)
              # append new utterance to utterance list
              for info in self.all_info:
                if info['turn_id'] == t:
                  utterance.append(info['utterance'])

    #----------------- get sentence transformer hidden state h_i from the utterance list --------------------
    #logger.info(f'Step 1: getting h_i using sentence transformer:')
    ue_h = self.ue.encode(utterance)

    # get edge index and features
    edge_index = dgl.graph((edge_idx_x,
              edge_idx_y)).to(device)
    features = torch.tensor(ue_h, dtype=torch.float).to(device)
     #----------- GAT feed in and get g_i -----------------
    #logger.info(f'Step 2: getting g_i using GAT:')
    g = F.dropout(features, p=0.6, training=self.training)
    g = self.gat1(edge_index, features)
    g = g.view(-1, self.dim_h*self.num_heads)
    g = F.elu(g)
    g = F.dropout(g, p=0.6, training=self.training)
    g = self.gat2(edge_index, g)
    g = g.mean(dim=1)
    #logger.info(f'INITIAL mean OUTPUT: {g}')

    g = torch.unsqueeze(g[:B],1).repeat(1,M,1) # we get a B*M*H tensor for encoder_hidden_states

    turn_id = turn_id[:B]
    #logger.info(f'Step 3: feed in to the Decoder Model and make cross attention:')
    outputs = self.ud(input_ids=input_ids,
    attention_mask=attention_mask,
    encoder_hidden_states=g,
    encoder_attention_mask=attention_mask,
    return_dict=True)

    return outputs, g, turn_id, labels









class UE_GAT_UD_CONCAT(nn.Module):
  def __init__(self, UttEncoder1, UttEncoder2,UttDecoder, graph, L,tokenizer):
    super().__init__()
    self.dim_in, self.dim_h, self.dim_out, self.num_heads = 384, 384, 384, 8
    self.ue1 = UttEncoder1
    self.ue2 = UttEncoder2
    self.gat1 = GATv2Conv(self.dim_in, self.dim_h, num_heads=self.num_heads)
    self.gat2 = GATv2Conv(self.dim_h*self.num_heads, self.dim_out, num_heads=1)
    self.ud = UttDecoder
    self.graph = graph
    self.all_info = L
    self.dense = nn.Linear(2*self.dim_h,self.dim_h)
    self.relu = nn.ReLU()

  def make_attention_mask(self,ids,tok):
    return torch.log(torch.ne(ids,tok.pad_token_id))

  def forward(self, input_ids, dialogue_id, turn_id, utterance, attention_mask,labels):
    #getting tensors' sizes
    B = input_ids.size()[0]
    M = input_ids.size()[1]

    # Get input for GAT
    edge_idx_x = []
    edge_idx_y = []
    tuple_l = [] # check replication
    for idx_x in range(len(dialogue_id)): # the index x of the sub graph here
      idx_y = len(turn_id) # the index y is corresponding to the newly appended nodes

      for t in self.graph[dialogue_id[idx_x]][turn_id[idx_x]]:
        if t in turn_id:
          tp = (idx_x, idx_x)
          if tp not in tuple_l:
              edge_idx_x.append(idx_x)
              edge_idx_y.append(idx_x)
              tuple_l.append(tp)

        if t not in turn_id:
          tp1 = (idx_x, idx_y)
          tp2 = (idx_y, idx_x)
          if (tp1 not in tuple_l) and (tp2 not in tuple_l):
              edge_idx_x.append(idx_x)
              edge_idx_y.append(idx_y)
              edge_idx_x.append(idx_y)
              edge_idx_y.append(idx_x)
              idx_y += 1

              tuple_l.append(tp1)
              tuple_l.append(tp2)

              # append new nodes
              turn_id.append(t)
              # append new utterance to utterance list
              for info in self.all_info:
                if info['turn_id'] == t:
                  utterance.append(info['utterance'])

    #----------------- get sentence transformer hidden state h_i from the utterance list --------------------
    #logger.info(f'Step 1: getting h_i using sentence transformer:')
    ue_h1 = self.ue1.encode(utterance)
    ue_h2 = self.ue2.encode(utterance)
    # get edge index and features
    edge_index = dgl.graph((edge_idx_x,
              edge_idx_y)).to(device)
    features1 = torch.tensor(ue_h1, dtype=torch.float).to(device)
    features2 = torch.tensor(ue_h2, dtype=torch.float).to(device)
    features = torch.cat((features1,features2),dim=-1)
    features = self.dense(features)
    features = self.relu(features)
     #----------- GAT feed in and get g_i -----------------
    #logger.info(f'Step 2: getting g_i using GAT:')
    g = F.dropout(features, p=0.6, training=self.training)
    g = self.gat1(edge_index, features)
    g = g.view(-1, self.dim_h*self.num_heads)
    g = F.elu(g)
    g = F.dropout(g, p=0.6, training=self.training)
    g = self.gat2(edge_index, g)
    g = g.mean(dim=1)
    #logger.info(f'INITIAL mean OUTPUT: {g}')

    g = torch.unsqueeze(g[:B],1).repeat(1,M,1) # we get a B*M*H tensor for encoder_hidden_states

    turn_id = turn_id[:B]
    #logger.info(f'Step 3: feed in to the Decoder Model and make cross attention:')
    outputs = self.ud(input_ids=input_ids,
    attention_mask=attention_mask,
    encoder_hidden_states=g,
    encoder_attention_mask=attention_mask,
    return_dict=True)

    return outputs, g, turn_id, labels











class UE_GAT_UD_ENSEMBL(nn.Module):
  def __init__(self, UttEncoder, UttDecoder, graph, L,tokenizer,num_intents):
    super().__init__()
    self.dim_in, self.dim_h, self.dim_out, self.num_heads = 384, 384, 384, 8
    self.ue = UttEncoder
    self.gat1 = GATv2Conv(self.dim_in, self.dim_h, num_heads=self.num_heads)
    self.gat2 = GATv2Conv(self.dim_h*self.num_heads, self.dim_out, num_heads=1)
    self.ud = UttDecoder
    self.graph = graph
    self.all_info = L
    self.classifier = nn.Linear(self.dim_out,num_intents)


  def make_attention_mask(self,ids,tok):
    return torch.log(torch.ne(ids,tok.pad_token_id))

  def forward(self, input_ids, dialogue_id, turn_id, utterance, attention_mask,labels):
    #getting tensors' sizes
    B = input_ids.size()[0]
    M = input_ids.size()[1]

    # Get input for GAT
    edge_idx_x = []
    edge_idx_y = []
    tuple_l = [] # check replication
    for idx_x in range(len(dialogue_id)): # the index x of the sub graph here
      idx_y = len(turn_id) # the index y is corresponding to the newly appended nodes

      for t in self.graph[dialogue_id[idx_x]][turn_id[idx_x]]:
        if t in turn_id:
          tp = (idx_x, idx_x)
          if tp not in tuple_l:
              edge_idx_x.append(idx_x)
              edge_idx_y.append(idx_x)
              tuple_l.append(tp)

        if t not in turn_id:
          tp1 = (idx_x, idx_y)
          tp2 = (idx_y, idx_x)
          if (tp1 not in tuple_l) and (tp2 not in tuple_l):
              edge_idx_x.append(idx_x)
              edge_idx_y.append(idx_y)
              edge_idx_x.append(idx_y)
              edge_idx_y.append(idx_x)
              idx_y += 1

              tuple_l.append(tp1)
              tuple_l.append(tp2)

              # append new nodes
              turn_id.append(t)
              # append new utterance to utterance list
              for info in self.all_info:
                if info['turn_id'] == t:
                  utterance.append(info['utterance'])

    #----------------- get sentence transformer hidden state h_i from the utterance list --------------------
    #logger.info(f'Step 1: getting h_i using sentence transformer:')
    ue_h = self.ue.encode(utterance)

    # get edge index and features
    edge_index = dgl.graph((edge_idx_x,
              edge_idx_y)).to(device)
    features = torch.tensor(ue_h, dtype=torch.float).to(device)
     #----------- GAT feed in and get g_i -----------------
    #logger.info(f'Step 2: getting g_i using GAT:')
    g = F.dropout(features, p=0.6, training=self.training)
    g = self.gat1(edge_index, features)
    g = g.view(-1, self.dim_h*self.num_heads)
    g = F.elu(g)
    g = F.dropout(g, p=0.6, training=self.training)
    g = self.gat2(edge_index, g)
    g = g.mean(dim=1)
    #logger.info(f'INITIAL mean OUTPUT: {g}')
    input_to_classifier = g[:B]
    classifier_logits = self.classifier(g[:B])
    g = torch.unsqueeze(g[:B],1).repeat(1,M,1) # we get a B*M*H tensor for encoder_hidden_states

    turn_id = turn_id[:B]
    #logger.info(f'Step 3: feed in to the Decoder Model and make cross attention:')
    outputs = self.ud(input_ids=input_ids,
    attention_mask=attention_mask,
    encoder_hidden_states=g,
    encoder_attention_mask=attention_mask,
    return_dict=True)

    return outputs, g, turn_id, labels,classifier_logits
