import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification,RobertaConfig
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import shutil
import random
import datasets
import numpy as np
import logging
import sys
import os
import math
import json
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


# Set up for dataset and training
gradient_accumulation_steps = 1
learning_rate = 3e-5
adam_epsilon = 1e-8
max_grad_norm = 1.0
num_train_epochs = 10
warmup_steps = 0
starting_epoch = 0
seed = 42
batch_size = 8
max_seq_length = 382
device = torch.device("cuda")

# domain = "banking"
domain = "finance"
# domain = "insurance"
num_models = 10
folder_name = domain+"_K="+str(num_models)+"_new"
model_path = "roberta-base"
output_path = folder_name+"/xvali-outs-"

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def train(train_dataset,test_dataset, model,tokenizer, output_path,order):
    logger.info("A new model, Model {}, begins training process.....................".format(order))
    # logger.info("Length of the dataset: {}".format(len(train_dataset)))
    # logger.info("See the dataset:{}".format(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
        # collate_fn=train_dataset.collate_fn
    )

    # Loss and Optimizer
    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    criterion = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    logger.info(f"learning rate: {learning_rate}")
    sm = torch.nn.Softmax(dim=-1)

    # Training!
    output_root = output_path
    samples = []

    # Start training
    model.train()
    global_step = 0
    for epoch in tqdm(range(num_train_epochs)):
        tr_loss = 0.0
        logger.info(f"epoch:{epoch + starting_epoch}")
        local_step = 0
        corr = 0
        total_q = 0
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            local_step += 1

            # get context and target label

            input_ids, labels,dial_idxs,turn_idxs,attention_mask = batch["input_ids"].to(device),batch["predicted_label"].to(device),batch["dialogue_idx"].to(device),batch["turn_idx"].to(device),batch["attention_mask"].to(device)
            # zero grad
            optimizer.zero_grad()

            # logits
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)

            # get shapes

            logits = outputs.logits
            B, V = logits.size()


            softmax_logits = sm(logits)

            predicted = torch.argmax(softmax_logits,dim=-1)
            predicted_p = torch.max(softmax_logits,dim=-1).values
            for i in range(len(predicted)):
                dial_idx = int(dial_idxs[i])
                turn_idx = int(turn_idxs[i])
                id = domain+"_"+f"{dial_idx:04d}"+"_"+f"{turn_idx:03d}"
                samples.append({"utterance":str(tokenizer.decode(input_ids[i])).strip("<pad>"),"classified_into":int(predicted[i]),"probability":float(predicted_p[i]),"id":id,"sudo_label":int(labels[i]),"sudo_label_prob":float(softmax_logits[i,int(labels[i])])})
                if int(labels[i]) == int(predicted[i]):
                    corr += 1
                total_q += 1


            loss = criterion(logits.view(B, V), labels.view(B))

            # accumulate total loss
            tr_loss += loss.item()

            # backprop
            loss.backward()

            # apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # model update
            optimizer.step()
            scheduler.step()

        ##Print loss & accuracy info
        logger.info("Loss of training set: {}".format(tr_loss))
        logger.info("Accuracy of training set classification due to sudo label: {}".format(corr/total_q))
        # save model
        state_dict = model.state_dict()
        output_dir = output_root
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        logger.info("-------------------------Now testing---------------------------------------")
        with open(folder_name+"/xvali-outs-"+str(order)+"/train_results.json","w") as j:
            json.dump(samples,j)
        test(test_dataset,model,tokenizer,order,epoch)
    logger.info("----------------------------One model finishes-------------------------")
def test(test_dataset,test_model,tokenizer,order,epoch):
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size)


    total_loss = 0.0
    total_tokens = 0
    test_model.eval()
    criterion1 = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
    sm = torch.nn.Softmax(dim=-1)


    samples = []
    with torch.no_grad():
        total_q = 0
        corr = 0
        for step,batch in enumerate(test_dataloader):

            # get context and target label
            input_ids, labels,dial_idxs,turn_idxs,attention_mask = batch["input_ids"].to(device),batch["predicted_label"].to(device),batch["dialogue_idx"].to(device),batch["turn_idx"].to(device),batch["attention_mask"].to(device)

            outputs = test_model(input_ids=input_ids,attention_mask=attention_mask)

            # get shapes
            logits = outputs.logits

            softmax_logits = sm(logits)

            predicted = torch.argmax(softmax_logits,dim=-1)
            predicted_p = torch.max(softmax_logits,dim=-1).values



            for i in range(len(predicted)):
                dial_idx = int(dial_idxs[i])
                turn_idx = int(turn_idxs[i])
                id = domain+"_"+f"{dial_idx:04d}"+"_"+f"{turn_idx:03d}"
                samples.append({"utterance":str(tokenizer.decode(input_ids[i])).strip("<pad>"),"classified_into":int(predicted[i]),"probability":float(predicted_p[i]),"id":id,"sudo_label":int(labels[i]),"sudo_label_prob":float(softmax_logits[i,int(labels[i])])})
                if int(labels[i]) == int(predicted[i]):
                    corr += 1
                total_q += 1


            # get shapes
            B, V= logits.size()

            # calculate sum loss. Notice criterion1 is using "sum" instead of "mean"
            # flatten logits and lm_labels into 2-D tensor and 1-D tensor respectively.
            # i.e. (B*N, V) for the predicts, labels (B*N)
            loss = criterion1(logits.view(B, V), labels.view(B))

            # accumulate total loss
            total_loss += loss.item()

        ##Print loss & accuracy info
        logger.info("Loss of testing set: {}".format(total_loss))
        logger.info("Accuracy of testing set classification due to sudo label: {}".format(corr/total_q))
        logger.info("-----------------------------------------------------------------")
    with open(folder_name+"/xvali-outs-"+str(order)+"/test_results.json","w") as j:
        json.dump(samples,j)


def main():

    ##Create folder and copy the raw predictions to it
    os.makedirs(folder_name+"/",exist_ok = True)
    shutil.copy("predictions_{}_qr.json".format(domain),folder_name+"/predictions.json")

    active_intents = []
    ##Turn a raw prediction to what I want
    with open(folder_name+"/predictions.json") as j:
        L = list(j)
        for i in range(len(L)):
            L[i] = json.loads(L[i])
        for i in range(len(L)):
            L[i]['predicted_label'] = int(L[i]['predicted_label'])
            if L[i]['predicted_label'] not in active_intents:
                active_intents.append(L[i]['predicted_label'])
            L[i]['utterance'] = L[i]['utterance'].replace('[SEP]','</s></s>')
            id = L[i]['turn_id']
            dial_idx = int(id.split('_')[1])
            turn_idx = int(id.split('_')[-1])
            L[i]['dialogue_idx'] = dial_idx
            L[i]['turn_idx'] = turn_idx
    num_labels = len(active_intents)
    xpred = {'version':'kmeans_model1+model2_0','data':L}
    with open(folder_name+"/xpredictions-0.json",'w') as j:
        json.dump(xpred,j)


    num_iter = 10
    for iteration in range(0,num_iter):
        # ##Which iteration?
        iteration = iteration + 1

        config = RobertaConfig()

        data_file_path = folder_name+"/xpredictions-"+str(iteration-1)+".json"

        ##Initialize models and tokenizers
        model = RobertaForSequenceClassification(config)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)

        ##Load dataset with 10-fold method
        split_param = int(100/num_models)
        tests_ds = datasets.load_dataset("json", data_files = data_file_path,field="data",split=[f"train[{k}%:{k+split_param}%]" for k in range(0, 100, split_param)])
        trains_ds = datasets.load_dataset("json", data_files = data_file_path,field="data",split=[f"train[:{k}%]+train[{k+split_param}%:]" for k in range(0, 100, split_param)])
        train_sets = []
        test_sets = []
        for train_dataset in trains_ds:
            train_dataset = train_dataset.map(lambda examples: tokenizer(examples["utterance"],padding="max_length",truncation=True, max_length=max_seq_length), batched=True)
            train_dataset.set_format(type="torch", columns=["input_ids", "predicted_label", "dialogue_idx","turn_idx","attention_mask"])
            train_sets.append(train_dataset)
        for test_dataset in tests_ds:
            test_dataset = test_dataset.map(lambda examples: tokenizer(examples["utterance"],padding="max_length"), batched=True)
            test_dataset.set_format(type="torch", columns=["input_ids", "predicted_label", "dialogue_idx","turn_idx", "attention_mask"])
            test_sets.append(test_dataset)

        ##Train & Eval the model
        for order in range(num_models):
            model = model.from_pretrained(model_path,num_labels=num_labels,ignore_mismatched_sizes=True)
            model.to(device)
            train(train_sets[order],test_sets[order], model,tokenizer, output_path+str(order),order)

        ##Merge the classification results from 10 test sets
        samples = []
        for i in range(num_models):
            with open(folder_name+"/xvali-outs-"+str(i)+"/"+"test_results.json") as j:
                L = json.load(j)
                for l in L:
                    samples.append(l)


        with open(folder_name+"/classified-"+str(iteration)+".json","w") as j:
            json.dump(samples,j)

        ##Generate the classification report
        with open(folder_name+"/classified-"+str(iteration)+".json") as j:
            L = json.load(j)

        corr = 0
        wrong = 0
        total = len(L)
        corr_s = 0
        wrong_s = 0
        wrong_s_2 = 0
        corr_items = []
        wrong_items = []
        for d in L:
            if d["classified_into"] == d["sudo_label"]:
                corr += 1
                corr_s += d["probability"]
                corr_items.append({"utterance":d["utterance"],"id":d["id"],"sudo-label":d["sudo_label"],"classified":d["classified_into"],\
                    "classfied_prob":d["probability"]})
            else:
                wrong += 1
                wrong_s += d["probability"]
                wrong_s_2 += d["sudo_label_prob"]
                wrong_items.append({"utterance":d["utterance"],"id":d["id"],"sudo-label":d["sudo_label"],"classified":d["classified_into"],\
                    "sudo_label_prob":d["sudo_label_prob"],"classified_prob":d["probability"]})
        D = {"Totol Number":total,"Total Correct":corr,"Proportion of Correct":corr/total, \
        "Avg Probability among Correct Samples":corr_s/corr,"Total Wrong":wrong,"Proportion of Wrong":wrong/total,\
            "Avg Probability of Sudo-Label among Wrong Samples":wrong_s_2/wrong,\
                "Avg Probability of Classified Group among Wrong Samples":wrong_s/wrong,\
                    "Correct Items":corr_items,\
                        "Wrong Items":wrong_items}
        with open(folder_name+"/classifier_report-"+str(iteration)+".json","w") as j:
            json.dump(D,j)

        #Rewrite prediction.json and xprediction.json
        data = []
        with open(folder_name+"/xpredictions-"+str(iteration-1)+".json") as j:
            L = json.load(j)
        preds = L["data"]
        with open(folder_name+"/classifier_report-"+str(iteration)+".json") as j:
            M = json.load(j)
        wrongs = M["Wrong Items"]
        for pred in preds:
            dial_idx = pred["dialogue_idx"]
            turn_idx = pred["turn_idx"]
            id = domain+"_"+f"{dial_idx:04}"+"_"+f"{turn_idx:03}"
            for wrong in wrongs:
                if wrong["id"] == id and wrong["classified_prob"]>M["Avg Probability of Classified Group among Wrong Samples"] and wrong["sudo_label_prob"]< M["Avg Probability of Sudo-Label among Wrong Samples"]:
                    pred["predicted_label"] = wrong["classified"]
            data.append(pred)


        with open(folder_name+"/xpredictions-"+str(iteration)+".json","w") as j:
            json.dump({"version": "kmeans_model1+model2_"+str(iteration),"data":data},j)

if __name__ == "__main__":
    main()
