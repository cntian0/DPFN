# @Author  : tiancn
# @Time    : 2022/9/18 11:57
from DPFN import SimpleBertModel
import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AdamW,get_linear_schedule_with_warmup
from data_utils import creatr_data_loader
import pandas as pd
import json
import pickle
from datetime import datetime
import math

class Instructor:
    def __init__(self,opt):
        self.opt = opt
        train_data_name = 'middleFile/twitter15_train_datas.pkl'
        val_data_name = 'middleFile/twitter15_val_datas.pkl'
        test_data_name = 'middleFile/twitter15_test_datas.pkl'
        # train_data_name = 'middleFile/twitter17_train_datas.pkl'
        # val_data_name = 'middleFile/twitter17_val_datas.pkl'
        # test_data_name = 'middleFile/twitter17_test_datas.pkl'
        if os.path.exists(train_data_name):
            self.train_data_loader = pickle.load(open(train_data_name, 'rb'))
        else:
            train_data_loader = creatr_data_loader(opt.dataset_file['train'], 'train', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(train_data_name, 'wb') as f:
                pickle.dump(train_data_loader, f)
            self.train_data_loader = train_data_loader
        if os.path.exists(val_data_name):
            self.val_data_loader = pickle.load(open(val_data_name, 'rb'))
        else:
            val_data_loader = creatr_data_loader(opt.dataset_file['dev'], 'dev', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(val_data_name, 'wb') as f:
                pickle.dump(val_data_loader, f)
            self.val_data_loader = val_data_loader
        if os.path.exists(test_data_name):
            self.test_data_loader = pickle.load(open(test_data_name, 'rb'))
        else:
            test_data_loader = creatr_data_loader(opt.dataset_file['test'], 'test', opt.MAX_LEN, opt.BATCH_SIZE)
            with open(test_data_name, 'wb') as f:
                pickle.dump(test_data_loader, f)
            self.test_data_loader = test_data_loader

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def train_epoch(self,loss_fn, optimizer, scheduler):
        self.model.train()
        losses = []
        correct_predictions = 0
        n_total = 0
        for i_batch,sample_batched in enumerate(self.train_data_loader):
            #print(i_batch)
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            outputs = self.model(inputs)
            targets = sample_batched['targets'].to(self.opt.device)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())
            n_total += len(outputs)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #print('train'+str(n_total))
        return correct_predictions / n_total, np.mean(losses)

    def eval_model(self,loss_fn,type):
        if type == 'dev':
            dataloader = self.val_data_loader
        else:
            dataloader = self.test_data_loader
        losses = []
        correct_predictions = 0
        n_total = 0
        rows = []
        self.model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['targets'].to(self.opt.device)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets).item()
                n_total+=len(outputs)
                losses.append(loss.item())

                rows.extend(
                    zip(
                        sample_batched["review_text"],
                        sample_batched["sentiment_targets"],
                        sample_batched["targets"].numpy(),
                        preds.cpu().numpy(),
                    )
                )
            if type == 'test':
                return (
                    correct_predictions / n_total,
                    np.mean(losses),
                    self.format_eval_output(rows)
                )

        return correct_predictions / n_total, np.mean(losses)

    def format_eval_output(self,rows):
        tweets, targets, labels, predictions = zip(*rows)
        tweets = np.vstack(tweets)
        targets = np.vstack(targets)
        labels = np.vstack(labels)
        predictions = np.vstack(predictions)
        results_df = pd.DataFrame()
        results_df["tweet"] = tweets.reshape(-1).tolist()
        results_df["target"] = targets.reshape(-1).tolist()
        results_df["label"] = labels
        results_df["prediction"] = predictions
        return results_df


    def run(self):
        results_per_run = {}
        accMax = [0,0]
        f1Max = [0,0]
        acc_max_seed = 0
        f1_max_seed = 0
        for run_number in range(self.opt.NUM_RUNS):
            #np.random.seed(self.opt.RANDOM_SEEDS[run_number] + 1)
            #torch.manual_seed(self.opt.RANDOM_SEEDS[run_number] + 1)
            np.random.seed(3)
            torch.manual_seed(3)
            # np.random.seed(1)
            # torch.manual_seed(1)
            #seed_num = 3
            self.model = self.opt.model_class(self.opt).to(self.opt.device)

            #Configure the optimizer and scheduler.
            optimizer = AdamW(self.model.parameters(), lr=self.opt.LEARNING_RATE, correct_bias=self.opt.ADAMW_CORRECT_BIAS)
            total_steps = len(self.train_data_loader) * self.opt.EPOCHS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=self.opt.NUM_WARMUP_STEPS, num_training_steps=total_steps
            )
            #loss
            loss_fn = nn.CrossEntropyLoss().to(self.opt.device)
            for epoch in range(self.opt.EPOCHS):
                print(f"Epoch {epoch + 1}/{self.opt.EPOCHS} -- RUN {run_number}")
                print("-" * 30)
                train_acc, train_loss = self.train_epoch(loss_fn, optimizer, scheduler)
                print(f"Train loss {train_loss} accuracy {train_acc}")

                val_acc, val_loss = self.eval_model(loss_fn,"dev")
                print(f"Val   loss {val_loss} accuracy {val_acc}")

                test_acc, _, detailed_results = self.eval_model(loss_fn, 'test')
                macro_f1 = f1_score(
                    detailed_results.label, detailed_results.prediction, average="macro"
                )
                if test_acc > accMax[0]:
                    accMax[0] = test_acc
                    accMax[1] = macro_f1
                    acc_max_seed = run_number
                if macro_f1 > f1Max[1]:
                    f1Max[0] = test_acc
                    f1Max[1] = macro_f1
                    f1_max_seed = run_number

                print(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")

            test_acc, _, detailed_results = self.eval_model(loss_fn,'test')
            macro_f1 = f1_score(
                detailed_results.label, detailed_results.prediction, average="macro"
            )
            print(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")

            results_per_run[run_number] = {
                "accuracy": test_acc,
                "macro-f1": macro_f1
            }
        with open('./result/results_per_run.json', 'w+') as f:
            json.dump(results_per_run, f)

        resMax = {
            "accMax":{
                "acc" : accMax[0],
                "f1" : accMax[1],
                "seed":acc_max_seed
            },
            "f1Max":{
                "acc": f1Max[0],
                "f1": f1Max[1],
                "seed":f1_max_seed
            }
        }

        curr_time = datetime.now()
        time_str = str(datetime.strftime(curr_time, '%Y-%m-%d_%H:%M:%S'))
        with open('./result/' + time_str + '.json', 'w+') as f:
            json.dump(resMax, f)
        print(resMax)


        print(f"AVERAGE ACC = {np.mean([_['accuracy'] for _ in results_per_run.values()])}")
        print(f"AVERAGE MAC-F1= {np.mean([_['macro-f1'] for _ in results_per_run.values()])}")


def main():
    model_classes = {
        "simpleBert":SimpleBertModel
    }
    # dataset filepath
    dataset_files = {
        'twitter15':{
            'train':'',
            'dev':'',
            'test':''
        },
        'twitter17': {
            'train': '',
            'dev': '',
            'test': ''
        }
    }
    input_colses = {
        "simpleBert":["input_ids","attention_mask","vit_feature","transformer_mask","target_input_ids","target_attention_mask","target_mask","text_length","word_length","tran_indices","context_asp_adj_matrix","globel_input_id","globel_mask"]
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='simpleBert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter15', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--MAX_LEN', default=50, type=int)
    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--DROPOUT_PROB', default=0.1, type=float)
    parser.add_argument('--NUM_CLASSES', default=3, type=int)
    parser.add_argument('--DEVICE', default="cuda:2", type=str)
    parser.add_argument('--EPOCHS', default=20, type=int)
    parser.add_argument('--LEARNING_RATE', default=5e-5, type=float)
    parser.add_argument('--ADAMW_CORRECT_BIAS',default=True, action='store_true')
    parser.add_argument('--NUM_WARMUP_STEPS', default=0, type=int)
    parser.add_argument('--NUM_RUNS', default=1,type=int)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.RANDOM_SEEDS = list(range(opt.NUM_RUNS))
    print(opt.RANDOM_SEEDS)

    opt.device = torch.device(opt.DEVICE if torch.cuda.is_available() else 'cpu')

    ins = Instructor(opt)
    ins.run()



if __name__ == '__main__':
    main()