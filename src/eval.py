import sys
import os

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import argparse
import torch
import re
import json
import os
import warnings
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
# from utils.data_utils import BatchSampler
# from utils.metrics import MultilabelF1Max
# from models.adapter_mdoel import AdapterModel
from data.batch_sampler import BatchSampler
from training.metrics import MultilabelF1Max
from models.adapter_model import AdapterModel
from models.lora_model import LoraModel
from peft import PeftModel
from typing import Dict, Any, Union, Tuple
from data.dataloader import prepare_dataloaders
from datetime import datetime

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def evaluate(model, plm_model, metrics, dataloader, loss_function, device=None):
    total_loss = 0
    total_samples = len(dataloader.dataset)
    print(f"Total samples: {total_samples}")
    epoch_iterator = tqdm(dataloader)
    pred_labels = []
    
    for i, batch in enumerate(epoch_iterator, 1):
            
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        
        logits = model(plm_model, batch)
        pred_labels.extend(logits.argmax(dim=1).cpu().numpy())
        
        for metric_name, metric in metrics_dict.items():
            if args.problem_type == 'regression' and args.num_labels == 1:
                loss = loss_function(logits.squeeze(), label.squeeze())
                metric(logits.squeeze(), label.squeeze())
            elif args.problem_type == 'multi_label_classification':
                loss = loss_function(logits, label.float())
                metric(logits, label)
            else:
                loss = loss_function(logits, label)
                metric(torch.argmax(logits, 1), label)
                
        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    epoch_loss = total_loss / len(dataloader.dataset)
    for k, v in metrics.items():
        metrics[k] = [v.compute().item()]
        print(f"{k}: {metrics[k][0]}")
    metrics['loss'] = [epoch_loss]
    return metrics, pred_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--eval_method', type=str, default=None, help='evaluation method')
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels')
    parser.add_argument('--pooling_method', type=str, default='mean', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--split', type=str, default=None, help='split name in Huggingface')
    parser.add_argument('--test_result_dir', type=str, default=None, help='test result directory')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for fixed batch size')
    parser.add_argument('--batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--use_foldseek', action='store_true', help='use foldseek')
    parser.add_argument('--use_ss8', action='store_true', help='use ss8')
    
    # model path
    parser.add_argument('--output_model_name', type=str, default=None, help='model name')
    parser.add_argument('--output_root', default="result", help='root directory to save trained models')
    parser.add_argument('--output_dir', default=None, help='directory to save trained models')
    parser.add_argument('--model_path', default=None, help='model path directly')
    parser.add_argument('--structure_seq', type=str, default="", help='structure sequence')
    parser.add_argument('--training_method', type=str, default="freeze", help='training method')
    args = parser.parse_args()
    
    if 'foldseek_seq' in args.structure_seq:
        args.use_foldseek = True
        print("Enabled foldseek_seq based on structure_seq parameter")
    if 'ss8_seq' in args.structure_seq:
        args.use_ss8 = True
        print("Enabled ss8_seq based on structure_seq parameter")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.test_result_dir, exist_ok=True)
    # build tokenizer and protein language model
    if "esm" in args.plm_model:
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model)
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model)
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model)
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model)
        args.hidden_size = plm_model.config.d_model
    elif "ProSST" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model)
        args.hidden_size = plm_model.config.hidden_size
    elif "Prime" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model)
        args.hidden_size = plm_model.config.hidden_size
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
        plm_model = AutoModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size

    args.vocab_size = plm_model.config.vocab_size
    
    # Define metric configurations
    metric_configs = {
        'accuracy': {
            'binary': BinaryAccuracy,
            'multi': lambda: Accuracy(task="multiclass", num_classes=args.num_labels)
        },
        'recall': {
            'binary': BinaryRecall,
            'multi': lambda: Recall(task="multiclass", num_classes=args.num_labels)
        },
        'precision': {
            'binary': BinaryPrecision,
            'multi': lambda: Precision(task="multiclass", num_classes=args.num_labels)
        },
        'f1': {
            'binary': BinaryF1Score,
            'multi': lambda: F1Score(task="multiclass", num_classes=args.num_labels)
        },
        'mcc': {
            'binary': BinaryMatthewsCorrCoef,
            'multi': lambda: MatthewsCorrCoef(task="multiclass", num_classes=args.num_labels)
        },
        'auroc': {
            'binary': BinaryAUROC,
            'multi': lambda: AUROC(task="multiclass", num_classes=args.num_labels)
        },
        'f1_max': {
            'any': lambda: MultilabelF1Max(num_labels=args.num_labels)
        },
        'spearman_corr': {
            'any': SpearmanCorrCoef
        }
    }

    # Initialize metrics dictionary
    metrics_dict = {}
    args.metrics = args.metrics.split(',')

    # Create metrics based on configurations
    for metric_name in args.metrics:
        if metric_name not in metric_configs:
            raise ValueError(f"Invalid metric: {metric_name}")
            
        config = metric_configs[metric_name]
        if 'any' in config:
            metrics_dict[metric_name] = config['any']()
        else:
            metrics_dict[metric_name] = (config['binary']() if args.num_labels == 2 
                                       else config['multi']())
        
        # Move metric to device
        metrics_dict[metric_name].to(device)

    
    # load adapter model
    print("---------- Load Model ----------")
    # model = AdapterModel(args)
    # if args.model_path is not None:
    #     model_path = args.model_path
    # else:
    #     model_path = f"{args.output_root}/{args.output_dir}/{args.output_model_name}"
    if args.eval_method in ["full", "ses-adapter", "freeze"]:
        model = AdapterModel(args)
    
    elif args.eval_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
        model = LoraModel(args)

    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = f"{args.output_root}/{args.output_dir}/{args.output_model_name}"
    if args.eval_method == "full":
        model_weights = torch.load(model_path)
        model.load_state_dict(model_weights['model_state_dict'])
        plm_model.load_state_dict(model_weights['plm_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    
    if args.eval_method == 'plm-lora':
        lora_path = model_path.replace(".pt", "_lora")
        plm_model = PeftModel.from_pretrained(plm_model,lora_path)
        plm_model = plm_model.merge_and_unload()
    elif args.eval_method == 'plm-qlora':
        lora_path = model_path.replace(".pt", "_qlora")
        plm_model = PeftModel.from_pretrained(plm_model,lora_path)
        plm_model = plm_model.merge_and_unload()
    elif args.eval_method == "plm-dora":
        dora_path = model_path.replace(".pt", "_dora")
        plm_model = PeftModel.from_pretrained(plm_model, dora_path)
        plm_model = plm_model.merge_and_unload()
    elif args.eval_method == "plm-adalora":
        adalora_path = model_path.replace(".pt", "_adalora")
        plm_model = PeftModel.from_pretrained(plm_model, adalora_path)
        plm_model = plm_model.merge_and_unload()
    elif args.eval_method == "plm-ia3":
        ia3_path = model_path.replace(".pt", "_ia3")
        plm_model = PeftModel.from_pretrained(plm_model, ia3_path)
        plm_model = plm_model.merge_and_unload()
    plm_model.to(device).eval()  

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
    def collate_fn(examples):
        aa_seqs, labels = [], []
        if args.use_foldseek:
            foldseek_seqs = []
        if args.use_ss8:
            ss8_seqs = []
        prosst_stru_tokens = [] if "ProSST" in args.plm_model else None
        
        for e in examples:
            aa_seq = e["aa_seq"]
            if args.use_foldseek:
                foldseek_seq = e["foldseek_seq"]
            if args.use_ss8:
                ss8_seq = e["ss8_seq"]
            

            if "ProSST" in args.plm_model and "prosst_stru_token" in e:
                stru_token = e["prosst_stru_token"]
                if isinstance(stru_token, str):
                    seq_clean = stru_token.strip("[]").replace(" ","")
                    tokens = list(map(int, seq_clean.split(','))) if seq_clean else []
                elif isinstance(stru_token, (list, tuple)):
                    tokens = [int(x) for x in stru_token]
                else:
                    tokens = []
                prosst_stru_tokens.append(torch.tensor(tokens))
            
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                if args.use_foldseek:
                    foldseek_seq = " ".join(list(foldseek_seq))
                if args.use_ss8:
                    ss8_seq = " ".join(list(ss8_seq))
            elif 'ankh' in args.plm_model:
                aa_seq = list(aa_seq)
                if args.use_foldseek:
                    foldseek_seq = list(foldseek_seq)
                if args.use_ss8:
                    ss8_seq = list(ss8_seq)
            
            aa_seqs.append(aa_seq)
            if args.use_foldseek:
                foldseek_seqs.append(foldseek_seq)
            if args.use_ss8:
                ss8_seqs.append(ss8_seq)
            labels.append(e["label"])
        
        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            if args.use_foldseek:
                foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
            if args.use_ss8:
                ss8_input_ids = tokenizer.batch_encode_plus(ss8_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
            if args.use_foldseek:
                foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            if args.use_ss8:
                ss8_input_ids = tokenizer(ss8_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
        
        data_dict = {
            "aa_seq_input_ids": aa_input_ids,
            "aa_seq_attention_mask": attention_mask,
            "label": labels
        }
        
        if "ProSST" in args.plm_model and prosst_stru_tokens:
            aa_max_length = len(aa_input_ids[0])
            padded_tokens = []
            for tokens in prosst_stru_tokens:
                if tokens is None or len(tokens) == 0:

                    padded_tokens.append([0] * aa_max_length)
                else:
                    struct_sequence = tokens.tolist()

                    padded_tokens.append(struct_sequence + [0] * (aa_max_length - len(struct_sequence)))
            
            data_dict["aa_seq_stru_tokens"] = torch.tensor(padded_tokens, dtype=torch.long)
        
        if args.use_foldseek:
            data_dict["foldseek_seq_input_ids"] = foldseek_input_ids
        if args.use_ss8:
            data_dict["ss8_seq_input_ids"] = ss8_input_ids
        
        return data_dict
        
    loss_function = nn.CrossEntropyLoss()
    
    def process_data_line(data):
        if args.problem_type == 'multi_label_classification':
            label_list = data['label'].split(',')
            data['label'] = [int(l) for l in label_list]
            binary_list = [0] * args.num_labels
            for index in data['label']:
                binary_list[index] = 1
            data['label'] = binary_list
        if args.max_seq_len is not None:
            data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
            if args.use_foldseek:
                data["foldseek_seq"] = data["foldseek_seq"][:args.max_seq_len]
            if args.use_ss8:
                data["ss8_seq"] = data["ss8_seq"][:args.max_seq_len]
            # 如果是 ProSST 模型且有结构标记，也需要截断
            if "ProSST" in args.plm_model and "prosst_stru_token" in data:
                # 结构标记可能是字符串或列表形式
                if isinstance(data["prosst_stru_token"], str):

                    pass
                elif isinstance(data["prosst_stru_token"], (list, tuple)):
                    data["prosst_stru_token"] = data["prosst_stru_token"][:args.max_seq_len]
            token_num = min(len(data["aa_seq"]), args.max_seq_len)
        else:
            token_num = len(data["aa_seq"])
        return data, token_num
    
    # process dataset from json file
    def process_dataset_from_json(file):
        dataset, token_nums = [], []
        for l in open(file):
            data = json.loads(l)
            data, token_num = process_data_line(data)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums


    # process dataset from list
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for l in data_list:
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums
    
    
    if args.test_file.endswith('json'):
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
    elif args.test_file.endswith('csv'):
        test_dataset, test_token_num = process_dataset_from_list(load_dataset("csv", data_files=args.test_file)['train'])
        if args.test_result_dir:
            test_result_df = pd.read_csv(args.test_file)
    elif '/' in args.test_file:  # Huggingface dataset (only csv now)
        raw_dataset = load_dataset(args.test_file)
        # Using the chosen split first.
        if args.split and args.split in raw_dataset:
            split = args.split
        elif 'test' in raw_dataset:
            split = 'test'
        elif 'validation' in raw_dataset:
            split = 'validation'
        elif 'train' in raw_dataset:
            split = 'train'
        else:
            split = list(raw_dataset.keys())[0]
        
        test_dataset, test_token_num = process_dataset_from_list(raw_dataset[split])
        if args.test_result_dir:
            test_result_df = pd.DataFrame(raw_dataset[split])
    else:
        raise ValueError("Invalid file format")
    
    
    if args.batch_size is None:
        if args.batch_token is None:
            raise ValueError("batch_size or batch_token must be specified")
        test_loader = DataLoader(
            test_dataset, 
            num_workers=args.num_workers, 
            collate_fn=collate_fn,
            batch_sampler=BatchSampler(test_token_num, args.batch_token, False)
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            shuffle=False
        )

    print("---------- Start Eval ----------")
    with torch.no_grad():
        metric, pred_labels = evaluate(model, plm_model, metrics_dict, test_loader, loss_function, device)
        if args.test_result_dir:
            pd.DataFrame(metric).to_csv(f"{args.test_result_dir}/evaluation_metrics.csv", index=False)
            test_result_df["pred_label"] = pred_labels
            test_result_df.to_csv(f"{args.test_result_dir}/evaluation_result.csv", index=False)
