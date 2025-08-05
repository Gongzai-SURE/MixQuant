import numpy as np
import random
from typing import List, Union
import os
import torch
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, LlamaTokenizer

def get_wikitext2(nsamples, seed, seqlen, tokenizer, train, local_dir):
    if train:
        if local_dir is not None:
            traindata = load_from_disk(local_dir)["train"]
        else:
            traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
            
    else:
        if local_dir is not None:
            testdata = load_from_disk(local_dir)["test"]
        else:
            testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors='pt')

        return testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, train, local_dir):
    if train:
        if local_dir is not None:
            traindata = load_from_disk(local_dir)["train"]
        else:
            traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        if local_dir is not None:
            testdata = load_from_disk(local_dir)["test"]
        else:
            testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        
        return testenc

def get_c4(nsamples, seed, seqlen, tokenizer, train, local_dir):
    if train:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        if local_dir is not None:
            valdata = load_from_disk(local_dir)
        else:
            valdata = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc

def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "wikitext2",
    tokenizer=None,
    n_samples=32,
    block_size=512,
    split="train",
    text_column="text",
    local_dir=None,
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        elif data == "wikitext2" and local_dir is None:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        elif data == "wikitext2" and local_dir is not None:
            local_dir = local_dir + f"{data}"
            dataset = load_from_disk(local_dir)[split]
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', train=True, local_dir = None
):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    if isinstance(tokenizer, LlamaTokenizer) and 'ptb' in name:
        tokenizer.tokens_trie.data = {}

    if local_dir is not None and os.path.exists(local_dir):
        local_dir = local_dir + f"{name}"
    else:
        local_dir = None
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, train,local_dir)
    elif 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer, train,local_dir)
    elif 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, train,local_dir)
    else: # custom dataset
        print(f"Custom dataset load from {name}")
        datas = torch.load(name)
        ids_shuffle = list(range(len(datas)))
        random.shuffle(ids_shuffle)
        return [tuple(datas[idx].unsqueeze(0)) for idx in ids_shuffle[:nsamples]]