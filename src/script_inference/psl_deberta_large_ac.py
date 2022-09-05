DEBUG = False

if DEBUG:
    print("Debug")
    import pandas as pd
    import random
    
    data_folder = "../input/feedback-prize-effectiveness/train"
    df = pd.read_csv("../input/feedback-prize-effectiveness/train.csv")
    idx = random.choices(df.essay_id.unique().tolist(),k=3000)
    df = df[df.essay_id.isin(idx)].reset_index(drop=True)
    df.to_csv('df_sample.csv',index=False)
    csv_file = "./df_sample.csv"
    test_ = "train"
    del df
    del idx
    
else:
    data_folder = "../input/feedback-prize-effectiveness/test/"
    csv_file = "../input/feedback-prize-effectiveness/test.csv"
    test_ = "test"


import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '../input/packages/transformers-4.20.1-py3-none-any.whl'])

import transformers
transformers.__version__

import sys, os,yaml
sys.path.insert(0, "../input/fprize2/src")

import torch
from utils.utils import make_sub,Path,pd
from train_utils import prediction_step

from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, AutoModel, AutoConfig

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

test_df = pd.read_csv(csv_file)
test_df.shape



class args:
    seed = 2022
    
    # Model
    model_name = "microsoft/deberta-large" #"funnel-transformer/large" #"allenai/longformer-large-4096"

    data_folder_test = Path(data_folder)
    name = "deberta_large"
    add_special_tokens = True
    input_type = "cls_end"
    sample = False

    # Names
    checkpoints_name = 'log'   
    
    model = {
        
            "pretrained_tokenizer":Path(fr'../input/checkpoints/deberta_large_tokenizer/deberta_large_tokenizer/tokenizer/tokenizer'),
            "pretrained_config":Path(fr'../input/checkpoints/deberta_large_tokenizer/deberta_large_tokenizer/config.pth'),
            "max_len":2048,
            'loss':"nn.CrossEntropyLoss",
            "pretrained_weights":None,
            "num_labels":3,
            "model_name":model_name
            }
    

    
    val_loader = {
            "batch_size":1,
            'drop_last':False,
            "num_workers":1,
            "pin_memory":False,
            "shuffle":False
            }
    
    callbacks = {"softmax_before":0
                }
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")


checkpoints = [x.as_posix() for x in (Path(fr'../input/ac-pl-swa-deberta-large')).glob("*.pth") if 'config' not in str(x)]

sub = prediction_step(args,test_df,checkpoints,[1/len(checkpoints)]*len(checkpoints))

sub['nb_ids'] = sub.groupby("discourse_ids")["take_it"].transform(len)
sub['sum_take_it'] = sub.groupby("discourse_ids")["take_it"].transform(sum)
sub.loc[sub['sum_take_it']==0,"take_it"] = 1

sub = sub[sub.take_it!=0]
sub.shape

sub = sub.groupby("discourse_id")[['Ineffective','Adequate','Effective']].mean().reset_index()

sub = sub.reset_index(drop=True)
sub[["discourse_id",'Ineffective','Adequate','Effective']].to_csv('psl_deberta_large_ac.csv',index=False)