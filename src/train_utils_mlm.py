from multiprocessing import reduction
import re
import math
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader
from data.data_utils import batch_to_device
from data.dataset_mlm import FeedbackDataset,CustomCollator
from transformers import AutoTokenizer,AutoConfig, AutoModel

from model_zoo.models import FeedbackModel
from sklearn.metrics import log_loss
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup

from utils.utils import *
from utils.logger import create_logger
from tqdm.notebook import tqdm
import torch.utils.checkpoint


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0)
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9)
    return sum_embeddings / sum_mask

def max_pooling(model_output, attention_mask):
    return torch.max(model_output[attention_mask], 0)[0]

def fast_mean_pooling(output_view, div):
    return torch.sum(output_view, 0) / div

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_checkpoint(net,checkpoint):
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint, map_location=dummy_device)
    try:
        net.load_state_dict(d)
    except:
        net.load_state_dict(d['model'])
        
    net = net
    return net

# ------------------------------------------ Loss function ------------------------------------------- #


# ------------------------------------------ Optimizer ------------------------------------------- #

class AutoSave:
  def __init__(self, top_k=3,metric_track="mae_val",mode="min", root=None):
    
    self.top_k = top_k
    self.logs = []
    self.metric = metric_track
    self.mode = -1 if mode=='min' else 1
    self.root = Path(root)
    assert self.root.exists()

    self.top_models = []
    self.top_metrics = []
    self.texte_log = []

  def log(self, model, metrics):
    metric = metrics[self.metric]
    rank = self.rank(self.mode*metric)

    self.top_metrics.insert(rank+1, self.mode*metric)
    if len(self.top_metrics) > self.top_k:
      self.top_metrics.pop(0)


    self.logs.append(metrics)
    self.save(model, rank, metrics)


  def save(self, model,rank, metrics):
    val_text = " "
    for k,v in metrics.items():
        if k in ["fold","epoch",'step']:
            val_text+=f"_{k}={v:.0f} "
        else:
            val_text+=f"_{k}={v:.4f} "

    name = val_text.strip()
    name = name+".pth"
    path = self.root.joinpath(name)

    old_model = None
    self.top_models.insert(rank+1, name)
    if len(self.top_models) > self.top_k:
      old_model = self.root.joinpath(self.top_models[0])
      self.top_models.pop(0)      

    torch.save(model.state_dict(), path.as_posix())

    if old_model is not None:
      old_model.unlink()


  def rank(self, val):
    r = -1
    for top_val in self.top_metrics:
      if val <= top_val:
        return r
      r += 1

    return r


# # ----------------- Opt/Sched --------------------- #
def get_optim_sched(model,args):

    optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])

    # if 'scheduler' in args:
    if args.scheduler['name'] == 'poly':

        params = args.scheduler['params']

        power = params['power']
        lr_end = params['lr_end']

        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)

    elif args.scheduler['name'] in ['linear','cosine']:
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
            
    else:
        scheduler = eval(args.scheduler['name'])(optimizer, **args.scheduler['params'])


    return optimizer,scheduler

# # ----------------- One Step --------------------- #
def training_step(args,model,criterion,data):
    
    data = batch_to_device(data, args.device)

    if args.trainer['use_amp']:
        with amp.autocast():
            pred = model(data)
    else:
        pred = model(data)

    mask = data["label"]!=-100
    labels = data["label"][mask]
    loss = criterion(pred[mask],labels)

    # Metrics 
    return loss,{"train_loss":loss.item()}

# ------------------------------------------ ------------------------------------------- #

def evaluate_step(args,model,criterion,val_loader,val_data):

    criterion = eval(args.model['loss'])(reduction="none").to(args.device)
    model.eval()
    ypred = []
    ytrue = []
    loss = []
    with torch.no_grad():
        for data in val_loader:
            data = batch_to_device(data, args.device)
            pred = model(data)
            # Loss 

            mask = data["label"]!=-100
            labels = data["label"][mask]
            # ytrue.append(labels)
            loss.append(criterion(pred[mask],labels))
            # ypred.append(pred[mask])

    # ytrue = torch.cat(ytrue,dim=0)
    # ypred = torch.cat(ypred,dim=0)
    loss = torch.cat(loss,dim=0).mean()
    # m = criterion(ytrue,ypred).detach().cpu()    

    return {"val_loss":loss,"val_log_loss":loss}
# ------------------------------------------ ------------------------------------------- #
# #----------------------------------- Training Steps -------------------------------------------------#

def fit_net(
                model,
                train_dataset,
                val_dataset,
                args,
                fold
    ):
   
    tokenizer = AutoTokenizer.from_pretrained(args.model['model_name'])
    data_collator = CustomCollator(tokenizer)
    criterion_tr = eval(args.model['loss'])().to(args.device)
    train_loader = DataLoader(train_dataset,**args.train_loader,collate_fn=data_collator)
    val_loader = DataLoader(val_dataset,**args.val_loader,collate_fn=data_collator)
    
    args.len_train_loader = len(train_loader)
    args.dataset_size = len(train_dataset)

    mode_ = -1 if args.callbacks["mode"]=='max' else 1
    best_epoch = mode_*np.inf
    best = mode_*np.inf

    es = args.callbacks['es']
    es_step = 0
    patience = args.callbacks['patience']

    if args.callbacks["save"]:
        args.checkpoints_path.mkdir(parents=True,exist_ok=True)

    saver = AutoSave(root=args.checkpoints_path,metric_track=args.callbacks['metric_track'],top_k=args.callbacks['top_k'],mode=args.callbacks['mode'])

  
    if args.trainer['use_amp'] and ("cuda" in str(args.device)):
        scaler = amp.GradScaler()
        print("Using Amp")
    else:
        scaler = None

    optimizer,scheduler = get_optim_sched(model,args)

    for epoch in range(args.trainer['epochs']):

        # Init
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        # Init Metrics
        trn_metric = {}
        for k in ["train_loss"]:
            trn_metric[k]=0
        

        nb_step_per_epoch = args.len_train_loader
        step_val = int(np.round(nb_step_per_epoch*args.callbacks['epoch_pct_eval']))
        nstep_val = int(1/args.callbacks['epoch_pct_eval'])

        pbar = tqdm(train_loader)
        for step,data in enumerate(pbar):
            if step==epoch and step==0:
                print(" ".join(train_dataset.tokenizer.convert_ids_to_tokens(data['input_ids'][0])))

            loss,tr_sc= training_step(args,model,criterion_tr,data)
            pbar.set_postfix(tr_sc)
            for k,v in tr_sc.items():
                trn_metric[k]+= v/nb_step_per_epoch 

            if args.trainer['use_amp']:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            # Evaluation
            if ((step + 1) % step_val == 0) or (step + 1 == nb_step_per_epoch):
                metric_val = evaluate_step(args,model,criterion_tr,val_loader,val_dataset)
                metrics = {
                    "epoch": epoch+1,
                    "step": int(np.ceil((step+1)/step_val)),
                    "fold":fold
                }
                metrics.update(metric_val)
                metrics.update(trn_metric)
                saver.log(model, metrics)
        
                elapsed_time = time.time() - start_time
                elapsed_time = elapsed_time * args.callbacks['verbose_eval']

                lr = scheduler.get_lr()[0]
                
                val_text = " "
                for k,v in metric_val.items():
                    val_text+=f" {k}={v:.4f} "

                trn_text = " "
                for k,v in trn_metric.items():
                    trn_text+=f" {k}={v:.4f} "

                texte = f"Epoch {epoch + 1}.{int(np.ceil((step+1)/step_val))}/{args.trainer['epochs']} lr={lr:.6f} t={elapsed_time:.0f}s "
                texte = texte+trn_text+val_text
                print(texte)
                metric_val = metric_val[args.callbacks['metric_track']]

                if es:
                    if args.callbacks['mode']=='min':
                        if (metric_val<best):
                            best = metric_val
                    else:
                        if (metric_val>best):
                            best = metric_val


        if es:
            if args.callbacks['mode']=='min':
                if (best<best_epoch):
                    best_epoch = best
                    es_step = 0
                else:
                    es_step+=1
                    print(f"es step {es_step}")
            else:
                if (best>best_epoch):
                    best_epoch = best
                    es_step = 0
                else:
                    es_step+=1
                    print(f"es step {es_step}")

            if (es_step>patience):
                break

    torch.cuda.empty_cache()
    return 1

# #----------------------------------- Training Folds -------------------------------------------------#

def train_one_fold(args,tokenizer,train_df,valid_df,fold):    
    
    checkpoints = [x.as_posix() for x in args.pretrained_path.glob("*.pth") if f'fold={fold}' in str(x)]
    checkpointg = [x.as_posix() for x in args.pretrained_path.glob("*.pth") if f'global' in str(x)]
    if len(checkpoints):
        pretrained_path = checkpoints[0]
    elif len(checkpointg):
        pretrained_path = checkpointg[0]
    else:
        checkpoints = [x.as_posix() for x in args.pretrained_path.glob("*.pth") if f'config' not in str(x)]
        pretrained_path = random.choice(checkpoints)
    
    train_dataset = FeedbackDataset(train_df,tokenizer,args.model["max_len"],
                                    data_folder=args.data_folder/'train',
                                    add_special_tokens=args.add_special_tokens,
                                    input_type=args.input_type,mask_pct=args.mask_pct
                                    )

    val_dataset = FeedbackDataset(valid_df,tokenizer,args.model["max_len"],
                                    data_folder=args.data_folder/'train',
                                    add_special_tokens=args.add_special_tokens,
                                    input_type=args.input_type,mask_pct=0
                                    )
    args.model['num_labels'] = len(tokenizer)
    model = FeedbackModel(args.model['model_name'],args.model['num_labels'],pretrained_path=pretrained_path,
                         tokenizer_size=len(tokenizer),
                         use_dropout = args.use_dropout,
                         use_gradient_checkpointing = args.use_gradient_checkpointing 
                         ).to(args.device) 

    if args.input_type!="":
        model.backbone.resize_token_embeddings(len(tokenizer))
        # model.resize_token_embeddings(len(tokenizer)) 
    model.zero_grad()    


    n_parameters = count_parameters(model)
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit_net(
        model,
        train_dataset,
        val_dataset,
        args,
        fold
    )

    return pred_val

def _from_token_to_span(preds,labels_ids):
    predictions = []
    ids = torch.unique(labels_ids)
    for idx in ids:
        if idx!=-1:
            mask = labels_ids==idx
            nb = (1*mask).sum()
            p = (preds[mask].sum(0).reshape(1,-1)+1e-15)/nb
            predictions.append(p)
    return torch.cat(predictions)

def from_token_to_span(preds,labels_ids):
    predictions = []
    for p,l in zip(preds,labels_ids):
        predictions.append(_from_token_to_span(p,l))
    return torch.cat(predictions)


def kfold(args,df):
    # create_logger(str(args.checkpoints_path/(args.checkpoints_name+'.txt')))
    k = len(df[args.kfold_name].unique())


    
    if args.model['pretrained_tokenizer']:
        tokenizer = AutoTokenizer.from_pretrained(args.model['pretrained_tokenizer'])

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model['model_name'])
    
    disc_types = [
                        "Claim",
                        "Concluding Statement",
                        "Counterclaim",
                        "Evidence",
                        "Lead",
                        "Position",
                        "Rebuttal",
                    ]

    if args.input_type=="cls_end" and not args.model['pretrained_tokenizer']:
        cls_tokens_map = {label: f"[cls_{label.lower().split()[0]}]" for label in disc_types}
        end_tokens_map = {label: f"[end_{label.lower().split()[0]}]" for label in disc_types}
        if len(tokenizer.encode("\n\n"))==2:
            print('adding new line token')
            new_line_tok = {"new_line":"\n"}
            tokenizer.add_special_tokens({"additional_special_tokens": list(cls_tokens_map.values())+list(end_tokens_map.values()) + list(new_line_tok.values())})
        else:
            tokenizer.add_special_tokens({"additional_special_tokens": list(cls_tokens_map.values())+list(end_tokens_map.values())})
    
    elif args.input_type=="":
        pass
    
    if args.model['pretrained_tokenizer']:
        pass
    else:
        print('saving tokenizer')
        tokenizer.save_pretrained(args.checkpoints_path/'tokenizer/')
        config = AutoConfig.from_pretrained(args.model['model_name'])
        torch.save(config, args.checkpoints_path/'config.pth')

    # tokenizer.save_pretrained(args.checkpoints_path/'tokenizer/')
    # config = AutoConfig.from_pretrained(args.model['model_name'])
    # torch.save(config, args.checkpoints_path/'config.pth')
    
    print(f"----------- {args.kfold_name} ---------")
    for i in args.selected_folds:
        
        if i in args.selected_folds:
            print(f"\n-------------   Fold {i+1} / {k}  -------------\n")

            if args.sample:
                train_df = df[df[args.kfold_name]!=i].reset_index(drop=True).sample(100)
                valid_df = df[df[args.kfold_name]==i].reset_index(drop=True).sample(100)

            else:
                train_df = df[df[args.kfold_name]!=i].reset_index(drop=True)#.sample(100)
                valid_df = df[df[args.kfold_name]==i].reset_index(drop=True)#.sample(100)

            _ = train_one_fold(args,tokenizer,train_df,valid_df,i)