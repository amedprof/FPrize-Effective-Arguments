from multiprocessing import reduction
import re
import math
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader
from data.data_utils import batch_to_device
from data.dataset import FeedbackDataset,CustomCollator
from transformers import AutoTokenizer,AutoConfig, AutoModel

from model_zoo.models import FeedbackModel
from sklearn.metrics import log_loss
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup

from utils.utils import *
from utils.logger import create_logger
from tqdm.notebook import tqdm
import gc
import torch.utils.checkpoint
import bitsandbytes as bnb



# ------------------------------------------ Loss function ------------------------------------------- #



import torch 

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=0.0001,
        adv_eps=0.001,
        start_epoch=1,
        adv_step=1,
        scaler=None,
        args = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.args = args
        self.criterion = eval(args.model['loss'])(reduction="mean").to(args.device)
    def attack_backward(self,data,epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            adv_loss,_ = training_step(self.args,self.model,self.criterion,data)            
            self.optimizer.zero_grad()
            if self.scaler is None:
                adv_loss.backward()
            else:
                self.scaler.scale(adv_loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


# ------------------------------------------ Loss function ------------------------------------------- #







def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_checkpoint(net,checkpoint):
    print(checkpoint)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint, map_location=dummy_device)
    try:
        net.load_state_dict(d)
    except:
        net.load_state_dict(d['model'])
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

    
    span_preds = from_token_to_span(pred,data['label_ids'])
    mask = data["span_labels"]!=-100
    span_labels = data["one_hot_label"][mask]
    # nb = span_preds.shape[0]
    loss = criterion(span_preds,span_labels)

    # Metrics 

    return loss,{"train_loss":loss.item()}



# # ----------------- Evaluation --------------------- #
def decoding_token_proba(preds,labels_ids,softmax_before=2):
    predictions = []
    ids = [x for x in torch.unique(labels_ids) if x!=-1]

    for idx in ids:
        if idx!=-1:
            mask = labels_ids==idx
            nb = (1*mask).sum()
            if softmax_before==1:
                p = ((preds[mask].softmax(-1)).sum(0).reshape(1,-1))/nb
                p_sum = p.sum()
                p = p/p_sum
                predictions.append(p)
            elif softmax_before==0:
                p = (preds[mask].sum(0).reshape(1,-1))/nb
                predictions.append(p.softmax(-1))
            else:
                p = (preds[mask].sum(0).reshape(1,-1))/nb
                predictions.append(p)       
    predictions = torch.cat(predictions)
    return predictions

# ------------------------------------------ ------------------------------------------- #
def batch_decoding_token_proba(preds,b_labels_ids,softmax_before=2):
    predictions = []
    for b,labels_ids in zip(preds,b_labels_ids):
        predictions.append(decoding_token_proba(b,labels_ids,softmax_before))
    return torch.cat(predictions)#.clip(max=1-10e-15,min=10e-15)

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
            span_preds = from_token_to_span(pred,data['label_ids'])
            mask = data["span_labels"]!=-100
            # mask = data["take_it"]==1
            # mask = mask1*mask2
            span_labels = data["one_hot_label"][mask]
            # span_preds
            # nb = span_preds.shape[0]
            ytrue.append(span_labels)
            loss.append(criterion(span_preds,span_labels))

            # Metrics 
            preds = batch_decoding_token_proba(pred,data["label_ids"],softmax_before=args.callbacks['softmax_before'])
            ypred.append(preds)

    ytrue = torch.cat(ytrue,dim=0).detach().cpu()
    ypred = torch.cat(ypred,dim=0).detach().cpu()
    loss = torch.cat(loss,dim=0).mean()
    m = log_loss(ytrue,ypred)    

    return {"val_loss":loss,"val_log_loss":m}



# ------------------------------------------ ------------------------------------------- #
def prediction_step(args,df_test,checkpoints,weights):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model['pretrained_tokenizer'])

    data_collator = CustomCollator(tokenizer,inference=False)
    test_dataset = FeedbackDataset(df_test,tokenizer,args.model["max_len"],data_folder=args.data_folder_test,
                                    add_special_tokens=args.add_special_tokens,
                                    input_type=args.input_type,
                                    mask_pct=0
                                    )


    test_loader = DataLoader(test_dataset,**args.val_loader,collate_fn=data_collator)

    def load_checkpoint(args,checkpoint):

        print(checkpoint)
        net = FeedbackModel(args.model['model_name'],args.model['num_labels'],pretrained_path=None,
                            test=True,config_path=args.model['pretrained_config'])
        if args.input_type!="":
            print(f'Using {args.input_type}')
            net.backbone.resize_token_embeddings(len(tokenizer))

        dummy_device = torch.device("cpu")
        d = torch.load(checkpoint, map_location=dummy_device)
        try:
            net.load_state_dict(d)
        except:
            net.load_state_dict(d['model'])
        
        del d
        gc.collect()
        return net

    

    # models = [load_checkpoint(args,cp).to(args.device).eval() for cp in checkpoints]
    # print(f"using {len(models)} checkpoints")
    ypred = 0
    ids = []
    take_it = []

    for j,(cp,weight) in enumerate(zip(checkpoints,weights)):
        net = load_checkpoint(args,cp).to(args.device)
        net.eval()

        ypred_model = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                data = batch_to_device(data, args.device)
                
                # for net,weight in zip(models,weights):
                pred = net(data)
                # Metrics 
                preds= batch_decoding_token_proba(pred,data["label_ids"],softmax_before=args.callbacks['softmax_before'])*weight
                
                ypred_model.append(preds)

                if j==0:
                    mask = data["discourse_ids"]!=-1
                    idx = data["discourse_ids"][mask]
                    ids.append(idx)

                    mask = data["take_it"]!=-1
                    idx = data["take_it"][mask]
                    take_it.append(idx)
        
        if args.callbacks['softmax_before']==2:
            ypred+= torch.cat(ypred_model,dim=0).softmax(-1).detach().cpu().numpy()
        else:
            ypred+= torch.cat(ypred_model,dim=0).detach().cpu().numpy() 

        del net
        del ypred_model
        del data
        del preds
        del pred

        torch.cuda.empty_cache()
        gc.collect()

    ids = torch.cat(ids,dim=0).detach().cpu().numpy()
    take_it = torch.cat(take_it,dim=0).detach().cpu().numpy()



    sub = pd.DataFrame({
                        "discourse_ids":ids,
                        "take_it":take_it,
                        "Ineffective":ypred[:,0],
                        "Adequate":ypred[:,1],
                        "Effective":ypred[:,2],
                        })
    TARGET = ['Ineffective','Adequate','Effective']
    df = test_dataset.df.drop(TARGET,axis=1)
    sub = df.merge(sub,how='left',on="discourse_ids")

    # del models
    del test_dataset
    del test_loader
    del take_it
    del ids
    del data_collator
    del tokenizer
    del load_checkpoint

    gc.collect()
    torch.cuda.empty_cache()
    return sub
# #----------------------------------- Training Steps -------------------------------------------------#

def fit_net(
                model,
                train_dataset,
                val_dataset,
                args,
                fold,
                tokenizer
    ):
   
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

    awp = AWP(model,
          optimizer,
        #   adv_lr=args.adv_lr,
        #   adv_eps=args.adv_eps,
        #   start_epoch=args.num_train_steps/args.epochs,
          scaler=scaler,
          args=args
             )

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
                if args.use_awp:
                    awp.attack_backward(data,epoch) 

                 # gradient clipping
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                                                        parameters=model.parameters(), max_norm=10
                                                    )

                scaler.step(optimizer)
                scaler.update()
                

            else:
                loss.backward()
                if args.use_awp:
                    awp.attack_backward(data,epoch)

                # gradient clipping
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                                                    parameters=model.parameters(), max_norm=10
                                                )

                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            # Evaluation
            if (((step + 1) % step_val == 0) or (step + 1 == nb_step_per_epoch)) and (epoch>=args.callbacks["start_eval_epoch"]):
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

            # if args.use_gradient_checkpointing:
            #     model.backbone.gradient_checkpointing_enable()
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

    train_dataset = FeedbackDataset(train_df.copy(),tokenizer,args.model["max_len"],
                                    data_folder=args.data_folder/'train',
                                    add_special_tokens=args.add_special_tokens,
                                    input_type=args.input_type,mask_pct=args.mask_pct
                                    )

    val_dataset = FeedbackDataset(valid_df.copy(),tokenizer,args.model["max_len_eval"],
                                    data_folder=args.data_folder/'train',
                                    add_special_tokens=args.add_special_tokens,
                                    input_type=args.input_type,mask_pct=0
                                    )
    model = FeedbackModel(args.model['model_name'],
                          args.model['num_labels'],
                          pretrained_path=pretrained_path,
                          config_path = args.model['pretrained_config'],
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
        fold,
        tokenizer
    )

    return pred_val

def _from_token_to_span(preds,labels_ids):
    predictions = []
    ids = torch.unique(labels_ids)
    for idx in ids:
        if idx!=-1:
            mask = labels_ids==idx
            nb = (1*mask).sum()
            p = (preds[mask].sum(0).reshape(1,-1))/nb
            predictions.append(p)
    return torch.cat(predictions)

def from_token_to_span(preds,labels_ids):
    predictions = []
    for p,l in zip(preds,labels_ids):
        predictions.append(_from_token_to_span(p,l))
    return torch.cat(predictions)


def kfold(args,dfx):
    # create_logger(str(args.checkpoints_path/(args.checkpoints_name+'.txt')))
    k = len(dfx[args.kfold_name].unique())
    TARGET = ['Ineffective','Adequate','Effective']

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
        tokenizer.add_special_tokens({"additional_special_tokens": list(cls_tokens_map.values())+list(end_tokens_map.values())})
    else :
        pass

    if args.model['pretrained_tokenizer']:
        pass
    else:
        tokenizer.save_pretrained(args.checkpoints_path/'tokenizer/')
        config = AutoConfig.from_pretrained(args.model['model_name'])
        torch.save(config, args.checkpoints_path/'config.pth')
    
    print(len(tokenizer))
    print(f"----------- {args.kfold_name} ---------")
    for i in args.selected_folds:
        
        if i in args.selected_folds:
            print(f"\n-------------   Fold {i+1} / {k}  -------------\n")
            
            df = dfx.copy()

            for c in TARGET:
                cols = [x for x in df.columns if x!=c  if c in x if f'fold_{i}' in x if any([z+'_fold' in x for z in args.prefix_names_psl])]
                # cols = [x for x in df.columns if x!=c  if c in x if f'fold_{i}' in x]
                print("------- getting soft labels from -------")
                print(cols)
                df[c] = df[cols].mean(axis=1)

            if args.add_soft_labels_fold:
                print("adding soft labels")
                df_fold = df[df[args.kfold_name]==i]
                df_fold[args.kfold_name] = 10
                df = pd.concat([df,df_fold],axis=0).reset_index(drop=True)

            
            df.loc[df[args.kfold_name]!=10,TARGET] = pd.get_dummies(df.loc[df[args.kfold_name]!=10].target).values
            print(df.loc[df[args.kfold_name]!=10,TARGET].sum())

            if args.sample:
                train_df = df[df[args.kfold_name]!=i].reset_index(drop=True).sample(100)
                valid_df = df[df[args.kfold_name]==i].reset_index(drop=True).sample(100)

            else:
                if args.use_all_data:
                    train_df = df.copy()
                else:
                    train_df = df[df[args.kfold_name]!=i].reset_index(drop=True)#.sample(100)
                
                valid_df = df[df[args.kfold_name]==i].reset_index(drop=True)#.sample(100)

            _ = train_one_fold(args,tokenizer,train_df,valid_df,i)