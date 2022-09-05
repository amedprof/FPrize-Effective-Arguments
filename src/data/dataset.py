import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from data.data_utils import add_text_to_df,clean_text,get_text_start_end,tqdm

## =============================================================================== ##
class FeedbackDataset(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 max_length,
                 data_folder=None,
                 add_special_tokens = True,
                 input_type = "cls_end",
                 mask_pct = 0.15
                ):
        
        self.mask_pct = mask_pct
        self.input_type = input_type
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        disc_types = [
                        "Claim",
                        "Concluding Statement",
                        "Counterclaim",
                        "Evidence",
                        "Lead",
                        "Position",
                        "Rebuttal",
                    ]

        if self.input_type=="cls_end":
            self.cls_tokens_map = {label: f"[cls_{label.lower().split()[0]}]" for label in disc_types}
            self.end_tokens_map = {label: f"[end_{label.lower().split()[0]}]" for label in disc_types}  
            self.cls_id_map = {
                                label: self.tokenizer.encode(tkn)[1]
                                for label, tkn in self.cls_tokens_map.items()
                               }
            self.end_id_map = {
                                label: self.tokenizer.encode(tkn)[1]
                                for label, tkn in self.end_tokens_map.items()
                              }
            self.cls_tokens = list(self.cls_id_map.values()) 
            self.end_tokens = list(self.end_id_map.values())

        elif self.input_type=="":
            pass

        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        self.texts = []
        self.essay_id = []
        self.discourse_id = []
        self.discourse_ids = []
        self.st_ed = []
        self.labels = []
        self.token_start_end = []
        self.take_it = []
        self.one_hot = []
        df = self.prepare_df(df)
        
        for essay_id,g in tqdm(df.sort_values(["essay_len","discourse_start"]).groupby('essay_id',sort=True)):
            starts = [0]
            ends = []
            text_start = [0]
            text_end = []

            text = g.essay_text.values[0]
            st_ed = []
            st_ed_token = []
            for tr,t,sfrom in g[['discourse_text_agg','discourse_text',"previous_discourse_end"]].values.tolist():
                s,e = get_text_start_end(text,t,sfrom)
                text = clean_text(self.find_n_replace(text,s,e,tr))
                s,e = get_text_start_end(text,t,sfrom)
                st,et = get_text_start_end(text,tr,sfrom)
                st_ed.append([s,e])
                st_ed_token.append([st,et])

            encoding = tokenizer(
                                text,
                                truncation=False,
                                return_offsets_mapping=True,
                                add_special_tokens = True
                                )

            input_size = len(encoding['offset_mapping'])
            z = [0]
            
            if input_size<=max_length:
                self.st_ed.append(st_ed)
                self.essay_id.append(essay_id)
                self.texts.append(text)
                self.labels.append(g.target.values.tolist())
                self.discourse_id.append(g['discourse_id'].values.tolist())   
                self.discourse_ids.append(g['discourse_ids'].values.tolist())
                self.take_it.append([1]*(len(g)))
                self.one_hot.append(g[["Ineffective","Adequate","Effective"]].values.tolist())
            else:
                
                while input_size>z[-1]+max_length:
                    x = encoding['offset_mapping'][z[-1]:z[-1]+max_length][-3:]
                    x = [j for j in x if j!=(0,0)]
                    _end = x[-1][0]
                    for pos,(s,e) in enumerate(st_ed_token):
                        if _end >s and _end<=e:
                            break

                    pos2 = min(pos-2,len(st_ed_token)-3) if len(st_ed_token)>3 else pos
                    eed = 0
                    while (input_size-eed)>max_length:
                        eed = [i for i,j in enumerate(encoding['offset_mapping']) if st_ed_token[pos2:][0][0] >=j[0] and st_ed_token[pos2:][0][0]<=j[1]][-1]
                        if (input_size-eed)>max_length:
                            pos2 = pos-1
            
                    starts.append(pos2)
                    ends.append(pos)
                    text_end.append(st_ed_token[:pos][-1][1])
                    text_start.append(st_ed_token[pos2:][0][0])

                    z = [i for i,j in enumerate(encoding['offset_mapping']) if text_start[-1] >=j[0] and text_start[-1]<=j[1]]


                ends.append(len(st_ed_token))
                text_end.append(len(text))
                n_start = len(starts)

                for i,(s,e,ts,te) in enumerate(zip(starts,ends,text_start,text_end)):

                    self.essay_id.append(essay_id)
                    self.texts.append(text[ts:te])
                    self.labels.append(g.target.values[s:e].tolist())
                    self.discourse_id.append(g['discourse_id'].values[s:e].tolist())   
                    self.discourse_ids.append(g['discourse_ids'].values[s:e].tolist()) 
                    self.one_hot.append(g[["Ineffective","Adequate","Effective"]].values[s:e].tolist())

                    if (e-s)<3:
                        self.take_it.append([1]*(e-s))
                    elif i==0 and input_size>max_length:
                        self.take_it.append([1]*(e-s-1)+[0])
                    elif i==0:
                        self.take_it.append([1]*(e-s))
                    elif i==n_start-1:
                        self.take_it.append([0]+[1]*(e-s-1))
                    else:
                        self.take_it.append([0]+[1]*(e-s-2)+[0])



                    st_ed = []
                    for tr,t in g[['discourse_text_agg','discourse_text']].values[s:e].tolist():
                        ss,ee = get_text_start_end(text[ts:te],t)

                        st_ed.append([ss,ee])

                    self.st_ed.append(st_ed)

        self.items = []
        for idx in range(len(self.texts)):
            self.items.append(self.make_one_item(idx))
    
        self.df = df
#         self.dfx = pd.DataFrame({'label_ids':self.labels,
#                                  "st_ed":self.st_ed,
# #                                  "st_ed_token":self.st_ed_token,
#                                  "texts":self.texts,
#                                  "essay_id":self.essay_id,
#                                 'discourse_id':self.discourse_id,
#                                 'discourse_ids':self.discourse_ids})

    def find_n_replace(self,text,s,e,text_rep):
        return text[:s] + text_rep + text[e:]

    def prepare_df(self,df):
        df = add_text_to_df(df,self.data_folder)
        df['essay_len'] = df['essay_text'].transform(len)
        if self.input_type=="cls_end":
            df['cls_token'] = df['discourse_type'].map(self.cls_tokens_map)
            df['end_token'] = df['discourse_type'].map(self.end_tokens_map)
        elif self.input_type=="":
            df['cls_token'] = ""
            df['end_token'] = ""


        df['discourse_text_agg'] = df['cls_token']+df['discourse_text']+df['end_token']
        return df
    
    def add_masking(self, data):
        mask_id = self.tokenizer.mask_token_id

        input_len = len(data['input_ids'])
        random_value = random.random() * self.mask_pct
        indices = random.sample(range(input_len), int(input_len * random_value))

        for idx in indices:
            if data['label_ids'][idx] != -1:
                data['input_ids'][idx] = mask_id

    def make_one_item(self,idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            add_special_tokens = self.add_special_tokens
        )
        
        outputs = dict(**encoding)
        
        target = np.zeros(len(encoding.offset_mapping))-100
        target_idx = np.zeros(len(encoding.offset_mapping))-1
        
        token_start_end = []

        for lab_id, ((start_span, end_span),lab) in enumerate(zip(self.st_ed[idx],self.labels[idx])):
            start_token = len(encoding.offset_mapping)
            end_token = 0
            for i,(s,e) in enumerate(encoding.offset_mapping):
                if min(end_span, e) - max(start_span, s) > 0:
                    target[i] = lab
                    target_idx[i] = lab_id+1
                    start_token = min(i,start_token)
                    end_token = max(i,end_token)+1
                
            token_start_end.append([start_token,end_token]) 

        self.token_start_end.append(token_start_end)

        outputs['cls_ids'] = [1 if val in self.cls_tokens else 0 for val in outputs['input_ids']]
        outputs['end_ids'] = [1 if val in self.end_tokens else 0 for val in outputs['input_ids']]

        outputs['label'] = target.tolist()
        outputs['label_ids'] = target_idx.tolist()
        outputs['span_labels'] = self.labels[idx]
        outputs['discourse_ids'] = self.discourse_ids[idx]
        outputs['take_it'] = self.take_it[idx]
        outputs['one_hot_label'] = self.one_hot[idx]

        if self.mask_pct>0:
            self.add_masking(outputs)
        return outputs
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self,idx):
        return self.items[idx]


## =============================================================================== ##

class FeedbackDatasetInference(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 max_length,
                 data_folder=None,
                 add_special_tokens = True,
                 input_type = "cls_end",
                 mask_pct = 0.15
                ):
        
        self.mask_pct = mask_pct
        self.input_type = input_type
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        disc_types = [
                        "Claim",
                        "Concluding Statement",
                        "Counterclaim",
                        "Evidence",
                        "Lead",
                        "Position",
                        "Rebuttal",
                    ]

        if self.input_type=="cls_end":
            self.cls_tokens_map = {label: f"[cls_{label.lower().split()[0]}]" for label in disc_types}
            self.end_tokens_map = {label: f"[end_{label.lower().split()[0]}]" for label in disc_types}  
            self.cls_id_map = {
                                label: self.tokenizer.encode(tkn)[1]
                                for label, tkn in self.cls_tokens_map.items()
                               }
            self.end_id_map = {
                                label: self.tokenizer.encode(tkn)[1]
                                for label, tkn in self.end_tokens_map.items()
                              }
            self.cls_tokens = list(self.cls_id_map.values()) 
            self.end_tokens = list(self.end_id_map.values())

        elif self.input_type=="":
            pass

        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        self.texts = []
        self.essay_id = []
        self.discourse_id = []
        self.discourse_ids = []
        self.st_ed = []
        self.labels = []
        self.token_start_end = []
        self.take_it = []
        df = self.prepare_df(df)
        
        for essay_id,g in tqdm(df.sort_values(["essay_len","discourse_start"]).groupby('essay_id',sort=True)):
            starts = [0]
            ends = []
            text_start = [0]
            text_end = []

            text = g.essay_text.values[0]
            st_ed = []
            st_ed_token = []
            for tr,t,sfrom in g[['discourse_text_agg','discourse_text',"previous_discourse_end"]].values.tolist():
                s,e = get_text_start_end(text,t,sfrom)
                text = clean_text(self.find_n_replace(text,s,e,tr))
                s,e = get_text_start_end(text,t,sfrom)
                st,et = get_text_start_end(text,tr,sfrom)
                st_ed.append([s,e])
                st_ed_token.append([st,et])

            encoding = tokenizer(
                                text,
                                truncation=False,
                                return_offsets_mapping=True,
                                add_special_tokens = True
                                )

            input_size = len(encoding['offset_mapping'])
            z = [0]
            
            if input_size<=max_length:
                self.st_ed.append(st_ed)
                self.essay_id.append(essay_id)
                self.texts.append(text)
                self.labels.append(g.target.values.tolist())
                self.discourse_id.append(g['discourse_id'].values.tolist())   
                self.discourse_ids.append(g['discourse_ids'].values.tolist())
                self.take_it.append([1]*(len(g)))

            else:
                
                while input_size>z[-1]+max_length:
                    x = encoding['offset_mapping'][z[-1]:z[-1]+max_length][-3:]
                    x = [j for j in x if j!=(0,0)]
                    _end = x[-1][0]
                    for pos,(s,e) in enumerate(st_ed_token):
                        if _end >s and _end<=e:
                            break

                    pos2 = min(pos-2,len(st_ed_token)-3) if len(st_ed_token)>3 else pos
                    eed = 0
                    while (input_size-eed)>max_length:
                        eed = [i for i,j in enumerate(encoding['offset_mapping']) if st_ed_token[pos2:][0][0] >=j[0] and st_ed_token[pos2:][0][0]<=j[1]][-1]
                        if (input_size-eed)>max_length:
                            pos2 = pos-1
            
                    starts.append(pos2)
                    ends.append(pos)
                    text_end.append(st_ed_token[:pos][-1][1])
                    text_start.append(st_ed_token[pos2:][0][0])

                    z = [i for i,j in enumerate(encoding['offset_mapping']) if text_start[-1] >=j[0] and text_start[-1]<=j[1]]


                ends.append(len(st_ed_token))
                text_end.append(len(text))
                n_start = len(starts)

                for i,(s,e,ts,te) in enumerate(zip(starts,ends,text_start,text_end)):

                    self.essay_id.append(essay_id)
                    self.texts.append(text[ts:te])
                    self.labels.append(g.target.values[s:e].tolist())
                    self.discourse_id.append(g['discourse_id'].values[s:e].tolist())   
                    self.discourse_ids.append(g['discourse_ids'].values[s:e].tolist()) 

                    if (e-s)<3:
                        self.take_it.append([1]*(e-s))
                    elif i==0 and input_size>max_length:
                        self.take_it.append([1]*(e-s-1)+[0])
                    elif i==0:
                        self.take_it.append([1]*(e-s))
                    elif i==n_start-1:
                        self.take_it.append([0]+[1]*(e-s-1))
                    else:
                        self.take_it.append([0]+[1]*(e-s-2)+[0])



                    st_ed = []
                    for tr,t in g[['discourse_text_agg','discourse_text']].values[s:e].tolist():
                        ss,ee = get_text_start_end(text[ts:te],t)

                        st_ed.append([ss,ee])

                    self.st_ed.append(st_ed)

        self.items = []
        for idx in range(len(self.texts)):
            self.items.append(self.make_one_item(idx))
    
        self.df = df

    def find_n_replace(self,text,s,e,text_rep):
        return text[:s] + text_rep + text[e:]

    def prepare_df(self,df):
        df = add_text_to_df(df,self.data_folder)
        df['essay_len'] = df['essay_text'].transform(len)
        if self.input_type=="cls_end":
            df['cls_token'] = df['discourse_type'].map(self.cls_tokens_map)
            df['end_token'] = df['discourse_type'].map(self.end_tokens_map)
        elif self.input_type=="":
            df['cls_token'] = ""
            df['end_token'] = ""


        df['discourse_text_agg'] = df['cls_token']+df['discourse_text']+df['end_token']
        return df
    

    def make_one_item(self,idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            add_special_tokens = self.add_special_tokens
        )
        
        outputs = dict(**encoding)
        
        target = np.zeros(len(encoding.offset_mapping))-100
        target_idx = np.zeros(len(encoding.offset_mapping))-1
        
        token_start_end = []

        for lab_id, ((start_span, end_span),lab) in enumerate(zip(self.st_ed[idx],self.labels[idx])):
            start_token = len(encoding.offset_mapping)
            end_token = 0
            for i,(s,e) in enumerate(encoding.offset_mapping):
                if min(end_span, e) - max(start_span, s) > 0:
                    target[i] = lab
                    target_idx[i] = lab_id+1
                    start_token = min(i,start_token)
                    end_token = max(i,end_token)+1
                
            token_start_end.append([start_token,end_token]) 

        self.token_start_end.append(token_start_end)

        outputs['label_ids'] = target_idx.tolist()
        outputs['span_labels'] = self.labels[idx]
        outputs['discourse_ids'] = self.discourse_ids[idx]
        outputs['take_it'] = self.take_it[idx]

        return outputs
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self,idx):
        return self.items[idx]

## =============================================================================== ##


class CustomCollatorInference():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["discourse_ids"] = [sample["discourse_ids"] for sample in batch]
        output["take_it"] = [sample["take_it"] for sample in batch]
        output["label_ids"] = [sample["label_ids"] for sample in batch]
        

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":

            output["label_ids"] = [s + (batch_max - len(s)) * [-1] for s in output["label_ids"]]
            output["take_it"] = [s + (batch_max - len(s)) * [-1] for s in output["take_it"]]
            output["discourse_ids"] = [s + (batch_max - len(s)) * [-1] for s in output["discourse_ids"]]
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["label_ids"] = [(batch_max - len(s)) * [-1] + s for s in output["label_ids"]]
            output["take_it"] = [(batch_max - len(s)) * [-1] + s for s in output["take_it"]]
            output["discourse_ids"] = [(batch_max - len(s)) * [-1] + s for s in output["discourse_ids"]]
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]


        
        # print(output["one_hot_label"])
        output["label_ids"] = torch.tensor(output["label_ids"], dtype=torch.long)
        output["take_it"] = torch.tensor(output["take_it"], dtype=torch.uint8)
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["discourse_ids"] = torch.tensor(output["discourse_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)

        return output

## =============================================================================== ##


class CustomCollator():
    def __init__(self, tokenizer,inference=False):
        self.tokenizer = tokenizer
        self.inference = inference

    def __call__(self, batch):
        output = dict()

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["discourse_ids"] = [sample["discourse_ids"] for sample in batch]
        output["take_it"] = [sample["take_it"] for sample in batch]
        output["label_ids"] = [sample["label_ids"] for sample in batch]
        output["one_hot_label"] = [sample["one_hot_label"] for sample in batch]

        if not self.inference:
            output["label"] = [sample["label"] for sample in batch]
            output["span_labels"] = [sample["span_labels"] for sample in batch]
        

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            if not self.inference:
                
                output["label"] = [s + (batch_max - len(s)) * [-100] for s in output["label"]]
                output["span_labels"] = [s + (batch_max - len(s)) * [-100] for s in output["span_labels"]]

            output["one_hot_label"] = [s + (batch_max - len(s)) * [[-1.,-1.,-1.]] for s in output["one_hot_label"]]
            output["label_ids"] = [s + (batch_max - len(s)) * [-1] for s in output["label_ids"]]
            output["take_it"] = [s + (batch_max - len(s)) * [-1] for s in output["take_it"]]
            output["discourse_ids"] = [s + (batch_max - len(s)) * [-1] for s in output["discourse_ids"]]
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            if not self.inference:
                
                output["label"] = [(batch_max - len(s)) * [-100] + s for s in output["label"]]
                output["span_labels"] = [(batch_max - len(s)) * [-100] + s for s in output["span_labels"]]
            
            output["one_hot_label"] = [(batch_max - len(s)) * [[-1.,-1.,-1.]] + s for s in output["one_hot_label"]]
            output["label_ids"] = [(batch_max - len(s)) * [-1] + s for s in output["label_ids"]]
            output["take_it"] = [(batch_max - len(s)) * [-1] + s for s in output["take_it"]]
            output["discourse_ids"] = [(batch_max - len(s)) * [-1] + s for s in output["discourse_ids"]]
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        if not self.inference:
            output["span_labels"] = torch.tensor(output["span_labels"], dtype=torch.long)
            output["label"] = torch.tensor(output["label"], dtype=torch.long)
        
        # print(output["one_hot_label"])
        output["one_hot_label"] = torch.tensor(output["one_hot_label"], dtype=torch.float)
        output["label_ids"] = torch.tensor(output["label_ids"], dtype=torch.long)
        output["take_it"] = torch.tensor(output["take_it"], dtype=torch.long)
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["discourse_ids"] = torch.tensor(output["discourse_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)

        return output