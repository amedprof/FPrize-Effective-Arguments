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
            self.special_tokens = list(self.cls_id_map.values()) + list(self.end_id_map.values())
        elif self.input_type=="":
            pass

        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.mlm_probability = mask_pct
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
                # oc = 0
                # while oc<=2:
                #     while input_size>z[-1]+max_length:
                #         x = encoding['offset_mapping'][z[-1]:z[-1]+max_length][-3:]
                #         x = [j for j in x if j!=(0,0)]
                #         _end = x[-1][0]
                #         for pos,(s,e) in enumerate(st_ed_token):
                #             if _end >s and _end<=e:
                #                 break

                #         pos2 = min(pos-2,len(st_ed_token)-3) if len(st_ed_token)>3 else pos
                #         eed = 0
                #         while (input_size-eed)>max_length:
                #             eed = [i for i,j in enumerate(encoding['offset_mapping']) if st_ed_token[pos2:][0][0] >j[0] and st_ed_token[pos2:][0][0]<=j[1]][-1]
                #             if (input_size-eed)>max_length:
                #                 pos2 = pos-1
                #             oc+=1
                #         starts.append(pos2)
                #         ends.append(pos)
                #         text_end.append(st_ed_token[:pos][-1][1])
                #         text_start.append(st_ed_token[pos2:][0][0])

                #         z = [i for i,j in enumerate(encoding['offset_mapping']) if text_start[-1] >j[0] and text_start[-1]<=j[1]]


                #     ends.append(len(st_ed_token))
                #     text_end.append(len(text))
                #     n_start = len(starts)

                #     for i,(s,e,ts,te) in enumerate(zip(starts,ends,text_start,text_end)):

                #         self.essay_id.append(essay_id)
                #         self.texts.append(text[ts:te])

                # if oc>2:
                x = encoding['offset_mapping'][:max_length][-3:]
                x = [j for j in x if j!=(0,0)]
                _end = x[-1][0]
                self.texts.append(text[:_end])

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

        inputs = torch.tensor(outputs['input_ids'].copy())
        target = torch.tensor(outputs['input_ids'].copy())

        if self.mlm_probability>0:
            probability_matrix = torch.full(inputs.shape, self.mlm_probability)

            special_tokens_mask = torch.tensor([1 if val in self.special_tokens else 0 for val in outputs['input_ids']])
            probability_matrix.masked_fill_(special_tokens_mask, value=1)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            target[~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer),inputs.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

        outputs['label'] = target.tolist()
        outputs['input_ids'] = inputs.tolist()

        return outputs
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self,idx):
        return self.items[idx]


## =============================================================================== ##
class CustomCollator():
    def __init__(self, tokenizer,inference=False):
        self.tokenizer = tokenizer
        self.inference = inference

    def __call__(self, batch):
        output = dict()

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["label"] = [sample["label"] for sample in batch]
            
        

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":

            output["label"] = [s + (batch_max - len(s)) * [-100] for s in output["label"]]
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:

            output["label"] = [(batch_max - len(s)) * [-100] + s for s in output["label"]]
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["label"] = torch.tensor(output["label"], dtype=torch.long)
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)

        return output