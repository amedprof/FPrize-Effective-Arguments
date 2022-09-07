import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.utils.checkpoint

class FeedbackModel(nn.Module):
    def __init__(self,
                 model_name,
                 num_labels,
                 test=False,
                 config_path=None,
                 pretrained_path = None,
                 tokenizer_size = 0,
                 use_dropout=False,
                 use_gradient_checkpointing = False
                 ):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) if not config_path else torch.load(config_path)

        self.use_dropout = use_dropout
        if not self.use_dropout:
            self.config.update(
                                {
                                    "hidden_dropout_prob": 0.0,
                                    "attention_probs_dropout_prob": 0.0,
                                }
                                    )

        self.backbone = AutoModel.from_pretrained(model_name,config=self.config) if not config_path else AutoModel.from_config(self.config)
        
        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)


    def forward(self,b):
        x = self.backbone(b["input_ids"],b["attention_mask"]).last_hidden_state
        x = self.dropout(x)
        x = self.fc(x)
        return x