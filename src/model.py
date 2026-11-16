import torch.nn as nn
from transformers import AutoModel


class TwoHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc = AutoModel.from_config(config)
        h = config.hidden_size
        self.drop, self.rec, self.band = nn.Dropout(0.1), nn.Linear(h, 3), nn.Linear(h, 3)

    def forward(self, ids, att):
        z = self.drop(self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0])
        return self.rec(z), self.band(z)