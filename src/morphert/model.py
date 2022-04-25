from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertModel, BertPreTrainedModel

class MorphertDataset(Dataset):
  def __init__(self, idxs, vocabs, embs):
    assert max(idxs) < len(vocabs)
    assert max(idxs) < embs.shape[0]
    self.vocabs = vocabs
    self.embs = embs
    self.idxs = idxs

  def __len__(self):
    return len(self.idxs)

  def __getitem__(self, idx):
    idx = self.idxs[idx]
    return {
        "word": self.vocabs[idx],
        "vec": self.embs[idx, :],
    }

  def get_word(self, idx):
    return self.vocabs[self.idxs[idx]]

class DataCollator:
  def __init__(self, tokenizer, device=None):
    if not device:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device
    self.tokenizer = tokenizer

  def __call__(self, Xs):
    words = [x["word"] for x in Xs]
    vec = np.vstack([x["vec"] for x in Xs])
    vec = torch.tensor(vec, dtype=torch.float32).to(self.device)
    input_batch = self.tokenizer(words, return_tensors="pt", padding="longest")
    input_batch = input_batch.to(self.device)     
    return {
        **input_batch, "labels": vec, "words": words
    }

@dataclass
class MorphertOutput:
  loss: float
  predictions: np.ndarray

class MorphertModel(BertPreTrainedModel):
  def __init__(self, config, *args, **kwargs):
    super().__init__(config, **kwargs)
    emb_dim = kwargs.get("emb_dim", 100)
    hdim = self.config.hidden_size
    self.bert = BertModel(config)
    self.proj = nn.Linear(hdim, emb_dim)
  
  def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        
    outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    cls_vec = outputs.last_hidden_state[:, 0]
    pred_vec = self.proj(cls_vec)    

    if labels is not None:
      loss_fct = nn.MSELoss()
      loss = loss_fct(pred_vec, labels)
    else:
      loss = float("NaN")

    return MorphertOutput(loss, pred_vec)