from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.spatial.distance import cdist
import functools
import torch


@dataclass
class NeighborPrediction:
    word: str
    pred_vec: np.ndarray
    neigh_words: List[str]
    neigh_vecs: np.ndarray   

    def __repr__(self):
        return (f"<NeighborPrediction {self.word}: "
            f"{' '.join(self.neigh_words[:5])}>")


def compute_metric(ref_words, pred_vecs, ref_emb, ds):
    dist_mat = cdist(pred_vecs, ref_emb)  
    pred_idxs = np.argmin(dist_mat, axis=1)
    pred_words = [ds.get_word(x) for x in pred_idxs]
    # print(*[(a,b) for a, b in zip(pred_words, ref_words)], sep="\n")
    return sum((a==b for a, b in zip(pred_words, ref_words)))

def print_neighbors(texts, tokenizer, model, full_emb, full_ds, topk=5):
    pred_vecs = predict_vectors(texts, tokenizer, model)
    dist_mat = cdist(pred_vecs, full_emb)  
    pred_idxs = np.argsort(dist_mat, axis=1)
    
    for i, word in enumerate(texts):
        pred_x = pred_idxs[i, :]
        pred_words = [full_ds.get_word(x) for x in pred_x[:topk]]
        marker = "*" if word not in full_ds.vocabs else " "                
        print(marker, word+":", " ".join(pred_words))        


def predict_neighbors(texts, tokenizer, model, full_emb, full_ds, topk=5):
    pred_vecs = predict_vectors(texts, tokenizer, model)
    dist_mat = cdist(pred_vecs, full_emb)  
    pred_idxs = np.argsort(dist_mat, axis=1)
    
    preds = []
    for i, word in enumerate(texts):
        pred_x = pred_idxs[i, :]
        pred_words = [full_ds.get_word(x) for x in pred_x[:topk]]        
        pred_x = NeighborPrediction(
            word=word, pred_vec=pred_vecs[i], 
            neigh_words=pred_words, 
            neigh_vecs=full_emb[pred_x[:topk], :]
        )
        preds.append(pred_x)
    
    return preds

def predict_vectors(texts, tokenizer, model):
    model.eval()
    in_batch = tokenizer(texts, padding=True, return_tensors="pt")
    in_batch = in_batch.to("cuda")
    with torch.no_grad():    
        out = model(**in_batch)  
        pred_vecs = out.predictions.cpu().numpy()
    return pred_vecs

def predict_from_token(token, tokenizer, norm_wemb, topk=20):
    tgt_idx = tokenizer.convert_tokens_to_ids([token])
    sorted_idxs = torch.argsort(-torch.matmul(norm_wemb, norm_wemb[tgt_idx[0]]))
    return tokenizer.convert_ids_to_tokens(sorted_idxs[:topk])

def get_print_neighbors_fn(tokenizer, model, full_emb, full_ds):
    return functools.partial(print_neighbors, 
        tokenizer=tokenizer, model=model, 
        full_emb=full_emb, full_ds=full_ds)

def get_predict_neighbors_fn(tokenizer, model, full_emb, full_ds):
    return functools.partial(predict_neighbors, 
        tokenizer=tokenizer, model=model, 
        full_emb=full_emb, full_ds=full_ds)

def get_predict_vectors_fn(tokenizer, model):
    return functools.partial(predict_vectors, 
        tokenizer=tokenizer, model=model)
    
def get_predict_from_token_fn(tokenizer, model):
    wemb = model.bert.embeddings.word_embeddings.weight
    norm_wemb = wemb / wemb.norm(dim=1).unsqueeze(1)
    return functools.partial(predict_from_token, 
        tokenizer=tokenizer, norm_wemb=norm_wemb)