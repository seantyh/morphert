from scipy.spatial.distance import cdist

from dataclasses import dataclass
import numpy as np
from .utils import NeighborPrediction

@dataclass
class SpaceIndices:
    N: int
    dist_vec: np.ndarray    
    dist_qs: np.ndarray
    dist_mean: float    
    dist_mtop5: float
    dist_range: float

    def __repr__(self):
        return ("<SpaceIndices: {N}, "
                "{dist_mtop5:.2f}/{dist_range:.2f}>"
                ).format(
            **self.__dict__
        )

def compute_space_indices(pred_neigh: NeighborPrediction):
    pred_vector = pred_neigh.pred_vec
    neigh_vecs = pred_neigh.neigh_vecs
    dist_vec = 1-cdist([pred_vector], neigh_vecs, 'cosine')[0]
    dist_qs = np.quantile(dist_vec, [0.1,0.25,0.5,0.75,0.9])
    ret = SpaceIndices(
        N=len(neigh_vecs),
        dist_vec=dist_vec,
        dist_qs = dist_qs,
        dist_mean=np.mean(dist_vec),
        dist_mtop5=np.mean(sorted(dist_vec)[-5:]),        
        dist_range = dist_qs[4] - dist_qs[0]
    )
    return ret
