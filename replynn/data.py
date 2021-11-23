import pkg_resources
from typing import Dict, List, OrderedDict, Tuple
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np



pad_id: int = 0
outdim: int = 21
seq_max_len: int = 30
## set atchley
atchleyf: str = pkg_resources.resource_filename('replynn', 'data/atchley_factors.csv')
atchley_dim: int = 5
atchley_weight: np.ndarray = np.loadtxt(
    fname=atchleyf, delimiter=",", skiprows=1, usecols=tuple(range(1, atchley_dim + 1))
)
with open(atchleyf, "r") as f:
    AAs: List[str] = [l.split(",")[0] for i, l in enumerate(f.readlines()) if i > 0]
num_aas: int = len(AAs)
## amino start from 1 since we have pad_id = 0
word2index: Dict[str, int] = {k: (i + 1) for i, k in enumerate(AAs)}
index2word: Dict[int, str] = {v: k for k, v in word2index.items()}

## blosum62
blosum62f :str = pkg_resources.resource_filename('replynn', 'data/blosum62.iij')
blosum62: np.ndarray = np.loadtxt(
    fname=blosum62f, comments="#", usecols=tuple(range(1, 25))
)
with open(blosum62f, "r") as f:
    aa2index_blosum: Dict[str, int] = {
        l.split()[0]: (i - 7) for i, l in enumerate(f.readlines()) if i >= 7
    }

index_of_aa_in_blosum: List[int] = [
    aa2index_blosum[index2word[i]] for i in range(1, num_aas + 1)
]

raw_blosum_score = np.ones(shape=[num_aas + 1, outdim]) * (-4)
raw_blosum_score[1:, 1:] = blosum62[index_of_aa_in_blosum][:, index_of_aa_in_blosum]
raw_blosum_score[0, 0] = 1.0

## non-negative version of blosum
nng_blosum_score = raw_blosum_score.copy()
nng_blosum_score[nng_blosum_score <= 0] = 0.0


class SubjectEmbedDataset(Dataset):
    def __init__(self, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.mask = mask

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, id):
        return self.x[id], self.mask[id], self.y[id]


class VAEDataset(Dataset):
    def __init__(self, cdr3s: List[str]):
        self.cdr3s: List[str] = cdr3s

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        """Padded x, length of x, mask of x in order."""
        padx: torch.LongTensor = torch.zeros([seq_max_len], dtype=torch.long)
        mask: torch.Tensor = torch.zeros([seq_max_len], dtype=torch.float)
        for i, c in enumerate(self.cdr3s[index]):
            padx[i] = word2index.get(c, pad_id)
            mask[i] = 1.0
        l: int = len(self.cdr3s[index])
        return (padx, torch.tensor(l, dtype=torch.long), mask)

    def __len__(self) -> int:
        return len(self.cdr3s)

class VAEDataLoader(DataLoader):
    def __init__(self, ds: VAEDataset, batch_size:int, shuffle: bool, pin_memory: bool = True):
        super().__init__(dataset = ds, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory)

def set_VAEDataLoader(
    config: Tuple[str, str, int, int], shuffle: bool = False
) -> VAEDataLoader:
    fnm, sep, aa_col, batch_size = config
    with open(fnm, "r") as f:
        lines: List[str] = f.readlines()
    cdr3s: List[str] = []
    for l in lines:
        cdr3: str = l.rstrip().split(sep)[aa_col]
        if len(cdr3) <= seq_max_len:
            cdr3s.append(cdr3)
    ds: VAEDataset = VAEDataset(cdr3s=cdr3s)
    dl: VAEDataLoader = VAEDataLoader(
        dataset=ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )
    return dl
