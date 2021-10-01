import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sort_seq_by_len(
    x: torch.LongTensor,
    l: torch.LongTensor,
    batch_first: bool = True,
    descending: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ref: torch.nn.utils.rnn.pack_padded_sequence
    """
    sorted_lens, sorted_indices = torch.sort(l, dim=-1, descending=descending)
    sorted_indices = sorted_indices.to(x.device)
    batch_dim = 0 if batch_first else 1
    x_: torch.LongTensor = x.index_select(batch_dim, sorted_indices)
    return (x_, sorted_lens, sorted_indices)


def randn(mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.Tensor:
    eps = torch.randn_like(logvar)
    sd = torch.exp(0.5 * logvar)
    return mu + eps * sd


def kl_to_stdn(mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.Tensor:
    """
    KL(q(z)||p0(z)), where p0(z) is the prior of the hidden variable z.
    and p0 is standard normal.
    """
    kl = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
    return kl


class GRUEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        hidden_size: int,
        rnndp: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.dpr = rnndp
        self.embedding: nn.Module = embedding
        self.embed_dim: int = self.embedding.embedding_dim
        self.hidden_size = hidden_size
        self.rnn: nn.Module = nn.GRU(
            input_size=out_channels,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            dropout=self.dpr,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def after_embed_hook(self, embedx: torch.FloatTensor) -> torch.Tensor:
        return embedx

    def forward(
        self,
        padx: torch.LongTensor,
        len_of_x: torch.LongTensor,
        lh: torch.Tensor = None,
        enforce_sorted=False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Output is padedout and last hidden.

        See torch.nn.utils.rnn.pack_padded_sequence and
        check torch.nn.utils.rnn.PackedSequence class for details.
        """
        embeded = self.embedding(padx)
        embeded = torch.transpose(input=embeded, dim0=1, dim1=2)
        afembed = self.after_embed_hook(embeded)
        packedx = nn.utils.rnn.pack_padded_sequence(
            afembed, len_of_x.cpu(), batch_first=True, enforce_sorted=enforce_sorted
        )
        ## lh shape : [num_layers(=1) * num_directtions, batch_size, hidden_size]
        ## lh is the last output of packedout,
        ## and it's automatically based on the length of sequence.
        packedout, lh = self.rnn(packedx, lh)
        ## [batch_size, len_of_seq, hidden_dim]
        ## NOTE: len_of_seq based on the max length of sequences in the current batch.
        ## So different batches may have different len_of_seqs.
        paddedout, _ = nn.utils.rnn.pad_packed_sequence(
            packedout, batch_first=True, padding_value=0.0, total_length=None
        )
        lh = lh.transpose(0, 1).flatten(1)
        return (paddedout, lh)


class ConvGRUEncoder(GRUEncoder):
    def __init__(
        self,
        embedding: nn.Module,
        hidden_size: int,
        out_channels: int,
        rnndp: float = 0.0,
    ):
        super().__init__(
            embedding=embedding,
            hidden_size=hidden_size,
            rnndp=rnndp,
            bidirectional=False,
        )
        self.embed_dim: int = self.embedding.embedding_dim
        self.out_channels: int = out_channels
        self.conv1d3mer = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
        )

    def after_embed_hook(self, embedx: torch.FloatTensor) -> torch.Tensor:
        convembed = self.conv1d3mer(embedx)
        convembed = torch.transpose(input=convembed, dim0=1, dim1=2)
        ## TODO: add batch normalization
        return convembed


class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        hidden_size: int,
        rnndp: float = 0.0,
    ) -> None:
        super().__init__(
            embedding=embedding,
            hidden_size=hidden_size,
            rnndp=rnndp,
            bidirectional=True,
        )


class MLPDecoder(nn.Module):
    def __init__(self, l1: int, l2: int):
        super().__init__()
        self.mlp: nn.Module = nn.Sequential(
            nn.Linear(in_features=l1, out_features=l2, bias=True),
            # nn.BatchNorm1d(num_features = l2),
            # nn.Tanh(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=l2, out_features=outdim * seq_max_len, bias=True),
            # nn.BatchNorm1d(num_features=outdim * seq_max_len),
            # nn.Tanh(),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.FloatTensor:
        """Output [batch_size, seq_max_len, out_dim]"""
        r = self.mlp(z)
        r = r.reshape([-1, seq_max_len, outdim])
        return r


class ConvGRUMLPVAE(nn.Module):
    def __init__(
        self,
        latent_size: int = 48,
        hidden_size: int = 16,
        out_channels: int = 32,
        rnndp: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.bidirectional: bool = bidirectional
        self.out_channels: int = out_channels
        self.indim_of_y_nn: int = 2 * hidden_size if self.bidirectional else hidden_size
        self.latent_size = latent_size
        self.decode_l2: int = 4 * self.latent_size
        self.embedding: nn.Embedding = self.set_embedding()
        self.encoder: ConvGRUEncoder = ConvGRUEncoder(
            embedding=self.embedding,
            hidden_size=self.hidden_size,
            out_channels=self.out_channels,
            rnndp=rnndp,
            bidirectional=bidirectional,
        )
        ## remove Tanh in both mu and logvar
        ## not limit the value of mu and var
        self.mumlp: nn.Module = nn.Sequential(
            nn.Linear(
                in_features=self.indim_of_y_nn,
                out_features=self.latent_size,
                bias=True,
            )
            # nn.Tanh(),
        )
        self.logvmlp: nn.Module = nn.Sequential(
            nn.Linear(
                in_features=self.indim_of_y_nn,
                out_features=self.latent_size,
                bias=True,
            )
            # nn.Tanh(),
        )
        self.decoder: MLPDecoder = MLPDecoder(
            l1=self.latent_size,
            l2=self.decode_l2,
        )
        return None

    def set_embedding(self) -> nn.Embedding:
        embedding = np.zeros([outdim, atchley_dim])
        embedding[
            1:,
        ] = atchley_weight
        return nn.Embedding.from_pretrained(
            embeddings=torch.from_numpy(embedding).to(torch.float), freeze=True
        )

    def encode(
        self, padx: torch.LongTensor, len_of_x: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        _, lh = self.encoder(padx=padx, len_of_x=len_of_x, lh=None)
        z_mu: torch.Tensor = self.mumlp(lh)
        z_logvar: torch.Tensor = self.logvmlp(lh)
        return (z_mu, z_logvar)

    def reconst_loss(
        self, out: torch.Tensor, x: torch.LongTensor, l: torch.Tensor, m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reconst_loss using blossum62 and cross entropy. Summation across batch.

        Input
        - out: [batch_size, max_seq_len, outdim]
        - x: [batch_size, seq_max_len]
        - l: [batch_size]
        - m: mask, [batch_size, seq_max_len]
        NOTE:
        loss design:
        - for one sample i: - sum_j  blossum_ij * logprob_ij
          - compared with cross entropy: - logprob_i
        """
        logp = out.log_softmax(dim=2)
        ## [batch_size, seq_max_len, outdim]
        w: torch.Tensor = blosum62[x].to(out.device)
        ## [batch_size, seq_max_len, outdim]
        wlogp: torch.Tensor = logp * w
        ## [batch_size, seq_max_len]
        ele_lb = torch.sum(wlogp, dim=2) * m
        ## [batch_size], normalized by sequence length
        ## NOTE: remove the normalization by sequence length
        # if self.normalize_seq > 0:
        #     lb = torch.sum(input=ele_lb, dim=1) / l.to(torch.float)
        # else:
        lb = torch.sum(input=ele_lb, dim=1)
        lb = -torch.sum(lb)
        ## cross entropy loss
        ## [batch_size, seq_max_len], ignore the padding index
        ele_logp = torch.sum(logp * F.one_hot(x, num_classes=outdim), dim=2) * m
        ## [batch_size], normalized by sequence length
        # if self.normalize_seq > 0:
        #     lce = torch.sum(ele_logp, dim=1) / l.to(torch.float)
        # else:
        ## NOTE: remove the normalization by sequence length
        lce = torch.sum(ele_logp, dim=1)
        lce = -torch.sum(lce)
        return (lb, lce)

    def forward(
        self, padx: torch.LongTensor, len_of_x: torch.LongTensor, mask: torch.Tensor
    ):
        """Get z, pred_out, z_mu, z_logvar in order."""
        x, l, sorted_indices = descend_sort_seq_by_length(
            x=padx, lens=len_of_x, batch_first=True
        )
        m = mask.index_select(dim=0, index=sorted_indices)
        z_mu, z_logvar = self.encode(padx=x, len_of_x=l)
        z = normal_reparam(z_mu, z_logvar)
        out = self.decoder(z)
        return (z, out, z_mu, z_logvar, x, l, m)
