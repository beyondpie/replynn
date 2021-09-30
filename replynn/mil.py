import os
import shutil
import sys
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        nquery: int = 1,
        kdim: int = 32,
        vdim: int = 32,
        embedim: int = 256,
        nhead: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.nquery: int = nquery
        self.kdim: int = kdim
        self.vdim: int = vdim
        self.embedim: int = embedim
        self.nhead: int = nhead
        self.dropout: float = dropout
        self.query: torch.Tensor = Parameter(
            data=torch.Tensor(self.nquery, self.embedim), requires_grad=True
        )
        self.layernorm_1: nn.Module = nn.LayerNorm(
            normalized_shape=kdim, eps=1e-5, elementwise_affine=True
        )
        self.layernorm_2: nn.Module = nn.LayerNorm(
            normalized_shape=self.nquery * self.embedim,
            eps=1e-5,
            elementwise_affine=True,
        )
        self.multiheadattn: nn.Module = nn.MultiheadAttention(
            embed_dim=self.embedim,
            num_heads=self.nhead,
            dropout=self.dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=self.kdim,
            vdim=self.vdim,
        )
        self.w: nn.Module = nn.Sequential(
            nn.Linear(in_features=self.vdim, out_features=self.embedim, bias=True),
            nn.Sigmoid(),
        )
        self.reset_query()
        return None

    def reset_query(self) -> None:
        nn.init.xavier_uniform_(tensor=self.query)
        return None

    def forward(
        self, x: torch.Tensor, m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MultiHeadAttention

        - Output:
          - out: [batch_size, embedim * nquery]
        - Input:
          - x: [batch_size, nchannel, reply_embedim]
          - m: [batch_size, nchannel]
        """
        n, s, e = x.shape
        ## tile is absent in pytorch version 1.7.0
        # q = torch.tile(self.query.unsqueeze(dim=1), dims=(1, n, 1))
        q = self.query.unsqueeze(dim = 1).repeat(repeats = (1, n, 1))
        ## m1: [s, n, e]
        m1 = self.layernorm_1(x).transpose(0, 1)
        v = m1
        k = m1
        ## attn_out: [self.nquery, n, self.embedim]
        ## attn_weights: [n, self.nquery, s], average weights per head
        ## True for masked positions
        key_padding_mask = m < 1.0
        attn_out, attn_weights = self.multiheadattn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=None,
        )
        ## attn_out: [n, self.nquery, self.emebdim]
        attn_out = attn_out.transpose(0,1)
        avg_x = torch.sum(x, dim=1) / torch.sum(m, dim=1, keepdim=True)
        # g = torch.tile(self.w(avg_x).unsqueeze(dim=1), dims=(1, self.nquery, 1))
        g = self.w(avg_x).unsqueeze(dim=1).repeat(repeats = (1, self.nquery, 1)) 
        ## gated attention
        r = (g * attn_out).contiguous().view(n, -1)
        r = self.layernorm_2(r)
        return r


class TumorTypeMLP(nn.Module):
    def __init__(
        self,
        indim: int,
        l1: int,
        l2: int,
        l3: int,
        outdim: int,
        dropout: float = 0.8,
        ttnlayer: int = 2,
    ) -> None:
        super().__init__()
        self.indim: int = indim
        self.l1: int = l1
        self.l2: int = l2
        self.l3: int = l3
        self.outdim: int = outdim
        self.dropout: float = dropout
        self.ttnlayer: int = ttnlayer

        if self.ttnlayer == 2:
            self.l = nn.Sequential(
                ## input => hidden layer 1
                nn.Linear(in_features=self.indim, out_features=self.l1, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                ## hidden layer 1 => hidden layer 2
                nn.Linear(in_features=self.l1, out_features=self.l2, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                ## hidden layer 2 => output
                nn.Linear(in_features=self.l2, out_features=self.outdim, bias=True),
                nn.ReLU(),
            )
        elif self.ttnlayer == 3:
            self.l = nn.Sequential(
                ## input => hidden layer 1
                nn.Linear(in_features=self.indim, out_features=self.l2, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                ## hidden layer 1 => hidden layer 2
                nn.Linear(in_features=self.l2, out_features=self.l1, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                ## hidden layer 2 => hidden layer 3
                nn.Linear(in_features=self.l1, out_features=self.l2, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                ## hidden layer 3 => output
                nn.Linear(in_features=self.l2, out_features=self.outdim, bias=True),
                nn.ReLU(),
            )
        elif self.ttnlayer == 5:
            self.l = nn.Sequential(
                nn.Linear(in_features=self.indim, out_features=self.l3, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.l3, out_features=self.l2, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.l2, out_features=self.l1, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.l1, out_features=self.l2, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.l2, out_features=self.l3, bias=True),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.l3, out_features=self.outdim, bias=True),
                nn.ReLU(),
            )
        else:
            raise RuntimeError(f"ttnlayer should be 2, 3 or 5 not {self.ttnlayer}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l(x)


class GAMIL(nn.Module):
    def __init__(
        self,
        index2nm: Dict[int, str],
        reply_embedim: int,
        ttnlayer: int,
        tt_l1: int,
        tt_l2: int,
        tt_l3: int,
        nchannel: int,
        dropout: float,
        nquery: int = 2,
        nhead: int = 8,
        mil_embedim: int = 256
    ):
        super().__init__()
        self.index2nm: Dict[int, str] = index2nm
        self.reply_embedim: int = reply_embedim
        self.ttnlayer: int = ttnlayer
        self.tt_l1: int = tt_l1
        self.tt_l2: int = tt_l2
        self.tt_l3: int = tt_l3
        self.nchannel: int = nchannel
        self.mlp_dropout: float = dropout
        self.mil_dropout: float = 0.0
        self.nhead: int = nhead
        self.nquery: int = nquery
        self.mil_embedim: int = mil_embedim

        self.mil: nn.Module = GatedMultiHeadAttention(
            nquery=self.nquery,
            kdim=self.reply_embedim,
            vdim=self.reply_embedim,
            embedim=self.mil_embedim,
            nhead=self.nhead,
            dropout=self.mil_dropout,
        )
        self.tumortype_classifier: nn.Module = TumorTypeMLP(
            indim=self.mil_embedim * self.nquery,
            l1=self.tt_l1,
            l2=self.tt_l2,
            l3=self.tt_l3,
            outdim=len(self.index2nm),
            dropout=self.mlp_dropout,
            ttnlayer=self.ttnlayer,
        )

    def forward(
        self, tt_x: torch.Tensor, tt_m: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tt_embed: torch.Tensor = self.mil(x=tt_x, m=tt_m)
        tt_out: torch.Tensor = self.tumortype_classifier(tt_embed)
        return tt_out, tt_embed
