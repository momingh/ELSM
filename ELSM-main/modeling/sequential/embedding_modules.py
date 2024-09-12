# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import pdb
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.initialization import truncated_normal
from modeling.sequential.transformer import MMFormer


class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        dataset_image_path: str,
        dataset_text_path: str,
        adj,
    ) -> None:
        super().__init__()
        self.id_adj = adj

        self._item_embedding_dim: int = item_embedding_dim

        self._item_emb = torch.nn.Embedding(num_items + 1, self._item_embedding_dim, padding_idx=0)
        self.reset_params()

        self.v_feat = torch.load(dataset_image_path)
        self.t_feat = torch.load(dataset_text_path)

        if isinstance(self.v_feat, np.ndarray):
            self.v_feat = torch.from_numpy(self.v_feat).type(torch.FloatTensor)
        if isinstance(self.t_feat, np.ndarray):
            self.t_feat = torch.from_numpy(self.t_feat).type(torch.FloatTensor)

        self.v_feat = torch.cat(
            [torch.zeros((1, self.v_feat.size(1))).type(torch.FloatTensor), self.v_feat.type(torch.FloatTensor)], dim=0)
        self.t_feat = torch.cat(
            [torch.zeros((1, self.t_feat.size(1))).type(torch.FloatTensor), self.t_feat.type(torch.FloatTensor)], dim=0)
        self._image_emb = torch.nn.Embedding.from_pretrained(self.v_feat, freeze=False, padding_idx=0)
        self._text_emb = torch.nn.Embedding.from_pretrained(self.t_feat, freeze=False, padding_idx=0)

        self.knn_k = 10
        self.n_layers = 2

        indices, image_adj = self.get_knn_adj_mat(self._text_emb.weight.detach())
        self.v_mm_adj = image_adj
        self.v_gcn = GCN(in_features=64, hidden_features=64, out_features=64, n_layers=2)

        indices, text_adj = self.get_knn_adj_mat(self._text_emb.weight.detach())
        self.t_mm_adj = text_adj
        self.t_gcn = GCN(in_features=64, hidden_features=64, out_features=64, n_layers=2)
        del text_adj
        del image_adj
        del indices

        self.query_num = 2
        self.fusion = MMFormer(
            self.query_num, 1, self._item_embedding_dim, 4,
            0, 0.8
        )

        self.mm_proj = torch.nn.Linear(self.query_num * self._item_embedding_dim, self._item_embedding_dim)
        torch.nn.init.normal_(self.mm_proj.weight, mean=0.0, std=0.02)

        self.id_linear = torch.nn.Linear(self._item_embedding_dim, self._item_embedding_dim)
        torch.nn.init.normal_(self.id_linear.weight, mean=0.0, std=0.02)

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if '_item_emb' in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return None

    def ELSM(self, item_ids: torch.Tensor) -> torch.Tensor:

        self.id = self.buildItemGraph(self._item_emb.weight,
                                      adj=self.id_adj.to(self._item_emb.weight.device)) + self._item_emb.weight

        self.vv = self._image_emb.weight
        self.tt = self._text_emb.weight
        self.v_h = self.v_gcn(h=self.vv, adj=self.t_mm_adj.to(self._image_emb.weight.device))
        self.t_h = self.t_gcn(h=self.tt, adj=self.t_mm_adj.to(self._text_emb.weight.device))
        self.id_embeddings = self.id_linear(self.id)

        self.v_embeddings = self.vv
        self.t_embeddings = self.tt

        fuse = torch.cat([self.v_embeddings.unsqueeze(1), self.t_embeddings.unsqueeze(1)], 1)
        fuse = self.fusion(fuse)[:, :self.query_num].view((fuse.size(0), -1))
        self.fuse = F.normalize(self.mm_proj(fuse), p=2, dim=-1, eps=1e-12)
        # resnet
        self.fuse_embeddings = F.normalize(self.id_embeddings, p=2, dim=-1, eps=1e-12) + self.vv + self.tt + self.fuse

        return self.fuse_embeddings[item_ids]

    def contrastive_loss(self):
        id_var, id_mean = torch.var(self.id_embeddings), torch.mean(self.id_embeddings)
        v_var, v_mean = torch.var(self.v_h), torch.mean(self.v_h)
        t_var, t_mean = torch.var(self.t_h), torch.mean(self.t_h)

        L_ccl = (
                (torch.abs(id_var - v_var) +
                 torch.abs(id_mean - v_mean)).mean() +
                (torch.abs(id_var - t_var) +
                 torch.abs(id_mean - t_mean)).mean() +
                (torch.abs(v_var - t_var) +
                 torch.abs(v_mean - t_mean)).mean())

        return L_ccl

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(mm_embeddings.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def buildItemGraph(self, h, adj):
        sum_h = torch.zeros_like(h)
        for i in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            sum_h += h

        return sum_h

    def pack_edge_index(self, inter_mat):
        inter_mat = inter_mat.coalesce()
        rows = inter_mat.indices()[0]
        cols = inter_mat.indices()[1]
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class GCN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.n_layers = n_layers

        self.layers.append(torch.nn.Linear(in_features, hidden_features))

        for _ in range(1, n_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_features, hidden_features))

        self.layers.append(torch.nn.Linear(hidden_features, out_features))

    def forward(self, h, adj):
        sum_h = torch.zeros_like(h)

        for i in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            h = self.layers[i](h)
            h = F.relu(h)

            sum_h = sum_h + h

        return sum_h


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
            self,
            num_items: int,
            item_embedding_dim: int,
            item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
