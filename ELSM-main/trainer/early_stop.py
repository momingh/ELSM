from logging import getLogger
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from scipy.sparse import csr_matrix, coo_matrix
import pandas as pd

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0):
        """
        patience (int): 在验证损失（或其他性能指标）不再改善的情况下，等待多少个训练周期（或称为"epochs"）之后才触发早停。
                            Default: 7
        verbose (bool): 参数是一个布尔值，如果设置为True，那么在每次验证损失（或性能指标）有所改善时都会打印一条消息。
                            Default: False
        delta (float): delta 参数表示被认为是性能改善的最小阈值。如果验证损失（或性能指标）的改善小于 delta，则不会被视为足够显著的改善，不会重置耐心计数器。只有当验证损失改善大于 delta 时，才会重置耐心计数器。
                            Default: 0
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        if score > self.best_score+self.delta:
            return False
        return True

    def __call__(self, score):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

