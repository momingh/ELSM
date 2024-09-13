import pdb
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
def get_graph(path):
    src, dst = [], []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        print('Create ID graph...')
        for row in tqdm(reader):
            sequence_item_ids = [int(item_id) for item_id in row['sequence_item_ids'].split(',')[:-2]]
            for node_pair in zip(sequence_item_ids, sequence_item_ids[1:]):
                # print(node_pair)
                src.append(node_pair[0] + 1)
                dst.append(node_pair[1] + 1)
            for node_pair in zip(sequence_item_ids, sequence_item_ids[2:]):
                # print(node_pair)
                src.append(node_pair[0] + 1)
                dst.append(node_pair[1] + 1)
            for node_pair in zip(sequence_item_ids, sequence_item_ids[3:]):
                # print(node_pair)
                src.append(node_pair[0] + 1)
                dst.append(node_pair[1] + 1)
        src_tensor = torch.tensor(src, dtype=torch.long)
        dst_tensor = torch.tensor(dst, dtype=torch.long)
        indices = torch.stack((src_tensor, dst_tensor))

    # 找出图中的节点数量（假设节点索引从0开始）
    num_nodes = max(src + dst) + 1

    adj = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]), (num_nodes, num_nodes))

    # 计算归一化的拉普拉斯矩阵
    row_sum = torch.sparse.sum(adj, -1).to_dense() + 1e-7
    r_inv_sqrt = torch.pow(row_sum, -0.5)
    rows_inv_sqrt = r_inv_sqrt[indices[0]]
    cols_inv_sqrt = r_inv_sqrt[indices[1]]
    values = rows_inv_sqrt * cols_inv_sqrt

    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))
