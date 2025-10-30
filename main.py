import math
import os
import time
import json
import argparse
import pickle
import random


# import implicit, scipy.sparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import scipy.sparse as sp
import scipy

from torch._C import device
from tqdm import tqdm
from sklearn.decomposition import NMF
import torch
from torch import nn, cosine_similarity
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import collate_fn, load_data
from model import HR_HGDN
from dataloader import HGDNDataset

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_nmf',    action='store_true', help='run NMF baseline')
parser.add_argument('--baseline_als',  action='store_true', help='run SVD++ baseline')
parser.add_argument('--use_social',    action='store_true', help='whether to use the true social graph')
parser.add_argument('--random_social', action='store_true', help='replace social graph with random graph of same density')
parser.add_argument('--dataset', default='Epinions', help='dataset: Ciao/Epinions')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=128, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1234, help='the number of random seed to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--device', type=int, default=0, help='the index of GPU device (-1 for CPU)')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--item_cl_weight', type=float, default=1.2, help='the weight of item_cl_loss')
parser.add_argument('--cl_weight', type=float, default=0.1, help='the weight of cl loss')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
parser.add_argument('--num_layers', type=int, default=2, help='the number of layers of GNN')
parser.add_argument('--eps', type=float, default=0.10, help='the random noise')
# parser.add_argument('--gat_weight', type=float, default=0.4, help='the gat weight')
args = parser.parse_args()
print(args)
print(torch.cuda.is_available)

device = (torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}'))

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

here = os.path.dirname(os.path.abspath(__file__))
fn = 'weights/' + args.dataset

if not os.path.exists(fn):
    os.mkdir(fn)


def main():

    train_set, valid_set, test_set, u_items_list, u_users_list, u_users_items_list, i_users_list, i_items_list, i_items_users_list, user_count, item_count, rate_count, time_count = load_data(
        args.dataset)

    train_data = HGDNDataset(train_set, u_items_list, u_users_list, i_users_list, i_items_list, user_count, item_count)
    valid_data = HGDNDataset(valid_set, u_items_list, u_users_list, i_users_list, i_items_list, user_count, item_count)
    test_data = HGDNDataset(test_set, u_items_list, u_users_list, i_users_list, i_items_list, user_count, item_count)


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. NMF 基线 —— 改用 Surprise 自带的 NMF（mask-only 版）
    if args.baseline_nmf:
        class TorchMF(nn.Module):
            def __init__(self, n_users, n_items, emb_dim):
                super().__init__()
                self.mu = nn.Parameter(torch.tensor(
                    [torch.mean(torch.tensor([r for (_, _, r, _) in train_set + valid_set], dtype=torch.float32))]))
                self.bu = nn.Embedding(n_users, 1)
                self.bi = nn.Embedding(n_items, 1)
                self.P = nn.Embedding(n_users, emb_dim)
                self.Q = nn.Embedding(n_items, emb_dim)

            def forward(self, uids, iids):
                p = self.P(uids)  # [B, d]
                q = self.Q(iids)  # [B, d]
                bu = self.bu(uids).squeeze()  # [B]
                bi = self.bi(iids).squeeze()  # [B]
                mu = self.mu  # scaler
                return mu + bu + bi + (p * q).sum(dim=1)

        mf = TorchMF(user_count + 1, item_count + 1, args.embed_dim).to(device)
        opt = optim.Adam(mf.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        loss_f = nn.MSELoss()
        # 训练
        for epoch in range(args.epoch):
            mf.train()
            for uids, iids, ratings, *_ in train_loader:
                uids, iids, ratings = uids.to(device), iids.to(device), ratings.to(device)
                pred = mf(uids, iids)
                loss = loss_f(pred, ratings)
                opt.zero_grad();
                loss.backward();
                opt.step()
            # 每轮打印一下
            with torch.no_grad():
                mf.eval()
                tot, cnt = 0.0, 0
                for uids, iids, ratings, *_ in valid_loader:
                    uids, iids, ratings = uids.to(device), iids.to(device), ratings.to(device)
                    pr = mf(uids, iids)
                    tot += ((pr - ratings) ** 2).sum().item();
                    cnt += ratings.size(0)
                print(f"[MF Epoch {epoch + 1}] Val RMSE: {math.sqrt(tot / cnt):.4f}")
        # 最后测试集
        mf.eval()
        tot_mae, tot_mse, cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for uids, iids, ratings, *_ in test_loader:
                uids, iids, ratings = uids.to(device), iids.to(device), ratings.to(device)
                pr = mf(uids, iids)
                tot_mae += torch.abs(pr - ratings).sum().item()
                tot_mse += ((pr - ratings) ** 2).sum().item()
                cnt += ratings.size(0)
        print(f"[MF final] MAE={tot_mae / cnt:.4f}, RMSE={(tot_mse / cnt) ** 0.5:.4f}")
        return

    if args.baseline_als:
        # 1) 构造 CSR 评分矩阵 R，shape=(n_users, n_items)
        R = build_rating_matrix(train_set + valid_set, user_count + 1, item_count + 1)
        R_csr = sp.csr_matrix(R, dtype=np.float32)

        # 2) 计算用户相似度矩阵（余弦）
        #    为了节省内存，只算非零行之间的相似度，输出 shape=(n_users, n_users)
        user_sim = sk_cosine(R_csr.toarray())

        # 3) 对测试集做预测
        K = 20  # 邻居数，你也可以调
        users, items, true_r, _ = zip(*test_set)
        users = np.array(users, dtype=int)
        items = np.array(items, dtype=int)
        preds = np.zeros_like(users, dtype=np.float32)

        for idx, (u, i) in enumerate(zip(users, items)):
            sim_u = user_sim[u]  # shape=(n_users,)
            # 选出与 u 相似度最高的 K 个 v ≠ u
            sim_u[u] = 0
            neigh = np.argpartition(-sim_u, K)[:K]
            sims = sim_u[neigh]  # shape=(K,)
            # 取这些邻居在物品 i 上的评分（可能有 0）
            ratings = R[neigh, i]  # shape=(K,)
            mask = ratings > 0
            if np.any(mask):
                preds[idx] = np.dot(sims[mask], ratings[mask]) / (sims[mask].sum() + 1e-8)
            else:
                # 没邻居评分就退回到 u 的全局平均
                nz = R[u].nonzero()[0]
                preds[idx] = R[u, nz].mean() if len(nz) else R_csr.data.mean()

        # 4) 裁剪到评分范围，评估
        all_r = np.array([r for (_, _, r, _) in train_set + valid_set])
        min_r, max_r = all_r.min(), all_r.max()
        preds = np.clip(preds, min_r, max_r)

        mae = np.mean(np.abs(preds - true_r))
        rmse = np.sqrt(np.mean((preds - true_r) ** 2))
        print(f"[User‐CF K={K}] MAE={mae:.4f}, RMSE={rmse:.4f}")
        return

    model = HR_HGDN(num_users=user_count + 1, num_items=item_count + 1, num_rate_levels=rate_count + 1,
                    emb_dim=args.embed_dim, device=args.device,use_social=args.use_social,random_social=args.random_social,gat_weight=args.gat_weight).to(device)

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load(f'{fn}/random_best_checkpoint.pth.tar', map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        avg_mae, avg_rmse, avg_cl_loss = validate(test_loader, model)
        print(f"Test: MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, CL Loss: {avg_cl_loss:.4f}")
        return

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_valid_loss = float('inf')
    best_epoch = 0
    valid_loss_list = []
    test_loss_list = []

    best_mae = float('inf')
    best_rmse = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.epoch)):
    # for epoch in range(args.epoch):
        train_for_epoch(train_loader, model, optimizer, epoch, args.epoch, criterion)
        avg_mae, avg_rmse, avg_cl_loss, cos_avg = validate(valid_loader, model)
        valid_loss_list.append([avg_mae, avg_rmse, avg_cl_loss,cos_avg])
        test_mae, test_rmse, test_cl_loss, cos_avg = validate(test_loader, model)
        test_loss_list.append([test_mae, test_rmse, test_cl_loss,cos_avg])
        print(f"[EPOCH {epoch + 1}] Val MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, "
          f"CL Loss={avg_cl_loss:.4f}, CosAvg={cos_avg:.4f}")
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        current_valid_loss = avg_rmse

        if avg_rmse < best_rmse or avg_mae < best_mae:
            best_rmse = avg_rmse
            best_mae = avg_mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'{fn}/random_best_checkpoint.pth.tar')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

        print(
            f'Epoch {epoch} validation: MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, CL Loss: {avg_cl_loss:.4f}, Best MAE: {best_mae:.4f}, test_MAE: {test_mae:.4f}, test_RMSE: {test_rmse:.4f}, test_CL Loss: {test_cl_loss:.4f}')

# class TorchSVDpp(nn.Module):
#     def __init__(self, num_users, num_items, emb_dim, u_items_list, device):
#         super().__init__()
#         self.device = device
#         self.P = nn.Embedding(num_users, emb_dim)   # 用户隐向量 p_u
#         self.Q = nn.Embedding(num_items, emb_dim)   # 物品隐向量 q_i
#         self.Y = nn.Embedding(num_items, emb_dim)   # 隐式反馈物品向量 y_j
#         self.u_items_list = u_items_list            # 每个用户历史交互物品列表
#
#     def forward(self, uids, iids):
#         # 基本用户向量 p_u 和物品向量 q_i
#         p_u = self.P(uids)           # [B, d]
#         q_i = self.Q(iids)           # [B, d]
#
#         # 隐式反馈：对每个 batch 中的用户，累加他们历史交互的 Y 向量
#         imp = []
#         # for u in uids.tolist():
#         #     hist = self.u_items_list[u]          # list of item ids
#         #     if len(hist)>0:
#         #         yj = self.Y(torch.LongTensor(hist).to(self.device)).sum(0)
#         #         yj = yj / math.sqrt(len(hist))
#         #     else:
#         #         yj = torch.zeros(self.Q.embedding_dim, device=self.device)
#         #     imp.append(yj)
#
#         for u in uids.tolist():
#             hist = self.u_items_list[u]
#             item_ids = [triple[1] for triple in hist]
#             if len(hist) > 0:
#                 yj = self.Y(torch.LongTensor(item_ids).to(self.device))  # [|hist|, d]
#
#                 yj = yj.sum(dim=0) / math.sqrt(len(item_ids))  # [d]
#
#             else:
#                 # 一定要在 else 也 append 一个同样维度的零向量
#                 yj = torch.zeros(self.Y.embedding_dim, device=self.device)
#             imp.append(yj)  # ← 统一在这里 append，无论 if/else
#
#             # 现在 imp 是一个长度 = batch_size 的列表，里面都是 [d] 向量
#
#
#         imp = torch.stack(imp, dim=0)  # <== 这步报错
#
#         usr_hat = p_u + imp  # p_u [batch_size, d] + imp [batch_size, d]
#         preds = (usr_hat * q_i).sum(dim=1)
#         return preds

def mean_pairwise_cosine(H: torch.Tensor) -> float:
    """
    H: [num_users, dim] 用户embedding矩阵
    return: 平均两两余弦相似度 (除去对角线)
    """
    Hn = torch.nn.functional.normalize(H, dim=1)  # 先单位化
    sim_matrix = Hn @ Hn.T                        # [num_users, num_users]
    n = H.size(0)
    # (总和 - 对角线) / (n*(n-1))
    return (sim_matrix.sum() - n) / (n * (n - 1))

def build_rating_matrix(triples, num_users, num_items):
    """用四元组列表构造 NumPy 用户×物品评分矩阵，未交互处为 0."""
    R = np.zeros((num_users, num_items), dtype=np.float32)
    for u, i, r, _ in triples:
        R[u, i] = r
    return R

def build_triplets(triples):
    """把四元组列表拆成 (users, items, ratings)。"""
    users  = np.array([u for u,i,r,t in triples], dtype=int)
    items  = np.array([i for u,i,r,t in triples], dtype=int)
    ratings= np.array([r for u,i,r,t in triples], dtype=np.float32)
    return users, items, ratings
def train_for_epoch(train_loader, model, optimizer, epoch, num_epochs, criterion):
    model.train()
    sum_epoch_loss = 0
    sum_cl_loss = 0
    sum_mae = 0
    sum_rmse = 0

    for i, (uids, iids, ratings, tids, u_item_pad, u_user_pad, i_user_pad, i_item_pad, soc_edge_index,
            inter_adj_matrix) in enumerate(
        train_loader):
        uids = uids.to(device)
        iids = iids.to(device)
        ratings = ratings.to(device)
        u_item_pad = u_item_pad.to(device)
        i_user_pad = i_user_pad.to(device)
        soc_edge_index = soc_edge_index.to(device)
        inter_adj_matrix = inter_adj_matrix.to(device)

        optimizer.zero_grad()
        preds, final_user_embedding, final_item_embedding, last_user_emb, last_item_emb = model(uids, iids,
                                                                                                u_item_pad,
                                                                                                i_user_pad,
                                                                                                soc_edge_index,
                                                                                                inter_adj_matrix,
                                                                                                perturbed=True,
                                                                                                num_layers=args.num_layers,
                                                                                                eps=args.eps,
																								gat_weight=args.gat_weight)

        rec_loss = criterion(preds, ratings)
        cl_loss = model.lightgcn.contrastive_loss(final_user_embedding, last_user_emb, final_item_embedding,
                                                  last_item_emb, args.item_cl_weight)

        total_loss = rec_loss + cl_loss * args.cl_weight
        total_loss.backward()
        optimizer.step()

        mae = torch.mean(torch.abs(preds - ratings))
        rmse = torch.sqrt(torch.mean((preds - ratings) ** 2))

        sum_epoch_loss += rec_loss.item()
        sum_cl_loss += cl_loss.item()
        sum_mae += mae.item()
        sum_rmse += rmse.item()

        if i % 100 == 0:
            mean_loss = sum_epoch_loss / (i + 1)
            mean_cl_loss = sum_cl_loss / (i + 1)
            mean_mae = sum_mae / (i + 1)
            mean_rmse = sum_rmse / (i + 1)
            print(
                f'[TRAIN] Epoch {epoch + 1}/{num_epochs}, Batch {i}, Loss: {total_loss.item():.4f}, CL Loss: {cl_loss.item():.4f}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}, Avg Loss: {mean_loss:.4f}, Avg CL Loss: {mean_cl_loss:.4f}, Avg MAE: {mean_mae:.4f}, Avg RMSE: {mean_rmse:.4f}')


# def validate(valid_loader, model):
#     model.eval()
#     sum_mae = 0
#     sum_rmse = 0
#     sum_cl_loss = 0
#     criterion = nn.MSELoss()
#
#     with torch.no_grad():
#         for uids, iids, ratings, tids, u_item_pad, u_user_pad, i_user_pad, i_item_pad, soc_edge_index, inter_adj_matrix in valid_loader:
#             uids = uids.to(device)
#             iids = iids.to(device)
#             ratings = ratings.to(device)
#             u_item_pad = u_item_pad.to(device)
#             i_user_pad = i_user_pad.to(device)
#             soc_edge_index = soc_edge_index.to(device)
#             inter_adj_matrix = inter_adj_matrix.to(device)
#
#             preds, final_user_embedding, final_item_embedding, last_user_emb, last_item_emb = model(uids, iids,
#                                                                                                     u_item_pad,
#                                                                                                     i_user_pad,
#                                                                                                     soc_edge_index,
#                                                                                                     inter_adj_matrix,
#                                                                                                     perturbed=True,
#                                                                                                     num_layers=args.num_layers,
#                                                                                                     eps=args.eps,
# 																									gat_weight=args.gat_weight)
#
#             rec_loss = criterion(preds, ratings)
#             cl_loss = model.lightgcn.contrastive_loss(final_user_embedding, last_user_emb, final_item_embedding,
#                                                       last_item_emb, args.item_cl_weight)
#             sum_cl_loss += cl_loss.item()
#
#             mae = torch.mean(torch.abs(preds - ratings))
#             rmse = torch.sqrt(torch.mean((preds - ratings) ** 2))
#             sum_mae += mae.item()
#             sum_rmse += rmse.item()
#
#     avg_mae = sum_mae / len(valid_loader)
#     avg_rmse = sum_rmse / len(valid_loader)
#     avg_cl_loss = sum_cl_loss / len(valid_loader)
#     return avg_mae, avg_rmse, avg_cl_loss
def validate(valid_loader, model):
    model.eval()
    sum_mae, sum_rmse, sum_cl_loss = 0, 0, 0
    criterion = nn.MSELoss()

    # 收集所有用户embedding
    all_user_embs = []

    with torch.no_grad():
        for uids, iids, ratings, tids, u_item_pad, u_user_pad, i_user_pad, i_item_pad, soc_edge_index, inter_adj_matrix in valid_loader:
            uids = uids.to(device)
            iids = iids.to(device)
            ratings = ratings.to(device)
            u_item_pad = u_item_pad.to(device)
            i_user_pad = i_user_pad.to(device)
            soc_edge_index = soc_edge_index.to(device)
            inter_adj_matrix = inter_adj_matrix.to(device)

            preds, final_user_embedding, final_item_embedding, last_user_emb, last_item_emb = model(
                uids, iids, u_item_pad, i_user_pad, soc_edge_index, inter_adj_matrix,
                perturbed=True, num_layers=args.num_layers, eps=args.eps, gat_weight=args.gat_weight
            )

            rec_loss = criterion(preds, ratings)
            cl_loss = model.lightgcn.contrastive_loss(final_user_embedding, last_user_emb,
                                                      final_item_embedding, last_item_emb,
                                                      args.item_cl_weight)
            sum_cl_loss += cl_loss.item()

            mae = torch.mean(torch.abs(preds - ratings))
            rmse = torch.sqrt(torch.mean((preds - ratings) ** 2))
            sum_mae += mae.item()
            sum_rmse += rmse.item()

            # 收集用户embedding
            all_user_embs.append(final_user_embedding.detach().cpu())

    avg_mae = sum_mae / len(valid_loader)
    avg_rmse = sum_rmse / len(valid_loader)
    avg_cl_loss = sum_cl_loss / len(valid_loader)

    # 拼接用户embedding，并计算全局平均相似度
    all_user_embs = torch.cat(all_user_embs, dim=0)  # shape [num_samples, d]
    cos_avg = mean_pairwise_cosine(all_user_embs)

    return avg_mae, avg_rmse, avg_cl_loss, cos_avg

if __name__ == '__main__':

    main()
