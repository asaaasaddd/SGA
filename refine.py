import copy
import torch
from torch import optim
from tqdm import tqdm
from model import *
import numpy as np
from scipy.sparse import csr_matrix
from utils import *
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
import torch.nn.functional as F

def train_model(model, s_adj, t_adj, optimizer, config):
    model.train()

    source_outputs = model(s_adj, 's')
    target_outputs = model(t_adj, 't')

    positive_pairs, negative_pairs = candidate_choose(source_outputs, target_outputs, config.theta)

    for epoch in tqdm(range(config.train_epochs),desc="Training Epochs", unit="epoch"):
        optimizer.zero_grad()
        source_outputs = model(s_adj, 's')
        target_outputs = model(t_adj, 't')
        loss = compute_contrastive_loss(source_outputs, target_outputs, positive_pairs, negative_pairs, 1)
        loss.backward()
        optimizer.step()
    print("Training completed.")

    # for epoch in range(config.train_epochs):
    #     optimizer.zero_grad()
    #     source_outputs = model(s_adj, 's')
    #     target_outputs = model(t_adj, 't')
    #     loss = compute_contrastive_loss(source_outputs, target_outputs, positive_pairs, negative_pairs, 1)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch+1}: Loss={loss.item()}")


# def candidate_choose(source_outputs, target_outputs, theta=0.99, delta_theta=0.2, max_neg_samples=4):
#     Sf = torch.zeros((len(source_outputs[0]), len(target_outputs[0])), dtype=torch.float32, device=source_outputs[0].device)
#     for i in range(len(source_outputs)):
#         S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
#         Sf += S
#     min_value = Sf.min(dim=1, keepdim=True).values
#     max_value = Sf.max(dim=1, keepdim=True).values
#     normalized_Sf = (Sf - min_value) / (max_value - min_value + 1e-8)  # 加一个小数防止除零
#
#     positive_pairs_f = []  # 存储正样本对 (i, j)
#     negative_pairs_f = []  # 存储负样本对 (i, j)
#
#     for i in range(normalized_Sf.shape[0]):
#         row_similarities = normalized_Sf[i]
#         positive_mask = row_similarities > theta
#         positive_indices = torch.where(positive_mask)[0].tolist()
#         row_positive_pairs = [(i, j) for j in positive_indices]
#
#         if not row_positive_pairs:
#             continue
#
#         negative_mask = row_similarities < (theta - delta_theta)
#         negative_indices = torch.where(negative_mask)[0].tolist()
#         row_negative_pairs = [(i, j) for j in negative_indices[:max_neg_samples]]
#
#         positive_pairs_f.extend(row_positive_pairs)
#         negative_pairs_f.extend(row_negative_pairs)
#
#     return positive_pairs_f, negative_pairs_f

def compute_contrastive_loss(source_outputs, target_outputs, positive_pairs, negative_pairs, margin=1):
    """
    使用对比损失（Contrastive Loss）计算正负样本对的距离。

    :param source_outputs: 源图的所有层嵌入输出，列表形式
    :param target_outputs: 目标图的所有层嵌入输出，列表形式
    :param positive_pairs: 正样本对列表，包含（源节点，目标节点）对
    :param negative_pairs: 负样本对列表，包含（源节点，目标节点）对
    :param margin: Margin 参数，推远负样本对的距离
    :return: 计算出的平均对比损失
    """
    source_embeddings = source_outputs[-1]
    target_embeddings = target_outputs[-1]
    total_loss = torch.tensor(0.0, device=source_embeddings.device, requires_grad=True)

    for src_idx, tgt_idx in positive_pairs:
        pos_distance = torch.norm(source_embeddings[src_idx] - target_embeddings[tgt_idx], p=2)
        total_loss = total_loss + pos_distance ** 2  
        
    for src_idx, tgt_idx in negative_pairs:
        neg_distance = torch.norm(source_embeddings[src_idx] - target_embeddings[tgt_idx], p=2)
        total_loss = total_loss + torch.clamp(margin - neg_distance, min=0) ** 2  # 负样本的距离越大越好
    num_pairs = len(positive_pairs) + len(negative_pairs)
    average_loss = total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=source_embeddings.device, requires_grad=True)
    return average_loss


def candidate_choose(source_outputs, target_outputs, theta=0.992, delta_theta=0.2, max_neg_samples=5):
    Sf = torch.zeros((len(source_outputs[0]), len(target_outputs[0])), dtype=torch.float32,
                     device=source_outputs[0].device)
    num_layers = len(source_outputs)

    for i in range(num_layers):
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        Sf += S

    Sf /= num_layers
    normalized_Sf = Sf
    positive_pairs_f = [] 
    negative_pairs_f = []  

    for i in range(normalized_Sf.shape[0]):
        row_similarities = normalized_Sf[i]
        positive_mask = row_similarities >= (theta + 0.009)
        positive_indices = torch.where(positive_mask)[0].tolist()
        row_positive_pairs = [(i, j) for j in positive_indices]
        if not row_positive_pairs:
            continue
        negative_mask = row_similarities < (theta - delta_theta)
        negative_indices = torch.where(negative_mask)[0].tolist()
        row_negative_pairs = [(i, j) for j in negative_indices[:max_neg_samples]]
        positive_pairs_f.extend(row_positive_pairs)
        negative_pairs_f.extend(row_negative_pairs)
    return positive_pairs_f, negative_pairs_f


def create_negative_pairs_dict(negative_pairs):
    negative_pairs_dict = {}
    for src_idx, tgt_idx in negative_pairs:
        if src_idx not in negative_pairs_dict:
            negative_pairs_dict[src_idx] = []
        negative_pairs_dict[src_idx].append(tgt_idx)

    return negative_pairs_dict

def compute_infoNCE_loss(source_outputs, target_outputs, positive_pairs, negative_pairs_dict, temperature=0.07):
    """
    使用 InfoNCE 损失计算正负样本对的相似性，针对每个源节点选择相应的负样本对。
    :param source_outputs: 源图的所有层嵌入输出，列表形式
    :param target_outputs: 目标图的所有层嵌入输出，列表形式
    :param positive_pairs: 正样本对列表，包含（源节点，目标节点）对
    :param negative_pairs_dict: 负样本对字典，键是源节点索引，值是与之相关的负样本对（目标节点索引）列表
    :param temperature: 温度参数，控制softmax的平滑度
    :return: 计算出的平均InfoNCE损失
    """
    source_embeddings = source_outputs[-1]
    target_embeddings = target_outputs[-1]

    total_loss = 0.0

    for src_idx, tgt_idx in positive_pairs:
        pos_similarity = F.cosine_similarity(source_embeddings[src_idx], target_embeddings[tgt_idx],
                                             dim=0) / temperature
        if src_idx in negative_pairs_dict:
            negative_targets = negative_pairs_dict[src_idx]
        else:
            continue
        if len(negative_targets) == 0:
            continue
        neg_similarities = []
        for neg_tgt_idx in negative_targets:
            neg_similarity = F.cosine_similarity(source_embeddings[src_idx], target_embeddings[neg_tgt_idx],
                                                 dim=0) / temperature
            neg_similarities.append(neg_similarity)
        neg_similarities = torch.stack(neg_similarities)
        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities])
        labels = torch.tensor([0], dtype=torch.long)
        logits = logits.unsqueeze(0)
        loss = F.cross_entropy(logits, labels)  # logits: [1, num_classes], labels: [1]
        total_loss += loss

    average_loss = total_loss / len(positive_pairs) if len(positive_pairs) > 0 else 0.0

    return average_loss


def log(iteration, acc, MAP, top1, top10):
    print("iteration is: {:d}, Accuracy: {:.4f}, MAP: {:.4f}, Precision_1: {:.4f}, Precision_10: {:.4f}".format(iteration + 1, acc, MAP, top1, top10))


def refine_alignment(model, s_sadj, t_sadj, args, s_sadj_ori, t_sadj_ori):
    s_sadj = s_sadj.to_dense()
    t_sadj = t_sadj.to_dense()
    s_sadj_ori = s_sadj_ori.to_dense()
    t_sadj_ori = t_sadj_ori.to_dense()
    CloAlign_S, groundtruth = refine(model, s_sadj, t_sadj, args.theta, args, s_sadj_ori, t_sadj_ori)

    return CloAlign_S, groundtruth

# def add_edges_based_on_candidates(source_A_hat, target_A_hat, seed_list1, seed_list2, s_edges_list, t_edges_list):
#     """
#     根据 seed_list1 和 seed_list2 中的种子节点对直接在 source_A_hat 和 target_A_hat 中添加边。
#
#     :param source_A_hat: 源图的邻接矩阵（原始邻接矩阵，稠密格式）
#     :param target_A_hat: 目标图的邻接矩阵（原始邻接矩阵，稠密格式）
#     :param seed_list1: 源图中的种子节点列表
#     :param seed_list2: 目标图中的种子节点列表
#     :return: 原始邻接矩阵（含补齐边）、归一化后的邻接矩阵
#     """
#     added_edges_source = 0
#     added_edges_target = 0
#
#     for i in range(len(seed_list1)):
#         for j in range(i + 1, len(seed_list1)):
#             src_i, src_j = seed_list1[i].item(), seed_list1[j].item()
#             tgt_i, tgt_j = seed_list2[i].item(), seed_list2[j].item()
#
#             # 在源图的邻接矩阵中添加边
#             if source_A_hat[src_i, src_j] == 0 and target_A_hat[tgt_i, tgt_j] != 0:
#                 source_A_hat[src_i, src_j] = 1
#                 source_A_hat[src_j, src_i] = 1  # 无向图，矩阵对称
#                 added_edges_source += 1
#
#             # 在目标图的邻接矩阵中添加边
#             if target_A_hat[tgt_i, tgt_j] == 0 and source_A_hat[src_i, src_j] != 0:
#                 target_A_hat[tgt_i, tgt_j] = 1
#                 target_A_hat[tgt_j, tgt_i] = 1  # 无向图，矩阵对称
#                 added_edges_target += 1
#
#     source_A_hat_norm = normalize(source_A_hat)
#     target_A_hat_norm = normalize(target_A_hat)
#
#     source_A_hat_norm = torch.tensor(source_A_hat_norm, dtype=torch.float32)
#     target_A_hat_norm = torch.tensor(target_A_hat_norm, dtype=torch.float32)
#
#     return source_A_hat, target_A_hat, source_A_hat_norm, target_A_hat_norm, added_edges_source, added_edges_target

# def add_edges_based_on_candidates(source_A_hat, target_A_hat, seed_list1, seed_list2, s_edges_list, t_edges_list):
#     """
#     根据 seed_list1 和 seed_list2 中的种子节点对直接在 source_A_hat 和 target_A_hat 中添加边，
#     同时通过 s_edges_list 和 t_edges_list 遍历边的结构差异，进行补边操作。
#
#     :param source_A_hat: 源图的邻接矩阵（原始邻接矩阵，稠密格式）
#     :param target_A_hat: 目标图的邻接矩阵（原始邻接矩阵，稠密格式）
#     :param seed_list1: 源图中的种子节点列表
#     :param seed_list2: 目标图中的种子节点列表
#     :param s_edges_list: 源图中的边列表，每一项为 (src_i, src_j) 的元组
#     :param t_edges_list: 目标图中的边列表，每一项为 (tgt_i, tgt_j) 的元组
#     :return: 原始邻接矩阵（含补齐边）、归一化后的邻接矩阵、新边添加到边列表、补齐边数量
#     """
#     added_edges_source = 0
#     added_edges_target = 0
#
#     # 将 seed_list1 和 seed_list2 转为字典，方便查找对应的目标节点
#     seed_dict_source_to_target = {seed_list1[i].item(): seed_list2[i].item() for i in range(len(seed_list1))}
#     seed_dict_target_to_source = {seed_list2[i].item(): seed_list1[i].item() for i in range(len(seed_list2))}
#
#     # 遍历源图的边列表，检查结构差异
#     for src_i, src_j in s_edges_list:
#         src_i, src_j = int(src_i), int(src_j)
#         if src_i in seed_dict_source_to_target and src_j in seed_dict_source_to_target:
#             # 获取目标图中的对应节点对
#             tgt_i = seed_dict_source_to_target[src_i]
#             tgt_j = seed_dict_source_to_target[src_j]
#
#             # 检查目标图中对应的边是否存在
#             if target_A_hat[tgt_i, tgt_j] == 0:
#                 # 若不存在，则在目标图中补上对应的边
#                 target_A_hat[tgt_i, tgt_j] = 1
#                 target_A_hat[tgt_j, tgt_i] = 1  # 无向图，矩阵对称
#                 added_edges_target += 1
#
#     # 遍历目标图的边列表，检查结构差异
#     for tgt_i, tgt_j in t_edges_list:
#         tgt_i, tgt_j = int(tgt_i), int(tgt_j)
#         if tgt_i in seed_dict_target_to_source and tgt_j in seed_dict_target_to_source:
#             # 获取源图中的对应节点对
#             src_i = seed_dict_target_to_source[tgt_i]
#             src_j = seed_dict_target_to_source[tgt_j]
#
#             # 检查源图中对应的边是否存在
#             if source_A_hat[src_i, src_j] == 0:
#                 # 若不存在，则在源图中补上对应的边
#                 source_A_hat[src_i, src_j] = 1
#                 source_A_hat[src_j, src_i] = 1  # 无向图，矩阵对称
#                 added_edges_source += 1
#
#     # 归一化操作
#     source_A_hat_norm = normalize(source_A_hat)
#     target_A_hat_norm = normalize(target_A_hat)
#
#     source_A_hat_norm = torch.tensor(source_A_hat_norm, dtype=torch.float32)
#     target_A_hat_norm = torch.tensor(target_A_hat_norm, dtype=torch.float32)
#
#     return source_A_hat, target_A_hat, source_A_hat_norm, target_A_hat_norm, added_edges_source, added_edges_target

def get_neighbors(adj_matrix, node):
    return set(torch.nonzero(adj_matrix[node]).view(-1).tolist())

def add_edges_based_on_candidates(source_A_hat, target_A_hat, seed_list1, seed_list2, s_edges_list, t_edges_list):
    """
    根据 seed_list1 和 seed_list2 中的种子节点对直接在 source_A_hat 和 target_A_hat 中添加边，
    同时通过 s_edges_list 和 t_edges_list 遍历边的结构差异，进行补边操作。

    :param source_A_hat: 源图的邻接矩阵（原始邻接矩阵，稠密格式）
    :param target_A_hat: 目标图的邻接矩阵（原始邻接矩阵，稠密格式）
    :param seed_list1: 源图中的种子节点列表
    :param seed_list2: 目标图中的种子节点列表
    :param s_edges_list: 源图中的边列表，每一项为 (src_i, src_j) 的元组
    :param t_edges_list: 目标图中的边列表，每一项为 (tgt_i, tgt_j) 的元组
    :param threshold_u: 补边的 Jaccard 相似度阈值
    :return: 原始邻接矩阵（含补齐边）、归一化后的邻接矩阵、新边添加到边列表、补齐边数量
    """
    added_edges_source = 0
    added_edges_target = 0
    threshold_u = 0.7
    output_file_path = './data/res/added_edges.txt'
    seed_dict_source_to_target = {seed_list1[i].item(): seed_list2[i].item() for i in range(len(seed_list1))}
    seed_dict_target_to_source = {seed_list2[i].item(): seed_list1[i].item() for i in range(len(seed_list2))}
    with open(output_file_path, 'a') as f:
        for src_i, src_j in s_edges_list:
            src_i, src_j = int(src_i), int(src_j)
            if src_i in seed_dict_source_to_target and src_j in seed_dict_source_to_target:
                tgt_i = seed_dict_source_to_target[src_i]
                tgt_j = seed_dict_source_to_target[src_j]

                if target_A_hat[tgt_i, tgt_j] == 0:
                    #search the candidates and compute the neiborhood consistency
                    src_neighbors_union = get_neighbors(source_A_hat, src_i) | get_neighbors(source_A_hat, src_j)
                    tgt_neighbors_union = {seed_dict_source_to_target.get(n) for n in src_neighbors_union if n in seed_dict_source_to_target}
                    tgt_neighbors_target = get_neighbors(target_A_hat, tgt_i) | get_neighbors(target_A_hat, tgt_j)
                    intersection = len(tgt_neighbors_union & tgt_neighbors_target)
                    union = len(tgt_neighbors_union | tgt_neighbors_target)
                    jaccard_similarity = intersection / union if union != 0 else 0
                    if jaccard_similarity >= threshold_u:
                        target_A_hat[tgt_i, tgt_j] = 1
                        target_A_hat[tgt_j, tgt_i] = 1
                        added_edges_target += 1
                        f.write(f"{tgt_i} {tgt_j} in target\n")

        for tgt_i, tgt_j in t_edges_list:
            tgt_i, tgt_j = int(tgt_i), int(tgt_j)
            if tgt_i in seed_dict_target_to_source and tgt_j in seed_dict_target_to_source:
                src_i = seed_dict_target_to_source[tgt_i]
                src_j = seed_dict_target_to_source[tgt_j]
                if source_A_hat[src_i, src_j] == 0:
                    tgt_neighbors_union = get_neighbors(target_A_hat, tgt_i) | get_neighbors(target_A_hat, tgt_j)
                    src_neighbors_union = {seed_dict_target_to_source.get(n) for n in tgt_neighbors_union if n in seed_dict_target_to_source}
                    src_neighbors_source = get_neighbors(source_A_hat, src_i) | get_neighbors(source_A_hat, src_j)
                    intersection = len(src_neighbors_union & src_neighbors_source)
                    union = len(src_neighbors_union | src_neighbors_source)
                    jaccard_similarity = intersection / union if union != 0 else 0
                    if jaccard_similarity >= threshold_u:
                        source_A_hat[src_i, src_j] = 1
                        source_A_hat[src_j, src_i] = 1
                        added_edges_source += 1
                        f.write(f"{src_i} {src_j} in source\n")

    source_A_hat_norm = normalize(source_A_hat)
    target_A_hat_norm = normalize(target_A_hat)

    source_A_hat_norm = torch.tensor(source_A_hat_norm, dtype=torch.float32)
    target_A_hat_norm = torch.tensor(target_A_hat_norm, dtype=torch.float32)

    return source_A_hat, target_A_hat, source_A_hat_norm, target_A_hat_norm, added_edges_source, added_edges_target

def refine(model, source_A_hat, target_A_hat, threshold, args, s_sadj_ori, t_sadj_ori):
    refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), False)
    S_max = None
    source_outputs = model(refinement_model(source_A_hat, 's'), 's')
    target_outputs = model(refinement_model(target_A_hat, 't'), 't')
    s_struct_edges = np.genfromtxt('./data/' + args.dataset + '/' + args.dataset + '_s_edge.txt',dtype=np.int32).tolist()
    t_struct_edges = np.genfromtxt('./data/' + args.dataset + '/' + args.dataset + '_t_edge.txt',dtype=np.int32).tolist()
    # 有监督读取训练集 ground_truth = np.genfromtxt('./data/douban/gt_dict/node,split=0.8.test.dict', dtype=np.int32)
    ground_truth = np.genfromtxt('./data/' + args.dataset + '/' + args.dataset + '_ground_True.txt', dtype=np.int32)

    ground_truth = list(ground_truth)
    full_di = dict(ground_truth)
    if args.dataset in ['douban', 'allmv_tmdb']:
        full_dic = dict([val, key] for key, val in full_di.items())
    elif args.dataset in ['ppi', 'flickr_myspace', 'citeseer', 'flickr', 'dblp', 'acm_dblp', 'nell', 'bgs_new',
                          'elliptic', 'fb_tw','fq_tw']:
        full_dic = full_di

    acc, S = get_acc(source_outputs, target_outputs, full_dic, args.alphas, just_S=True)
    score = np.max(S, axis=1).mean()

    alpha_source_max = None
    alpha_target_max = None
    if 1:
        refinement_model.score_max = score
        alpha_source_max = refinement_model.alpha_source
        alpha_target_max = refinement_model.alpha_target
        S_max = S
        acc_max = 0
        MAP_max = 0
        top1_max = 0
        top10_max = 0
        total_added_edges_source = 0
        total_added_edges_target = 0

    source_candidates, target_candidates = [], []
    temp_s_candidates, temp_t_candidates = [], []
    alpha_source_max = refinement_model.alpha_source + 0
    alpha_target_max = refinement_model.alpha_target + 0
    score = []

    for iteration in range(args.r_epochs):
        source_candidates, target_candidates, len_source_candidates, count_true_candidates = topo_refine(
            source_outputs, target_outputs, threshold, full_dic, source_A_hat, target_A_hat)
        if len(temp_s_candidates) == len(source_candidates) and iteration % 15 == 0:
            break
        if len(temp_s_candidates) != len(source_candidates):
            temp_s_candidates = source_candidates
            temp_t_candidates = target_candidates
            source_A_hat, target_A_hat, source_A_hat_norm, target_A_hat_norm, added_edges_source, added_edges_target = add_edges_based_on_candidates(
                s_sadj_ori, t_sadj_ori, temp_s_candidates, temp_t_candidates, s_struct_edges, t_struct_edges
            )
            total_added_edges_source += added_edges_source
            total_added_edges_target += added_edges_target

        refinement_model.alpha_source[source_candidates] *= args.reweigh_factor[0]
        refinement_model.alpha_target[target_candidates] *= args.reweigh_factor[0]

        source_outputs = model(refinement_model(source_A_hat_norm, 's'), 's')
        target_outputs = model(refinement_model(target_A_hat_norm, 't'), 't')

        acc, S = get_acc(source_outputs, target_outputs, full_dic, args.alphas, just_S=True)
        top = [1, 10]
        acc, MAP, top1, top10 = get_statistics(S, full_dic, top, use_greedy_match=False, get_all_metric=True)
        log(iteration, acc, MAP, top1, top10)
        score.append(np.max(S, axis=1).mean())

    print(f"Num candidates: {len_source_candidates}, true candidates: {count_true_candidates}")
    print("Done refinement!")

    return S_max, full_dic

def get_acc(source_outputs, target_outputs, test_dict=None, alphas=None, just_S=False):
    global acc, MAP, top1, top10
    Sf = np.zeros((len(source_outputs[0]), len(target_outputs[0])))
    accs = ""
    for i in range(0, len(source_outputs)):
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        S_numpy = S.detach().cpu().numpy()
        if test_dict is not None:
            if not just_S:
                acc = get_statistics(S_numpy, test_dict)
                accs += "Acc layer {} is: {:.4f}, ".format(i, acc)
        if alphas is not None:
            Sf += alphas[i] * S_numpy
        else:
            Sf += S_numpy
    if test_dict is not None:
        if not just_S:
            accs += "Final acc is: {:.4f}".format(acc)
    return accs, Sf


def get_statistics(alignment_matrix, groundtruth, top, groundtruth_matrix=None,  use_greedy_match=False,
                   get_all_metric=False):
    if use_greedy_match:
        print("This is greedy match accuracy")
        pred = greedy_match(alignment_matrix)
    else:
        pred = get_nn_alignment_matrix(alignment_matrix)
    acc = compute_accuracy(pred, groundtruth)
    if get_all_metric:
        MAP, Hit, AUC = compute_MAP_Hit_AUC(alignment_matrix, groundtruth)
        pred_top_1 = top_k(alignment_matrix, top[0])
        top1 = compute_precision_k(pred_top_1, groundtruth)
        pred_top_10 = top_k(alignment_matrix, top[1])
        top10 = compute_precision_k(pred_top_10, groundtruth)
        return acc, MAP, top1, top10
    return acc


def compute_accuracy(pred, gt):
    matched_pairs = []
    if type(gt) == dict:
        for key, value in gt.items():
            if pred[key, value] == 1:
                matched_pairs.append((key, value))
    else:
        for i in range(pred.shape[0]):
            if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
                matched_pairs.append((i, np.argmax(gt[i])))

    # 保存匹配对到文件
    # with open('./data/res/matched_pairs_01.txt', 'w') as file:
    #     for pair in matched_pairs:
    #         file.write(f"{pair[0]} {pair[1]}\n")
    accuracy = len(matched_pairs) / len(gt) if type(gt) == dict else len(matched_pairs) / (gt == 1).sum()
    return accuracy


def compute_precision_k(top_k_matrix, gt):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            try:
                if top_k_matrix[key, value] == 1:
                    n_matched += 1
            except:
                n_matched += 1
        return n_matched / len(gt)

    gt_candidates = np.argmax(gt, axis=1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1

    n_nodes = (gt == 1).sum()
    return n_matched / n_nodes


def greedy_match(S):
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m, n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while (matched <= min_size):
        ipos = ix[index - 1]
        jc = int(np.ceil(ipos / m))
        ic = ipos - (jc - 1) * m
        if ic == 0: ic = 1
        if (used_rows[ic - 1] == 0 and used_cols[jc - 1] == 0):
            row[matched - 1] = ic - 1
            col[matched - 1] = jc - 1
            max_list[matched - 1] = x[index - 1]
            used_rows[ic - 1] = 1
            used_cols[jc - 1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result


def get_nn_alignment_matrix(alignment_matrix):
    # Sparse
    row = np.arange(len(alignment_matrix))
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))]
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    return result


def compute_MAP_Hit_AUC(alignment_matrix, gt):
    MAP = 0
    AUC = 0
    Hit = 0
    for key, value in gt.items():
        ele_key = alignment_matrix[key].argsort()[::-1]
        for i in range(len(ele_key)):
            if ele_key[i] == value:
                ra = i + 1  # r1
                MAP += 1 / ra
                Hit += (alignment_matrix.shape[1] + 1) / alignment_matrix.shape[1]
                AUC += (alignment_matrix.shape[1] - ra) / (alignment_matrix.shape[1] - 1)
                break
    n_nodes = len(gt)
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, Hit, AUC


def top_k(S, k=1):
    top = np.argsort(-S)[:, :k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx, elm] = 1
    return result

def topo_refine(source_outputs, target_outputs, threshold, full_dict, s_sadj, t_sadj):
    List_S = get_similarity_matrices(source_outputs, target_outputs)[1:]
    source_candidates = []
    target_candidates = []
    count_true_candidates = 0
    num_source_nodes = s_sadj.shape[0]
    num_target_nodes = t_sadj.shape[0]
    for i in range(min(num_source_nodes, num_target_nodes)):
        node_i_is_stable = True
        for j in range(len(List_S)):
            if List_S[j][i].argmax() != List_S[j - 1][i].argmax() or List_S[j][i].max() < (threshold-0.11):
                node_i_is_stable = False
        if node_i_is_stable:
            tg_candi = List_S[-1][i].argmax()
            source_candidates.append(i)
            target_candidates.append(tg_candi)
            try:
                if full_dict == None:
                    continue
                elif full_dict[i] == tg_candi:
                    count_true_candidates += 1
            except:
                 continue
    return (torch.LongTensor(source_candidates), torch.LongTensor(target_candidates), len(
        source_candidates), count_true_candidates)


def get_t_s_matrix(s_matrix, t_matrix, k):
    s_matrix = np.array(s_matrix.cpu().detach())
    t_matrix = np.array(t_matrix.cpu().detach())
    return

def get_similarity_matrices(source_outputs, target_outputs):
    """
    Construct Similarity matrix in each layer
    :params source_outputs: List of embedding at each layer of source graph
    :params target_outputs: List of embedding at each layer of target graph
    """
    list_S = []
    for i in range(len(source_outputs)):
        source_output_i = source_outputs[i]
        target_output_i = target_outputs[i]
        S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
        #S = get_cos_matrix(source_output_i.cpu().detach().numpy(), target_output_i.cpu().detach().numpy())
        #S = torch.from_numpy(S).cuda()
        list_S.append(S)
    return list_S


def get_cos_matrix(source_output, target_output):
    return cosine_similarity(source_output, target_output)


def save_embeddings(source_outputs, target_outputs, path_prefix):
    """
            Save the last layer embeddings of source and target graphs.

            Parameters:
            - source_outputs: list of tensors containing source graph embeddings from each layer
            - target_outputs: list of tensors containing target graph embeddings from each layer
            - path_prefix: string prefix for the file paths where embeddings will be saved
            """
    if not isinstance(source_outputs, list) or not isinstance(target_outputs, list):
        raise TypeError("source_outputs and target_outputs should be lists of tensors.")

    if len(source_outputs) == 0 or len(target_outputs) == 0:
        raise ValueError("source_outputs and target_outputs should contain at least one tensor.")

    source_last_layer = source_outputs[-1]
    target_last_layer = target_outputs[-1]

    source_embeddings = source_last_layer.detach().cpu().numpy()
    target_embeddings = target_last_layer.detach().cpu().numpy()

    np.save(f"{path_prefix}/source_embeddings.npy", source_embeddings)
    np.save(f"{path_prefix}/target_embeddings.npy", target_embeddings)

    print(f"Embeddings saved as {path_prefix}/source_embeddings.npy and {path_prefix}/target_embeddings.npy")
