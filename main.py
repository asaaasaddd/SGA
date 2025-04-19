import itertools
from torch import optim
import json
import test
from config import Config
import argparse
from refine import *
import os
import time
from model import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from time import time
from datetime import datetime
import time
#2025.04.19

def get_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', "--dataset", type=str, default="douban")
    parse.add_argument('-k', "--gcn_block", type=int, default=3)
    parse.add_argument('-out', "--output_dim", type=int, default="128")
    parse.add_argument("--alphas", type=list, default=[1, 1, 1, 1])
    parse.add_argument("--reweigh_factor", type=list, default=[1.1])
    return parse.parse_args()


def load_ground_truth(file_path):
    source_candidates = []
    target_candidates = []
    with open(file_path, 'r') as file:
        for line in file:
            tgt, src = line.strip().split()
            source_candidates.append(int(src))
            target_candidates.append(int(tgt))
    return torch.tensor(source_candidates), torch.tensor(target_candidates)


def run():
    start_time = time.time()  # 开始时间
    args = get_parse()
    config_file = './config/' + args.dataset + '.ini'
    config = Config(config_file)
    cuda = not config.no_cuda and torch.cuda.is_available()
    use_seed = not config.no_seed
    args.r_epochs = config.r_epochs
    args.theta = config.theta

    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    print('Dataset Loading')
    config.name = 's'
    config.n = config.n1
    s_sadj, s_sadj_ori = load_graph(config)
    s_feature = load_data(config)
    config.name = 't'
    config.n = config.n2
    t_sadj, t_sadj_ori = load_graph(config)
    t_feature = load_data(config)
    print('Dataset end')
    print('Dataset: ' + args.dataset)

    model = SGAlign(args.gcn_block, args.output_dim, s_feature, t_feature, 'tanh', config)

    optimizer = optim.Adam(model.parameters(), config.lr)
    train_model(model, s_sadj, t_sadj, optimizer, config)  # 运用对比损失函数训练 GCN

    refine_alignment(model, s_sadj, t_sadj, args, s_sadj_ori, t_sadj_ori)

    end_time = time.time()  # 结束时间
    print(f"Execution time: {end_time - start_time} seconds")  # 输出执行时间


if __name__ == '__main__':
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run()
