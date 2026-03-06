

import torch
from torch_geometric.data import Data
import pandas as pd 
import re
import json
import argparse
import logging

import dgl


def bump(g):
    return Data.from_dict(g.__dict__)
def load_amazon_data(dataset_name, use_text=True, use_dgl=False, seed=0):
    if dataset_name == 'amazon-computers':
        data_path = '../../datasets/Amazon-Computers/Computers_Final_with_BoW_embeddings.pt'
        file_path = "../../datasets/Amazon-Computers/Computers.csv"

    elif dataset_name == 'amazon-photo':
        data_path = '../../datasets/Amazon-Photo/Photo_Final_with_BoW_embeddings.pt'
        file_path = "../../datasets/Amazon-Photo/Photo_Final.csv"

    elif dataset_name == 'amazon-history':
        data_path = '../../datasets/Amazon-History/History_Final_with_BoW_embeddings.pt'
        file_path = "../../datasets/Amazon-History/History_Final.csv"
    
    elif dataset_name == 'amazon-children':
        data_path = '../../datasets/Amazon-Children/Amazon-Books-Children.pt'
        file_path = "../../datasets/Amazon-Children/Children_Final.csv"

    elif dataset_name == 'amazon-sports':
        # data_path = '../../datasets/Amazon-Fitness/Sports_Final_with_BoW_embeddings.pt'
        # file_path = "../../datasets/Amazon-Fitness/Sports_Final.csv"
        data_path = '../../datasets/Amazon-Fitness/Sports_Final_with_BoW_embeddings.pt'
        file_path = "../../datasets/Amazon-Fitness/Sports_Final.csv"

    else:
        assert False, "no such amazon dataset"

    if dataset_name in ['amazon-history', 'amazon-photo', 'amazon-computers']:
        data = torch.load(data_path)
        data = bump(data)

        if isinstance(data.x, torch.LongTensor):
            data.x = data.x.float()  # special preprocess for amazon-photo-BOW

        if use_text:
            df = pd.read_csv(file_path)
            text = list(df["text"])
            data.raw_texts = text
            category_list = list(df["category"])

            label_list = df['label']
            label_to_category = dict(zip(label_list, category_list))
            # 获取所有类别
            category_names = [label_to_category[label] for label in sorted(label_to_category.keys())]
            data.label_texts = category_names

    else:
        df = pd.read_csv(file_path)
        category_list = list(df['category'])

        text_list = list(df['text'])
        label_list = df['label']
        neighbour_list = df['neighbour']

        neighbor = []
        for row in neighbour_list:
            neighbor_indices = re.findall(r'\d+', row)  
            neighbor_indices = [int(idx) for idx in neighbor_indices]  
            neighbor.append(neighbor_indices)

    
        # 创建标签到类别的映射字典
        label_to_category = dict(zip(label_list, category_list))
        
        # 获取所有类别
        category_names = [label_to_category[label] for label in sorted(label_to_category.keys())]
        x = None
        y = torch.tensor(label_list.values, dtype=torch.float)
        edge_index_list = []
        for index, neighbours in enumerate(neighbour_list):
            # 将邻居列表转换为整数
            neighbours = neighbours.strip('[]').split(',')
            neighbours = [int(neighbour.strip()) for neighbour in neighbours if neighbour.strip()]
            # 创建边索引对
            edges = [(index, neighbour) for neighbour in neighbours]
            if (index,index) not in edges:
                edges.append((index,index))
            edge_index_list.extend(edges)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        num_nodes = len(text_list)
        data = Data(x=x, edge_index=edge_index, y=y, category_names = category_names, num_nodes = num_nodes)
        data.neighbour = neighbor
        if use_text:
            text = list(df["text"])
            data.raw_texts = text_list
            data.label_texts = list(set(category_list))
            category_list = list(df["category"])

            label_list = df['label']
            label_to_category = dict(zip(label_list, category_list))
            # 获取所有类别
            category_names = [label_to_category[label] for label in sorted(label_to_category.keys())]
            data.label_texts = category_names
            

    idx_list = list(range(data.num_nodes))
    print(f'data.num_nodes: {data.num_nodes}')
    # if setting == 'sup':
    #     train_idx, test_idx = train_test_split(idx_list, test_size=0.2, random_state=seed)
    #     train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=seed)
    # elif setting == 'semi':
    #     train_idx, val_idx, test_idx = split_data_k(y=data.y)
    # print(f'train num: {len(train_idx)}')
    # print(f'val num: {len(val_idx)}')
    # print(f'test num: {len(test_idx)}')
    # data.train_id = train_idx
    # data.val_id = val_idx
    # data.test_id = test_idx

    
    if use_dgl:
        g = dgl.DGLGraph()
        edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])
        g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y).squeeze()
        g.temp_edge_index = data.edge_index 
        if use_text:
            g.text = text 
        return g
    data.LMemb = None
    return data


def main():
    parser = argparse.ArgumentParser(description='Generate multi-hop features.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    args = parser.parse_args()

    data = torch.load(f'../../datasets/{args.dataset}/processed_data.pt')
    print(data.label_texts)
    if args.dataset == 'amazon-sports':
        data.label_texts = ['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing']
    elif args.dataset == 'amazon-computers':
        data.label_texts = ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts']
    elif args.dataset == 'amazon-photo':
        data.label_texts = ['Video Surveillance', 'Accessories', 'Binoculars & Scopes', 'Video', 'Lighting & Studio', 'Bags & Cases', 'Tripods & Monopods', 'Flashes', 'Digital Cameras', 'Film Photography', 'Lenses', 'Underwater Photography']
    torch.save(data,f'../../datasets/{args.dataset}/processed_data.pt')
    data = torch.load(f'../../datasets/{args.dataset}/processed_data.pt')
    print(data.label_texts)
    
    # pdb.set_trace()

main()