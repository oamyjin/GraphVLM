import random
from collections import defaultdict
import random
import torch
import json

import copy
import numpy as np
from tqdm import trange
from sentence_transformers import SentenceTransformer
DEFAULT_GRAPH_PAD_ID = -500
def get_data_split(dataset,setting='sup'):
    np_filename = f'/scratch/jl11523/GraphGPT/dataset/mgllm/{dataset}/{dataset}_split.json'
    with open(np_filename, 'r') as file:
        loaded_data_dict = json.load(file)
    # Convert the numpy arrays or non-Python int types to standard Python lists of int
    train_ids = loaded_data_dict['train']
    val_ids = loaded_data_dict['val']
    
    data_path = f'/scratch/jl11523/GraphGPT/dataset/mgllm/{dataset}/{dataset}Graph_cliptext.pt'
    data = torch.load(data_path)
    node_list = [i for i in range(len(data.y))]
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_ids = [node for node in node_list if node not in train_set and node not in val_set]
    
    print(f"Loaded data from {np_filename}: train_id length = {len(train_ids)}, test_id length = {len(test_ids)}, val_id length = {len(val_ids)}")
    
    return {'train': train_ids, 'test': test_ids, 'val': val_ids}


def generate_edge_list(data):
    # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    row, col = data.edge_index
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()

    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    # torch.save(edge_list, os.path.join(data_dir, "edge_list.pt"))
    return edge_list
#FIXME
def get_fix_shape_subgraph_sequence_new(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = []
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and  avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence


def generate_jsonl_data(dataset, k_hop=2, k_shot=3, sample_size=10,with_example=True, with_label_embbedding=False):
    data_path = f'/scratch/jl11523/GraphGPT/dataset/mgllm/{dataset}/{dataset}Graph_cliptext.pt'
    data = torch.load(data_path)
    
    label_texts = data["category_label_mapping"]
    print(f"num nodes: {len(data.y)}")
    print(f'{dataset} label list: {label_texts}')
    data_y = data.y
    edge_list = generate_edge_list(data)
    
    split_dict = get_data_split(dataset)
    num_classes = len(torch.unique(data_y))
    print(torch.unique(data_y).tolist())
    print(f'num_classes: {num_classes}')

    categories = list(label_texts.values())
    print("categories", categories)
    candidates_str = '\n'.join(categories)
    
    for set in ["train","test"]:

        node_ids = split_dict[set]
        #label_dict = defaultdict(list)
        # for node_index, label_index in enumerate(data.y):
        #     if node_index in node_ids:  # 仅在 node_index 存在于 train_ids 中时添加
        #         label_dict[label_index.item()].append(node_index)  # 使用 .item() 以确保索引为整数

        # 将 defaultdict 转换为普通字典
        # label_dict = dict(label_dict)
        output_file = f'/scratch/jl11523/projects/LLaGA/dataset/{dataset}/sampled_2_10_{set}.jsonl'
        with open(output_file, 'w') as f:
            for node_id in node_ids:
                # 获取图结构
                graph = get_fix_shape_subgraph_sequence_new(edge_list, node_id, k_hop, sample_size)
                label = label_texts[int(data_y[node_id])]
                label_index = int(data_y[node_id])
                # title = data.text[node_id]
                # V1
                # network_type = "a co-purchase network in an e-commerce" if dataset != "Reddit" else "an interaction network in a social media"
                # target_type = "product" if dataset != "Reddit" else "post"
                # prompt = (
                #     f"Given a graph representing {network_type} platform: <graph>, where each node represents a {target_type} and the 0th node is the target node, we need to classify the taget node into {num_classes} classes: {candidates_str}.\n"
                #     "Please predict which category the target node belongs to."
                # )
                # UPDATED to be consistent with original LLaGA prompt (no '0th node is the taregt node', ...)
                if dataset == "Reddit":
                    node_type = "posts published on Reddit"
                    edge_type = "posts indicate interactions through shared user comments"
                else:
                    node_type = "products sold in Amazon"
                    edge_type = "products indicate they are purchased together"
                prompt = (
                    f"Given a node-centered graph: <graph>, where nodes represents {node_type}, and edges between {edge_type}. We need to classify the center node into {num_classes} classes: {candidates_str}.\n"
                    "Please tell me which class the center node belongs to?"
                )              

                conversations = [
                    {
                        "from": "human",
                        "value": prompt
                    },
                    {
                        "from": "gpt",
                        "value": f"{label}"
                    }
                ]
                # 构建数据项
                data_item = {
                    "id": node_id,
                    "graph": graph,
                    'class': label_index,
                    "conversations": conversations,
                    "dataset": dataset
                }
                f.write(json.dumps(data_item) + '\n')

def get_sbert_embedding(texts, device):
    sbert_model = SentenceTransformer('/scratch/jl11523/projects/local_model/all-MiniLM-L6-v2', device=device)
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

def main():
    datasets = ['CD']#["Movies", "Toys", "Grocery", "Reddit", "Arts"]
    # Iterate through each dataset and setting combination
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        k_hop = 2
        sample_size = 10
        k_shot = 3
        
        # Call the generate_jsonl_data function
        generate_jsonl_data(dataset, k_hop, k_shot, sample_size,with_example=True, with_label_embbedding=False)
        print(f"Completed dataset: {dataset}")

if __name__ == "__main__":
    main()