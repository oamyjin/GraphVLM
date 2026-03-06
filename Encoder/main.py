import argparse
import os
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from nc_dataset import NodeClassificationDataset, NodeClassificationEvaluator
from dgl.nn import GATv2Conv
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from models import SAGE, GCN, MLP

PROJETC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), './')
CONFIG_DIR = os.path.join(PROJETC_DIR, "configs")
log = logging.getLogger(__name__)

class Logger(object):
    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            log.info(f'Run {run + 1:02d}:')
            log.info(f'Highest Valid: {result[:, 0].max():.2f}')
            log.info(f'   Final Test: {result[argmax, 1]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            log.info(f'All runs:')
            r = best_result[:, 0]
            log.info(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            log.info(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


def evaluate(cfg, model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            if cfg.model_name == "MLP":
                x = blocks[-1].dstdata["feat"]
            else:
                x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    pred = torch.argmax(torch.cat(y_hats), dim=1)
    acc = (pred == torch.cat(ys)).sum()/torch.cat(ys).shape[0]
    return acc


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        pred = torch.argmax(pred, dim=1)
        acc = (pred == label).sum()/label.shape[0]
        return acc


def train(cfg, device, g, dataset, model, num_classes, run):
    # create sampler & dataloader
    if not os.path.exists(cfg.checkpoint_folder):
        os.makedirs(cfg.checkpoint_folder)
    checkpoint_path = cfg.checkpoint_folder + cfg.model_name + "_" + cfg.dataset + "_" + "batch_size_" + str(
            cfg.batch_size) + "_n_layers_" + str(cfg.num_layers) + "_hidden_dim_" + str(cfg.hidden_dim) + "_lr_" + str(
            cfg.lr) + "_exclude_degree_" + str(cfg.exclude_target_degree) + "_full_neighbor_" + str(
            cfg.full_neighbor) + "_accu_num_" + str(cfg.accum_iter_number) + "_trail_" + str(run) + "_best.pth"
    train_idx = torch.LongTensor(dataset['train_idx']).to(device)
    val_idx = torch.LongTensor(dataset['val_idx']).to(device)
    if cfg.full_neighbor:
        log.info("We use the full neighbor of the target node to train the models. ")
        sampler = MultiLayerFullNeighborSampler(num_layers=cfg.num_layers, prefetch_node_feats=['feat'])
    else:
        log.info("We sample the neighbor node of the target node to train the models. ")
        sampler = NeighborSampler([cfg.num_of_neighbors] * cfg.num_layers, prefetch_node_feats=['feat'])
    use_uva = cfg.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, 
        step_size=cfg.lr_scheduler_step_size,
        gamma=cfg.lr_scheduler_gamma,
    )

    best_acc = 0
    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss = 0
        valid_result = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            if cfg.model_name == "MLP":
                x = blocks[-1].dstdata["feat"]
            else:
                x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(cfg, model, g, val_dataloader, num_classes)
        log.info(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
        lr_scheduler.step()

        if acc > valid_result:
            valid_result = acc
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),}, checkpoint_path)
            
            # test the model
            log.info("Validation...")
            log.info("Validation Accuracy {:.4f}".format(acc.item()))
            log.info("Testing...")
            acc = layerwise_infer(
                device, g, torch.LongTensor(dataset['test_idx']), model, num_classes, batch_size=1024
            )
            log.info("Test Accuracy {:.4f}".format(acc.item()))
        
        if acc.item() > best_acc:
            best_acc = acc.item()
    
    return best_acc


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    # save configs
    if not os.path.isfile("config.yaml"):
        OmegaConf.save(config=cfg, f=os.path.join("config.yaml"))
    
    print("model_name:", cfg.model_name)

    # load and preprocess dataset
    log.info("Loading data")
    data_path = './dataset/' # replace this with the path where you save the datasets
    dataset_name = cfg.dataset 
    print("dataset_name:", dataset_name)
    feature_path = data_path + dataset_name
    feat_name = cfg.feat 
    verbose = True
    device = torch.device('cpu') #'cpu' # use 'cuda' if GPU is available

    # BOOKS_PATH = os.path.join(data_path, dataset_name) # Jiajin added

    dataset = NodeClassificationDataset(
        root=os.path.join(data_path, dataset_name),
        feat_name=feat_name,
        verbose=verbose,
        device=device
    )

    g = dataset.graph
    labels = dataset.labels

    feat_path = os.path.join(feature_path, f'{cfg.feat}_feat.pt')
    if os.path.exists(feat_path):
        clip_feat = torch.load(feat_path)
        print(f"feat_path: {feat_path} loaded.")
    else:
        print(f"feat_path: {feat_path} does not exist.")
        return

  
    g.ndata['feat'] = clip_feat.to('cpu')
    g.ndata['label'] = labels
    g = dgl.remove_self_loop(g)
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    g = g.to("cuda" if cfg.mode == "puregpu" else "cpu")
    
    labels_pt_data = torch.load(os.path.join(feature_path, "labels-w-missing.pt"))
    num_classes = len(set(labels_pt_data))
    print("num_classes:", num_classes)\
        
    # change device to gpu
    device = torch.device('cpu' if cfg.mode == 'cpu' else 'cuda')
    
    # load dataset splits
    with_zero_splits = torch.load(os.path.join(feature_path, 'split.pt'))
    splits ={}         
    splits['train_idx'] = with_zero_splits['train_idx']
    splits['val_idx'] = with_zero_splits['val_idx']
    splits['test_idx'] = with_zero_splits['test_idx']

    # model initialization
    num_nodes = g.num_nodes()
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    accs = []
    for run in range(cfg.runs):
        log.info("Run {}/{}".format(run + 1, cfg.runs))
        if cfg.model_name == "SAGE":
            model = SAGE(in_size, cfg.hidden_dim, out_size, cfg.num_layers).to(device)
        elif cfg.model_name == "GCN":
            model = GCN(in_size, cfg.hidden_dim, out_size, cfg.num_layers).to(device)
            if cfg.add_self_loop:
                g = dgl.add_self_loop(g)
        elif cfg.model_name == "MLP":
            model = MLP(in_size, cfg.hidden_dim, out_size, 1).to(device) # MLP with only 1 layer         
        # model training
        log.info("Training...")
        
        acc = train(cfg, device, g, splits, model, num_classes, run)
        accs.append(acc)

    # calculate mean and std of the accuracy
    mean_acc = torch.mean(torch.tensor(accs)).item()
    std_acc = torch.std(torch.tensor(accs)).item()
    print("mean_acc:", mean_acc)
    print("std_acc:", std_acc)

    # append results to the file
    new_result = f"{cfg.dataset}, {cfg.model_name}, {cfg.feat}: mean_acc={mean_acc}, std_acc={std_acc}\n"
    with open(cfg.result_file, "a") as file:
        file.write(new_result)

    return mean_acc

if __name__=='__main__':
    main()