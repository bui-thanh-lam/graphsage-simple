import torch
import numpy as np

def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2) # use default lr
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def load_cora(embedding_path="data/cora/preprocess_data.content", graph_path="data/cora/vnd_sample.cites", device="cpu", return_shape=True):
    """
    Load data in CORA format.

    Args:
        embedding_path  (str)
        graph_path      (str)
        device          (str)

    Returns:
        features    (nn.Embedding [V * M]): node embedding
        labels      (np array shape [V * 1]): node label
        adj_lists   (set shape [V * ...]): adjacency list represent the graph
        shape       (tuple size 2): n_nodes & n_features
    """
    _features = list()
    labels = list()
    node_map = dict()
    label_map = {"bad": 1, "good": 0}
    with open(embedding_path) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            _features.append([float(e) for e in info[1:-1]])
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels.append(label_map[info[-1]])
    _features = np.asarray(_features, dtype=np.float32)
    n_nodes, n_features = _features.shape
    features = torch.nn.Embedding(n_nodes, n_features)
    features.weight = torch.nn.Parameter(torch.FloatTensor(_features), requires_grad=False)
    if device == "cuda": features = features.cuda()
    labels = np.asarray(labels, dtype=np.float32)

    adj_lists = [set() for _ in range(n_nodes)]
    with open(graph_path) as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            try:
                node_1 = node_map[info[0]]
                node_2 = node_map[info[1]]
                adj_lists[node_1].add(node_2)
                adj_lists[node_2].add(node_1)
            except:
                pass
    if return_shape: return features, labels, adj_lists, (n_nodes, n_features)
    else: return features, labels, adj_lists
    