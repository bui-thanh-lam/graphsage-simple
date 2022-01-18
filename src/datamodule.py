from torch.utils.data import Dataset


class NodeDataset(Dataset):
    def __init__(self, n_nodes, labels):
        self.node_ids = list(range(0, n_nodes))
        self.labels = labels

    def __getitem__(self, index):
        return (self.node_ids[index], self.labels[index])

    def __len__(self):
        return len(self.node_ids)