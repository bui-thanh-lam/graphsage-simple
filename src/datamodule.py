from torch.utils.data import DataLoader, Dataset


class NodeDataset(Dataset):
    def __init__(self, n_nodes):
        self.node_ids = list(range(0, n_nodes))

    def __getitem__(self, index):
        return self.node_ids[index]

    def __len__(self):
        return len(self.node_ids)