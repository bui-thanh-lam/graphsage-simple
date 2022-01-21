from torch.utils.data import Dataset


class NodeDataset(Dataset):
    def __init__(self, indices, labels):
        self.node_ids = indices
        self.labels = labels

    def __getitem__(self, index):
        return (self.node_ids[index], self.labels[self.node_ids[index]])

    def __len__(self):
        return len(self.node_ids)