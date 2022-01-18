from argparse import ArgumentParser
import torch
from tqdm import tqdm
from time import time
import numpy as np
from torch.utils.data import DataLoader

from src.model import SupervisedGraphSage
from src.datamodule import NodeDataset
from src.trainer import GCN_creditTrainer
from src.utils import load_cora

parser = ArgumentParser()
#data parameters
#parser.add_argument('--dataset_path', type=str, default='.', help='dataset path')

#model parameters
parser.add_argument('--num_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--num_sample', type=int, default=5, help='number of neighbor samples')
parser.add_argument('--embed_dim', type=int, default=128, help='embedding size')
parser.add_argument('--num_classes', type=int, default=2, help='number of user types for classification')


#trainer parameters
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--optim', type=str, default='adam', help='optimizer option')
parser.add_argument('--load_path', type=str, default=None, help='load model path')
parser.add_argument('--save_path', type=str, required=True, help='save model path')


args = parser.parse_args()
args = vars(args)


def main():
    global args

    #dataloader
    features, labels, adj_lists, (n_nodes, n_features) = load_cora()
    train_dataset = NodeDataset(n_nodes, labels)
    # test_dataset =

    print('building dataloader...')
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=2, shuffle=True)
    # test_dataloader =

    print('-' * 50)

    #model
    print('building model...')
    gcn_credit_model = SupervisedGraphSage(**args, features=features, adj_lists=adj_lists, n_nodes=n_nodes, n_features=n_features)
    print(gcn_credit_model)
    print('-' * 50)

    #trainer
    #train
    print('start training')
    trainer = GCN_creditTrainer(gcn_credit_model, **args)
    log = 'epoch: {} - avg_loss: {} - duration: {}'
    for epoch in range(1, args['num_epochs'] + 1):
        train_loss = 0
        start_time = time()
        for batch in tqdm(train_dataloader):
            loss = trainer.update(batch)
            train_loss += loss
        avg_loss = train_loss / len(train_dataset) * args['batch_size']
        duration = time() - start_time
        print(log.format(epoch, avg_loss, duration))
    trainer.save(args['save_path'])
    #evaluate

if __name__ == '__main__':
    main()