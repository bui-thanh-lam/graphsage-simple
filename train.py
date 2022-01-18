from argparse import ArgumentParser
import torch

from src.model import GCN_creditModel
from src.datamodule import SupervisedGraphSage, collate_fn
from src.trainer import GCN_creditTrainer

from tqdm import tqdm
from time import time
import numpy as np

parser = ArgumentParser()
#data parameters
parser.add_argument('--dataset_path', type=str, default='.', help='dataset path')

#model parameters
parser.add_argument('--num_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--num_sample', type=int, default=5, help='number of neighbor samples')
parser.add_argument('--embed_dim', type=int, default=128, help='embedding size')
parser.add_argument('--num_classes', type=int, default=2, help='number of user types for classification')


#trainer parameters
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--load_path', type=str, default=None, help='load model path')
parser.add_argument('--save_path', type=str, required=True, help='save model path')


args = parser.parse_args()
args = vars(args)


def main():
    global args
    #dataloader
    # train_dataset =
    # test_dataset =
    # num_feats =
    args['num_feats'] = num_feats
    print('building dataloader...')
    # train_dataloader =
    # test_dataloader =
    print('-' * 50)

    #model
    print('building model...')
    gcn_credit_model = SupervisedGraphSage(**args)
    print('-' * 50)

    #trainer
    #train
    print('start training')
    trainer = GCN_creditTrainer(gcn_credit_model, **args)
    log = 'epoch: {} - avg_loss: {} - duration: {}'
    for epoch in enumerate(1, args['num_epochs'] + 1):
        train_loss = 0
        start_time = time()
        for batch_idx, batch in tqdm(enumerate(train_dataloader)):
            loss = trainer.predict(batch)
            train_loss += loss
        avg_loss = train_loss / len(train_dataset) * args['train_batch_size']
        duration = time() - start_time
        print(log.format(epoch, avg_loss, duration))
    trainer.save(args['save_path'])
    #evaluate

if __name__ == '__main__':
    main()