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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

parser = ArgumentParser()
#data parameters
#parser.add_argument('--dataset_path', type=str, default='.', help='dataset path')

#model parameters
parser.add_argument('--num_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--num_sample', type=int, default=5, help='number of neighbor samples')
parser.add_argument('--embed_dim', type=int, default=128, help='embedding size')
parser.add_argument('--num_classes', type=int, default=1, help='number of user types for classification')


#trainer parameters
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='')
parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
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
    
    train_indices, test_indices = train_test_split([(i, label) for i, label in enumerate(labels)], test_size=0.2, random_state=2022)

    train_indices = [index for index, label in train_indices]
    test_indices = [index for index, label in test_indices]
    
    train_dataset = NodeDataset(train_indices, labels)
    test_dataset = NodeDataset(test_indices, labels)

    print('building dataloader...')
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=2, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], num_workers=2)

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
    print('start evaluate...')
    probs = []
    outputs = []
    labels = []
    for batch in tqdm(train_dataloader):
        batch_probs, batch_outputs, batch_labels = trainer.predict(batch, infer=False)
        batch_probs = batch_probs.cpu().detach().numpy().tolist()
        batch_labels = batch_labels.cpu().detach().numpy().tolist()
        
        probs.extend(batch_probs)
        outputs.extend(batch_outputs)
        labels.extend(batch_labels)
    #print(probs)
    #print(outputs)
    #implement f1, acc, auc metric
    cf_mat = confusion_matrix(labels, outputs, labels= [0, 1])
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, outputs)

    torch.save(probs, 'outputs.pt')

    tp = cf_mat[0][0]
    fn = cf_mat[0][1]
    fp = cf_mat[1][0]
    tn = cf_mat[1][1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = precision * recall * 2 / (precision + recall)
    print('Confusion matrix:\n',cf_mat)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)
    print('auc: ', auc)
    print('accuracy: ', acc)

if __name__ == '__main__':
    main()