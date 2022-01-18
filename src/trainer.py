import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import src.utils as utils

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class GCN_creditTrainer(Trainer):
    def __init__(self, model, **args):
        self.args = args
        self.model = model
        if self.args['load_path'] is not None:
            self.load(args['load_path'])
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.args['cuda']:
            #model params, loss, inputs, labels must be on the same device
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'])

    def update(self, batch):
        inputs, labels = batch
        labels = labels.long()
        if self.args['cuda']:
            #model params, loss, inputs, labels must be on the same device
            labels = labels.cuda()

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, batch, infer=True):
        if infer:
            inputs = batch
            if self.args['cuda']:
                inputs = inputs.cuda()

            self.model.eval()
            outputs = self.model(inputs)
            return outputs
        else:
            inputs, labels = batch
            if self.args['cuda']:
                # model params, loss, inputs, labels must be on the same device
                inputs = inputs.cuda()
                labels = labels.cuda()

            self.optimizer.eval()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            return loss.item()

