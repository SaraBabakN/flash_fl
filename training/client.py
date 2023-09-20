import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from core import Masking
from utiles import apply_mask
from models.model_options import get_model
from constants import *


def to_dict(my_dict, device):
    for key in my_dict:
        my_dict[key] = my_dict[key].to(device)


class Client:
    def __init__(self, args, train_loader, test_loader, id):
        self.id = id
        self.args = args
        self.freeze = False
        self.predefined_mask = None
        self.train_criterion = F.cross_entropy
        self.client_density = self.args.density
        self.total_num_batch_size = len(train_loader)
        self.trainloader, self.testloader = train_loader, test_loader
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    def get_optimizer(self, model):
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr)
        return optimizer

    def get_parameters(self, model):
        model.train()
        model.cpu()
        weights = deepcopy(model.state_dict())
        return weights

    def set_masks(self, masks):
        self.predefined_mask = masks

    def get_mask(self, model, optimizer):
        T_max = len(self.trainloader) * (self.args.local_epoch)
        mask = Masking(optimizer, self.args.prune_rate, T_max, conv_only=self.args.conv_only, remove_first_last=self.args.remove_first_last)
        mask.add_module(model, density=self.client_density, init_masks=self.predefined_mask)
        return mask

    def init_model(self, weights):
        model = get_model(self.args)
        model.load_state_dict(weights, strict=True)
        model.to(self.device)
        model.train()
        return model

    def update(self, weights):
        model = self.init_model(deepcopy(weights))
        optimizer = self.get_optimizer(model)

        if self.args.hetero_client:
            to_dict(self.predefined_mask, self.device)
            apply_mask(model, self.predefined_mask)

        if self.args.experiment_type == DENSE:
            assert self.predefined_mask is None
            self.update_model_mask(model, optimizer, self.args.local_epoch, mask=None)

        elif self.args.experiment_type == PDST or self.freeze:
            assert self.predefined_mask is not None
            to_dict(self.predefined_mask, self.device)
            self.update_model_mask(model, optimizer, self.args.local_epoch, mask=None)
            to_dict(self.predefined_mask, 'cpu')

        else:
            to_dict(self.predefined_mask, self.device)
            mask = self.get_mask(model, optimizer)
            self.update_model_mask(model, optimizer, self.args.local_epoch, mask=mask)
            self.predefined_mask = deepcopy(mask.masks)
            to_dict(self.predefined_mask, 'cpu')

        current_weight = self.get_parameters(model)
        self.client_density = self.args.density
        return current_weight, deepcopy(self.predefined_mask)

    def update_model_mask(self, model, optimizer, total_epochs, mask):
        for epoch in range(1, total_epochs + 1):
            self.train(epoch, model, optimizer, mask)
            if mask is not None and epoch < total_epochs:
                mask.at_end_of_epoch()

    def train(self, epoch, model, optimizer, mask):
        model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            ce_loss = self.train_criterion(output, target)
            ce_loss.backward()
            if mask is not None:
                mask.step()
                if self.args.local_epoch == 1 and batch_idx == self.total_num_batch_size // 2:
                    mask.at_end_of_epoch()
            else:
                optimizer.step()
                if self.predefined_mask is not None:
                    model = apply_mask(model, self.predefined_mask)
