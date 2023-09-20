import numpy as np
from tqdm import tqdm
from copy import deepcopy

from constants import DENSE, NST, SPDST, JMWST, PDST
from training.aggregator import Server
from models.model_options import get_model
from data_prepration.data import get_partitioned_data
from utiles import get_rand_mask, get_next_round_mask, apply_mask, get_avg_mask
from training.client import Client


density_levels = [0.5, 0.75]
hetero = [0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1, 1, 1, 1]


class System():
    def __init__(self, args):
        self.args = args
        self.model = get_model(self.args)
        self.num_clients = self.args.num_users
        self.mask_key_path = self.args.mask_key_path
        self.num_participants = max(int(args.frac * self.num_clients), 1)
        self.initialize_client()
        self.initialize_mask()

    def get_global_weight(self):
        return deepcopy(self.model.cpu().state_dict())

    def set_global_weight(self, weight):
        self.model.load_state_dict(deepcopy(weight))

    def select_clients(self, num_participant):
        p = np.random.choice(range(self.num_clients), num_participant, replace=False)
        return [self.clients[idx] for idx in p]

    def initialize_client(self):
        test_data, self.clients_train_data, self.clients_test_data, self.client_weight = get_partitioned_data(self.args)
        self.clients = [Client(self.args, self.clients_train_data[i], self.clients_test_data[i], i) for i in range(self.num_clients)]
        self.aggregator = Server(test_data, self.args)

    def initialize_mask(self):
        self.server_mask = None
        exp_type = self.args.experiment_type
        if exp_type == DENSE:
            return
        if exp_type in [NST, PDST]:
            self.server_mask = get_rand_mask(self.args.density, self.mask_key_path)
        elif exp_type in [SPDST, JMWST]:
            clients_mask = self.train_server_mask()
            self.server_mask = get_avg_mask(clients_mask, self.mask_key_path, self.args.density)
            self.model = get_model(self.args)
        self.model = apply_mask(self.model, self.server_mask)
        if self.args.experiment_type in [PDST, NST]:
            [client.set_masks(deepcopy(self.server_mask)) for client in self.clients]

    def train_server_mask(self):
        local_epoch, exp_type = self.args.local_epoch, self.args.experiment_type
        self.args.local_epoch, self.args.experiment_type = self.args.init_epoch, NST
        mask = get_rand_mask(self.args.density, self.mask_key_path)
        clients = self.select_clients(max(int(self.args.init_frac * self.num_clients), 1))
        [client.set_masks(deepcopy(mask)) for client in clients]
        clients_mask = self.train_mask_only(clients)
        self.args.experiment_type, self.args.local_epoch = exp_type, local_epoch
        if exp_type == SPDST:
            self.args.update_mode = 0 if not self.args.hetero_client else self.args.update_mode
            self.args.experiment_type = PDST
        return clients_mask

    def start_federated_learning(self):
        for self.round_idx in tqdm(range(self.args.fl_rounds)):
            clients_mask = self.train_clients()
            if self.args.experiment_type == JMWST and self.round_idx % self.args.jmwst_update_interval == 0:
                self.server_mask = get_next_round_mask(clients_mask, self.get_global_weight(),
                                                       self.args.density, self.mask_key_path,
                                                       self.args.subsample_method)
            if self.args.hetero_client:
                self.hetero_mask = {self.args.density: self.server_mask}
                for d in density_levels:
                    density = d * self.args.density
                    self.hetero_mask[density] = get_next_round_mask(clients_mask, self.get_global_weight(),
                                                                    density, self.mask_key_path,
                                                                    self.args.subsample_method)

    def set_round_param(self, client, i):
        if self.args.hetero_client and self.round_idx > 0:
            client.client_density = hetero[i] * self.args.density
            client.set_masks(deepcopy(self.hetero_mask[client.client_density]))
        elif self.args.experiment_type == JMWST:
            client.set_masks(deepcopy(self.server_mask))
        if self.args.experiment_type == JMWST and self.args.jmwst_update_interval > 1:
            client.freeze = bool(self.round_idx % self.args.jmwst_update_interval)

    def train_clients(self):
        weights, clients_mask = [], []
        participants = self.select_clients(self.num_participants)
        for i, client in enumerate(participants):
            self.set_round_param(client, i)
            w, mask = client.update(self.get_global_weight())
            weights.append(w)
            clients_mask.append(mask)
        new_weights = self.aggregator.aggregate(weights, clients_mask, self.get_global_weight())
        self.set_global_weight(new_weights)
        self.aggregator.evaluate(self.model, self.round_idx)
        self.args.lr = self.args.lr_start * np.exp(self.round_idx / 1000 * self.args.lr_gamma)
        return clients_mask

    def train_mask_only(self, participants):
        clients_mask = []
        for i, client in enumerate(participants):
            _, mask = client.update(self.get_global_weight())
            clients_mask.append(mask)
        return clients_mask
