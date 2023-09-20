import math
import torch
import numpy as np
from copy import deepcopy


def redistribution(masking, name, weight, mask):
    return torch.abs(weight)[mask.bool()].mean().item()


def prune(masking, mask, weight, name):
    num_remove = math.ceil(masking.name2prune_rate[name] * masking.name2nonzeros[name])
    num_zeros = masking.name2zeros[name]
    k = math.ceil(num_zeros + num_remove)
    if num_remove == 0.0:
        return weight.data != 0.0
    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
    mask.data.view(-1)[idx[:k]] = 0.0
    return mask


def growth(masking, name, new_mask, total_regrowth, weight):
    n = (new_mask == 0).sum().item()
    if n == 0:
        return new_mask
    expeced_growth_probability = (total_regrowth / n)
    new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability
    return new_mask.byte() | new_weights


class LinearDecay:
    """Anneals the pruning rate linearly with each step."""
    def __init__(self, prune_rate, T_max):
        self.decrement = prune_rate / float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate


class Masking:
    def __init__(self, optimizer, prune_rate, T_max, conv_only=True, remove_first_last=False,):
        self.prune_rate_decay = LinearDecay(prune_rate, T_max)
        self.optimizer = optimizer
        self.prune_rate = prune_rate
        self.prune_conv_only = conv_only
        self.remove_first_last = remove_first_last

        self.masks = {}
        self.modules = []
        self.names = []
        self.adjustments = []
        self.total_removed = 0
        self.total_variance = 0
        self.adjusted_growth = 0
        self.baseline_nonzero = None
        self.name2zeros = {}
        self.name2variance = {}
        self.name2nonzeros = {}
        self.name2prune_rate = {}
        self.name2baseline_nonzero = {}

        self.prune_func = prune
        self.growth_func = growth
        self.redistribution_func = redistribution

    def at_end_of_epoch(self):
        self.truncate_weights()

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)

    def add_module(self, module, density, init_masks):
        self.modules.append(module)
        self.baseline_nonzero = 0
        for name in init_masks:
            weight = deepcopy(init_masks[name])
            self.masks[name] = weight
            self.baseline_nonzero += weight.numel() * density
        self.apply_mask()

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data * self.masks[name]

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                if name not in self.name2prune_rate:
                    self.name2prune_rate[name] = self.prune_rate

                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name] / float(self.masks[name].numel())
                if sparsity < 0.2:
                    expected_variance = 1.0 / len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance / actual_variance
                    if expected_vs_actual < 1.0:
                        self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def truncate_weights(self):
        self.gather_statistics()
        self.adjust_prune_rate()

        total_nonzero_new = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                new_mask = self.prune_func(self, mask, weight, name)
                removed = self.name2nonzeros[name] - new_mask.sum().item()
                self.total_removed += removed
                self.masks[name][:] = new_mask
        name2regrowth = self.calc_growth_redistribution()
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                new_mask = self.masks[name].data.byte()
                new_mask = self.growth_func(self, name, new_mask, math.floor(name2regrowth[name]), weight)
                new_nonzero = new_mask.sum().item()
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                total_nonzero_new += new_nonzero
        self.apply_mask()
        self.adjustments.append(self.baseline_nonzero - total_nonzero_new)
        self.adjusted_growth = 0.25 * self.adjusted_growth + (0.75 * (self.baseline_nonzero - total_nonzero_new)) + np.mean(self.adjustments)

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}

        self.total_variance = 0.0
        self.total_removed = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                # redistribution
                self.name2variance[name] = self.redistribution_func(self, name, weight, mask)

                if not np.isnan(self.name2variance[name]):
                    self.total_variance += self.name2variance[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                self.name2zeros[name] / float(self.masks[name].numel())

        for name in self.name2variance:
            if self.total_variance != 0.0:
                self.name2variance[name] /= self.total_variance
            else:
                print('Total variance was zero!')

    def calc_growth_redistribution(self):
        residual = 0
        residual = 9999
        mean_residual = 0
        name2regrowth = {}
        i = 0
        while residual > 0 and i < 1000:
            residual = 0
            for name in self.name2variance:
                prune_rate = self.name2prune_rate[name]
                num_remove = math.ceil(prune_rate * self.name2nonzeros[name])
                num_zero = self.name2zeros[name]
                max_regrowth = num_zero + num_remove

                if name in name2regrowth:
                    regrowth = name2regrowth[name]
                else:
                    regrowth = math.ceil(self.name2variance[name] * (self.total_removed + self.adjusted_growth))
                regrowth += mean_residual

                if regrowth > 0.99 * max_regrowth:
                    name2regrowth[name] = 0.99 * max_regrowth
                    residual += regrowth - name2regrowth[name]
                else:
                    name2regrowth[name] = regrowth
            if len(name2regrowth) == 0:
                mean_residual = 0
            else:
                mean_residual = residual / len(name2regrowth)
            i += 1

        if i == 1000:
            print('Error resolving the residual! Layers are too full! Residual left over: {0}'.format(residual))

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue
        return name2regrowth
