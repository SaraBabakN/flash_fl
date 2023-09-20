import torch
from copy import deepcopy
from utiles import get_density


class Server():
    def __init__(self, test_loader, args):
        self.args = args
        self.test_loader = test_loader

    def aggregate(self, weights, masks, current_weights=None):
        update_mode = self.args.update_mode
        if update_mode == 0:
            return average_weights(weights)
        elif update_mode == 1:
            return masked_average_weights(weights, masks)
        raise NotImplementedError

    def test_inference(self, model):
        model.eval()
        correct = 0.0
        device = torch.device("cuda")
        model.to(device)
        total = len(self.test_loader.dataset)
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                pred_labels = model(images)
                correct += (pred_labels.argmax(1) == labels).type(torch.float).sum().item()
        return total, correct

    def evaluate(self, model, round_idx):
        density = get_density(model.state_dict())
        total, correct = self.test_inference(model)
        print('Density', density, 'Server side Accuracy', 100 * correct / total, "round", round_idx)


def nan_to_zero(a):
    a[a != a] = 0
    return a


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def masked_average_weights(w, masks):
    w_avg = deepcopy(w[0])
    sum_mask = deepcopy(masks[0])
    with torch.no_grad():
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
                if key in sum_mask:
                    sum_mask[key] += masks[i][key]
            if key not in sum_mask:
                w_avg[key] = torch.div(w_avg[key], len(w))
            else:
                w_avg[key] = torch.div(w_avg[key], sum_mask[key])
                w_avg[key] = nan_to_zero(w_avg[key])
    return w_avg
