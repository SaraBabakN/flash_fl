import torch


def get_avg_mask(masks, mask_path, target_density, device=torch.device('cpu')):
    mask_body = torch.load(mask_path)
    num_clients = len(masks)
    layer_density = {}
    sum_ones = 0
    sum_valid_one = 0
    for name in mask_body:
        density = 0
        numel = torch.numel(masks[0][name])
        sum_valid_one += numel

        for client_mask in masks:
            density += torch.sum(client_mask[name] != 0.0).item() / numel

        density = density / num_clients
        sum_ones += density * numel
        layer_density[name] = density
    ratio = sum_valid_one / sum_ones * target_density
    mask = {}
    for name, shape in mask_body.items():
        density = min(layer_density[name] * ratio, 1)
        mask[name] = (torch.rand(shape, device=device) < density).float().data
    return mask


def get_aggregation_based_sensitivity(weights, mask_path):
    mask_body = torch.load(mask_path)
    layer_density = {}
    total_ones = 0
    total_elem = 0
    for name in mask_body:
        numel = torch.numel(weights[name])
        total_elem += numel
        layer_ones = torch.sum(weights[name] != 0.0).item()
        total_ones += layer_ones
        layer_density[name] = layer_ones
    return total_elem, total_ones, mask_body, layer_density


def get_avg_based_sensivity(clients_mask, mask_path):
    layer_density = {}
    num_clients = len(clients_mask)
    mask_body = torch.load(mask_path)
    total_ones, total_elem = 0, 0
    for name in mask_body:
        numel = torch.numel(clients_mask[0][name])
        total_elem += numel
        density = 0
        for mask in clients_mask:
            density += torch.sum(mask[name] != 0.0).item() / numel
        density = density / num_clients
        total_ones += density * numel
        layer_density[name] = density * numel
    return total_elem, total_ones, mask_body, layer_density


def get_next_round_mask(clients_mask, server_weights, target, mask_path, method):
    if method == 'avg':
        total_elem, total_ones, mask_body, layer_density = get_avg_based_sensivity(clients_mask, mask_path)
    elif method == 'aggregate':
        total_elem, total_ones, mask_body, layer_density = get_aggregation_based_sensitivity(server_weights, mask_path)
    else:
        raise NotImplementedError()

    ratio = (total_elem / total_ones) * target
    ratio = min(1, ratio)
    mask = {}
    for name, shape in mask_body.items():
        num_ones = int(layer_density[name] * ratio)
        mask[name] = torch.zeros(shape)
        _, idx = torch.sort(torch.abs(server_weights[name].view(-1)), descending=True)
        mask[name].view(-1)[idx[:num_ones]] = 1.0
    return mask


def get_rand_mask(density, path, device=torch.device('cpu')):
    mask_body = torch.load(path)
    mask = {}
    for name, shape in mask_body.items():
        mask[name] = (torch.rand(shape, device=device) < density).float().data
    return mask


def apply_mask(model, masks):
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            if name in masks:
                tensor.data = tensor.data * masks[name]
    return model


def get_density(model):
    d = 0
    w = 0
    for _, weight in model.items():
        w += torch.numel(weight)
        d += torch.sum(weight == 0).item()
    return 1 - d / w
