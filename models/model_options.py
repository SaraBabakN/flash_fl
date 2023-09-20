from models.VGG import VGG
from models.ResNet import ResNet18, ResNet152, ResNet34
from models.general import MNISTNet, leaf_cnn


def get_model(args):
    if args.model_type == 'VGG16':
        global_model = VGG('VGG16', init_weights=True)
    elif args.model_type == 'resnet18':
        global_model = ResNet18(num_classes=args.num_classes)
    elif args.model_type == 'resnet18_tiny':
        global_model = ResNet18(num_classes=args.num_classes, cifar=False)
    elif args.model_type == 'resnet34':
        global_model = ResNet34(num_classes=args.num_classes, cifar=False)
    elif args.model_type == 'resnet152':
        global_model = ResNet152(num_classes=args.num_classes)
    elif args.dataset == 'mnist':
        global_model = MNISTNet()
    elif args.dataset[:len('femnist')] == 'femnist':
        global_model = leaf_cnn(args.num_classes)
    else:
        raise NotImplementedError()
    return global_model
