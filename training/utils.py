import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import cv2

from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

    
def load_data(batch_size, 
             in_path = '/mnt/data/ImageNet/ILSVRC2012/',
             in_info_path = '../info/'):
    """
    load 16-class-ImageNet
    :param batch_size: the batch size used in training and test
    :param in_path: the path to ImageNet
    :param in_info_path: the path to the directory 
                              that contains imagenet_class_index.json, wordnet.is_a.txt, words.txt
    :return: train_loader, test_loader
    """

    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid = common_superclass_wnid('geirhos_16')  # 16-class-imagenet
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

    custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py
    
    ### parameters for normalization: choose one of them if you want to use normalization #############
    #normalize = transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])  # norm
    # https://github.com/MadryLab/robustness/blob/master/robustness/datasets.py
    
    # If you want to use normalization parameters of ImageNet from pyrotch:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # norm-in
    
    # 16-class-imagenet
    #normalize = transforms.Normalize(mean=[0.4677, 0.4377, 0.3986], std=[0.2769, 0.2724, 0.2821])  # norm16
    #normalize = transforms.Normalize(mean=[0.4759, 0.4459, 0.4066], std=[0.2768, 0.2723, 0.2827])  # norm16-2
    ############################################################################
    # add normalization 
    custom_dataset.transform_train.transforms.append(normalize)
    custom_dataset.transform_test.transforms.append(normalize)
    
    train_loader, test_loader = custom_dataset.make_loaders(workers=10,
                                                            batch_size=batch_size)

    return train_loader, test_loader


def load_model(arch, num_classes=16):
    """
    Load model from pytorch model zoo and change the number of final layser's units
    Input: 
        arch: name of architecture
        Args:
            num_classes: number of final layer's units
    output: model architecture
    """
    model = models.__dict__[arch]()
    if arch.startswith('alexnet') or arch.startswith('vgg')  \
        or arch.startswith('mnasnet') or arch.startswith('mobilenet'):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch.startswith('resne') or arch.startswith('shufflenet') \
        or arch.startswith('inception') or  arch.startswith('wide_resnet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)   
    elif arch.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)  
    elif arch.startswith('squeezenet'):
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))   
        
    return model


def GaussianBlurAll(imgs, sigma, kernel_size=(0,0)):
    """
    input: Torch.Tensor (num_images, 3, 32, 32)
    output: Torch.Tensor (num_images, 3, 32, 32)
    """
    imgs = imgs.numpy()
    imgs_list = []
    for img in imgs:
         imgs_list.append(cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma))
    imgs_list = np.array(imgs_list)
    imgs_list = imgs_list.transpose(0, 3, 1, 2)
    
    return torch.from_numpy(imgs_list)


def adjust_multi_steps(epoch):
    """
    for 'multi-steps' mode
    :param blur: flag of whether blur training images or not (default: True)
    :return: sigma
    """
    if epoch < 10:
        sigma = 4
    elif epoch < 20:
        sigma = 3
    elif epoch < 30:
        sigma = 2
    elif epoch < 40:
        sigma = 1
    else:
        sigma = 0  # no blur
    
    return sigma


def adjust_multi_steps_cbt(init_sigma, epoch, decay_rate=0.9, every=5):
    """
    Sets the sigma of Gaussian Blur decayed every 5 epoch.
    This is for 'multi-steps-cbt' mode.
    This idea is based on "Curriculum By Texture"
    Args: 
        init_sigma: initial sigma of Gaussian kernel 
        epoch: training epoch at the moment
        decay_rate: how much the model decreases the sigma value
        every: the number of epochs the model decrease sigma value
    Return: sigma of Gaussian kernel (GaussianBlur)
    """
    return init_sigma * (decay_rate ** (epoch // every))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_model(state, param_path, epoch):
    filename = param_path + 'epoch_{}.pth.tar'.format(epoch)
    torch.save(state, filename)
    

def print_settings(model, args):
    print('=' * 5 + ' settings ' + '=' * 5)
    print('TRAINING MODE: {}'.format(args.mode))
    if args.mode == 'mix':
        print('### MIX ###')
        print('Sigma: {}'.format(args.sigma))
    elif args.mode == 'single-step':
        print('### SINGLE STEP ###')
        print('## NO BLUR FROM EPOCH {:d}'.format(args.epochs // 2))
        print('Sigma: {}'.format(args.sigma))
    elif args.mode == 'reversed-single-step':
        print('### REVERSED SINGLE STEP ###')
        print('## NO BLUR TILL EPOCH {:d}'.format(args.epochs // 2))
        print('Sigma: {}'.format(args.sigma))
    elif args.mode == 'multi-steps':
        print('### MULTI STEPS ###')
        print('Step: 1-10 -> 11-20 -> 21-30 -> 31-40 -> 41-50 -> 51-{}'.format(args.epochs))
        print('Sigma: 5 -> 4 -> 3 -> 2 -> 1 -> none')
        print('#' * 20)
    elif args.mode == 'all':
        print('Sigma: {}'.format(args.sigma))
    if args.blur_val:
        print('VALIDATION MODE: blur-val')
    print('Random seed: {}'.format(args.seed))
    print('Epochs: {}'.format(args.epochs))
    print('Learning rate: {}'.format(args.lr))
    print('Weight_decay: {}'.format(args.weight_decay))
    print()
    print(model)
    print('=' * 20)
    print()
