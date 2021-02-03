import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy


def load_data(
    batch_size, in_path="/mnt/data/ImageNet/ILSVRC2012/", in_info_path="../info/"
):
    """
    load 16-class-ImageNet
    Arguments:
        batch_size: the batch size used in training and test
        in_path: the path to ImageNet
        in_info_path: the path to the directory that contains
                    imagenet_class_index.json, wordnet.is_a.txt, words.txt
    Returns: train_loader, test_loader
    """

    # 16-class-imagenet
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid = common_superclass_wnid("geirhos_16")
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

    custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py

    # standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # add normalization
    custom_dataset.transform_train.transforms.append(normalize)
    custom_dataset.transform_test.transforms.append(normalize)

    train_loader, test_loader = custom_dataset.make_loaders(
        workers=10, batch_size=batch_size
    )

    return train_loader, test_loader


def load_model(arch, num_classes=16):
    """
    Load model from pytorch model zoo and change the number of final layser's units
    Args:
        arch: name of architecture
        num_classes: number of last layer's units
    Returns: model (torch.model)
    """
    model = models.__dict__[arch]()
    if (
        arch.startswith("alexnet")
        or arch.startswith("vgg")
        or arch.startswith("mnasnet")
        or arch.startswith("mobilenet")
    ):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif (
        arch.startswith("resne")
        or arch.startswith("shufflenet")
        or arch.startswith("inception")
        or arch.startswith("wide_resnet")
    ):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch.startswith("squeezenet"):
        model.classifier[1] = nn.Conv2d(
            model.classifier[1].in_channels,
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    return model


def GaussianBlurAll(imgs, sigma, kernel_size=(0, 0)) -> torch.Tensor:
    """
    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        sigma: Standard deviation of Gaussian kernel.
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    if sigma == 0:
        return imgs  # do nothing
    else:
        imgs = imgs.numpy()
        imgs_list = []
        for img in imgs:
            imgs_list.append(
                cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma)
            )
        imgs_list = np.array(imgs_list)
        imgs_list = imgs_list.transpose(0, 3, 1, 2)
        return torch.from_numpy(imgs_list)


def RandomGaussianBlurAll(
    imgs, min_sigma, max_sigma, kernel_size=(0, 0)
) -> torch.Tensor:
    """Return Blurred images by random sigma.
    Each image is blurred by a sigma chosen by randomly from [min_sigma, max_sigma].

    Args:
        imgs: Images (torch.Tensor)
            size: (N, 3, 224, 224)
        kernel_size: This size will be automatically adjusted.
    Returns: Blurred images (torch.Tensor)
            size: (N, 3, 224, 224)
    """
    imgs = imgs.numpy()

    imgs_list = []
    for img in imgs:
        # Choose a random sigma for each image
        sigma = random.uniform(min_sigma, max_sigma)

        imgs_list.append(cv2.GaussianBlur(img.transpose(1, 2, 0), kernel_size, sigma))

    imgs_list = np.array(imgs_list)
    imgs_list = imgs_list.transpose(0, 3, 1, 2)
    return torch.from_numpy(imgs_list)


def adjust_multi_steps_cbt(sigma, epoch, decay_rate=0.9, every=5):
    """
    Sets the sigma of Gaussian Blur decayed every 5 epoch.
    This is for 'multi-steps-cbt' mode.
    This idea is based on "Curriculum By Texture"
    :param sigma: blur parameter
    :param blur: flag of whether blur training images or not (default: True)
    :param epoch: training epoch at the moment
    :param decay_rate: how much the model decreases the sigma value
    :param every: the number of epochs the model decrease sigma value
    :return: sigma, blur
    """
    if epoch < every:
        pass
    elif epoch % every == 0:
        sigma = sigma * decay_rate

    # if epoch >= 40:
    #    blur = False

    # return args.init_sigma * (args.cbt_rate ** (epoch // every))
    return sigma


def adjust_multi_steps(epoch):
    """
    for 'multi-steps' mode
    :param blur: flag of whether blur training images or not (default: True)
    :param epoch: training epoch at the moment
    :return: sigma, blur
    """
    """
    if epoch < 10:
        sigma = 5
    elif epoch < 20:
        sigma = 4
    elif epoch < 30:
        sigma = 3
    elif epoch < 40:
        sigma = 2
    elif epoch < 50:
        sigma = 1
    else:
        sigma = 0  # no blur
    
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


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
    filename = param_path + "epoch_{}.pth.tar".format(epoch)
    torch.save(state, filename)


def print_settings(model, args):
    print("=" * 5 + " settings " + "=" * 5)
    print("TRAINING MODE: {}".format(args.mode))
    if args.mode == "mix":
        print("### MIX ###")
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "single-step":
        print("### SINGLE STEP ###")
        print("## NO BLUR FROM EPOCH {:d}".format(args.epochs // 2))
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "reversed-single-step":
        print("### REVERSED SINGLE STEP ###")
        print("## NO BLUR TILL EPOCH {:d}".format(args.epochs // 2))
        print("Sigma: {}".format(args.sigma))
    elif args.mode == "multi-steps":
        print("### MULTI STEPS ###")
        print(
            "Step: 1-10 -> 11-20 -> 21-30 -> 31-40 -> 41-50 -> 51-{}".format(
                args.epochs
            )
        )
        print("Sigma: 5 -> 4 -> 3 -> 2 -> 1 -> none")
        print("#" * 20)
    elif args.mode == "all":
        print("Sigma: {}".format(args.sigma))
    if args.blur_val:
        print("VALIDATION MODE: blur-val")
    print("Random seed: {}".format(args.seed))
    print("Epochs: {}".format(args.epochs))
    print("Learning rate: {}".format(args.lr))
    print("Weight_decay: {}".format(args.weight_decay))
    print()
    print(model)
    print("=" * 20)
    print()
