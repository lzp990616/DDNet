import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import dataset
import logging
import model_net
from model_net import *
from dataset import *
from PIL import Image
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

from torch.autograd import Variable
import warnings

warnings.filterwarnings("ignore")

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.RandomResizedCrop(224),#Resizes all images into same dimension
    # transforms.RandomRoation(10),# Rotates the images upto Max of 10 Degrees
    # transforms.RandomHorizontalFlip(p=0.4),#Performs Horizantal Flip over images
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Coverts into Tensors
    # transforms.Normalize(mean=mean_nums, std=std_nums)  # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    # normalize
])

transform_test = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 224
    transforms.ToTensor(),
    # transforms.Normalize(mean = mean_nums, std=std_nums) # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

filepath_bus = './data/BUS/BUS/'
filepath_STU = './data/STU-Hospital-master/'
# filepath_busi_m = './data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/'
# filepath_cloth = './data/archive/'
filepath_Polyp = './data/Kvasir-SEG/'
filepath_load = './data/my_datasets/'

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


criterion_focal = FocalLoss()
criterion_dice = DiceLoss()
criterion_bce = nn.BCELoss()
criterion_mse = nn.MSELoss()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def train(train_loader, model, epoch, args, DEVICE):
    model.train()
    sum_total_loss_batch = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(7)]
    pre_score = 0
    recall_score = 0
    dice_score = 0
    jaccard_score = 0
    acc_score = 0
    f1_score = 0
    spe_score = 0

    if args.loss_name == "dice":
        criterion_loss = criterion_dice
    elif args.loss_name == "focal":
        criterion_loss = criterion_focal

    for batch_idx, (img, label) in enumerate(train_loader):
        loss = [0 for i in range(6)]
        total_loss = 0
        img, label = img.to(DEVICE), label.to(DEVICE)
        model.zero_grad()
        output = model(img)
        loss[0] = criterion_bce(F.sigmoid(output[0]), label)
        for i in range(1, 6):
            loss[i] = args.LOSSK * criterion_loss(output[i], label)
        for i in range(6):
            loss_sum[i] += loss[i].data.item()
        for i in range(6):
            total_loss += loss[i]
        dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(torch.where(output[0] > 0.5, 1., 0.).cpu(),
                                                                    label.cpu())
        pre_score += pre
        recall_score += rec
        dice_score += dice
        jaccard_score += jc
        spe_score += spe
        acc_score += acc
        f1_score += f1
        sum_total_loss += total_loss.data.item()
        sum_total_loss_batch += total_loss.data.item()
        total_loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 and batch_idx != 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss0: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\tLoss4: {:.6f}\tLoss5: {:.6f}\t'.format(
                    epoch, batch_idx * len(img), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), sum_total_loss_batch / 10, loss_sum[0] / 10,
                           loss_sum[1] / 10, loss_sum[2] / 10, loss_sum[3] / 10, loss_sum[4] / 10, loss_sum[5] / 10))
            sum_total_loss_batch = 0
            loss_sum = [0 for i in range(6)]
    # pdb.set_trace()

    print("\nTrain Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(train_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(train_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(train_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(train_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(train_loader)))
    print("acc_score: \t{:.4f}".format(acc_score / len(train_loader)))
    print("f1_score: \t{:.4f}".format(f1_score / len(train_loader)))
    print('Train_loss: \t{:.4f}'.format((sum_total_loss * args.batch_size) / (len(train_loader.dataset))))
    return (sum_total_loss * args.batch_size) / len(train_loader.dataset)


def calculate_metric_percase(pred, gt):
    # pdb.set_trace()
    if torch.is_tensor(pred):
        predict = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    pred = numpy.atleast_1d(predict.astype(numpy.bool))
    gt = numpy.atleast_1d(target.astype(numpy.bool))

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    pre = metric.binary.precision(pred, gt)
    rec = metric.binary.recall(pred, gt)
    spe = metric.binary.specificity(pred, gt)
    acc = accuracy_score(np.ravel(pred.astype(np.int)), np.ravel(gt.astype(np.int)))
    f1 = f1_score(np.ravel(pred.astype(np.int)), np.ravel(gt.astype(np.int)))
    # acc = accuracy_score(np.ravel(pred.astype(int)), np.ravel(gt.astype(int)))
    # f1 = f1_score(np.ravel(pred.astype(bool)), np.ravel(gt.astype(bool)))
    return dice, jc, pre, rec, spe, acc, f1


def test(test_loader, model, args):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    acc_score = 0
    f1_score = 0
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, mask_true = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            Out0, Out1, Out2, Out3, Out4, Out5 = model(img)
            # mask_pred = (Out1>0.5).float()
            mask_pred = torch.where(Out0 > 0.5, 1., 0.)
            # Visualization
            # plt.imshow(transforms.ToPILImage()(mask_pred.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(Out1.squeeze()).show()  # Alternatively
            # plt.imshow(transforms.ToPILImage()(mask_true.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_true.squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(Out0[0].squeeze()).show()
            mask_pred = mask_pred.float().cpu()
            mask_true = mask_true.to(device=DEVICE, dtype=torch.long).float().cpu()
            loss = criterion_mse(mask_true, mask_pred)
            dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(mask_pred, mask_true)

            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe
            acc_score += acc
            f1_score += f1

            sum_total_loss += loss.data.item()

    print("Test Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(test_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    print("acc_score: \t{:.4f}".format(acc_score / len(test_loader)))
    print("f1_score: \t{:.4f}".format(f1_score / len(test_loader)))
    print("test_loss: \t{:.4f}\n".format(sum_total_loss / len(test_loader)))

    return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(
        test_loader), jaccard_score / len(test_loader), spe_score / len(test_loader), sum_total_loss / len(
        test_loader), acc_score / len(
        test_loader), f1_score / len(
        test_loader)


def adjust_learning_rate(optimizer, epoch):
    if epoch % 60 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--log_name", type=str, default="./log/test.log")
    parse.add_argument("--LOSSK", type=float, default=0.1, help="The coefficient of the loss function")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--EPOCH", type=int, default=100)
    parse.add_argument("--LR", type=float, default=0.00005)
    parse.add_argument("--DEVICE", type=int, default=0)
    parse.add_argument("--M", type=int, default=1, help="Number of dendrites")
    parse.add_argument("--DNM", type=int, default=1)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    parse.add_argument("--data_name", type=str, default="stu")
    parse.add_argument("--loss_name", type=str, default="dice")
    args = parse.parse_args()

    if args.data_name == "bus":
        filepath = filepath_bus
    elif args.data_name == "polyp":
        filepath = filepath_Polyp
    elif args.data_name == "stu":
        filepath = filepath_STU

    imagefilepath = filepath + 'data_mask/images/'
    imagefilepath_label = filepath + 'data_mask/masks/'

    if args.DEVICE == 0:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    logger = get_logger(args.log_name)

    total_img = os.listdir(imagefilepath)
    total_label = os.listdir(imagefilepath_label)

    skf = KFold(n_splits=4, shuffle=True)

    for i, (train_idx, val_idx) in enumerate(skf.split(total_img, total_label)):
        print("k_fold training : {} ".format(i))
        logging.info("k_fold training : {} ".format(i))

        train_dataset = dataset.K_fold(imagefilepath, imagefilepath_label, transform,
                                       transform_test, train_idx)
        val_dataset = dataset.K_fold(imagefilepath, imagefilepath_label, transform, transform_test, val_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, drop_last=True)
        val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=False, drop_last=True)

        model = model_net.DDNet(m=args.M, flag=args.DNM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.LR)

        for epoch in range(1, args.EPOCH + 1):
            train_loss = train(train_loader, model, epoch, args, DEVICE)
            pre, recall, dice, jaccard, spe, test_loss, acc, f1 = test(val_dataset, model, args)
            logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                        f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},acc={acc:.4f},f1={f1:.4f},test_loss={test_loss:.4f},')
            adjust_learning_rate(optimizer, epoch)
