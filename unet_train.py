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

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--log_name", type=str, default="./log/test.log")
parse.add_argument("--batch_size", type=int, default=2)
parse.add_argument("--EPOCH", type=int, default=100)
parse.add_argument("--LR", type=float, default=0.00005)
parse.add_argument("--DEVICE", type=int, default=0)
parse.add_argument("--M", type=int, default=10, help="Number of dendrites")
parse.add_argument("--DNM", type=int, default=1)
parse.add_argument("--DNM2", type=int, default=1)
parse.add_argument("--LOSSK", type=float, default=0.5, help="The coefficient of the loss function")
parse.add_argument("--data_name", type=str, default="stu")

parse.add_argument("--ckpt", type=str, help="the path of model weight file")
args = parse.parse_args()
if args.DEVICE == 0:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 
else:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    # transforms.RandomResizedCrop(224),#Resizes all images into same dimension
    # transforms.RandomRoation(10),# Rotates the images upto Max of 10 Degrees
    # transforms.RandomHorizontalFlip(p=0.4),#Performs Horizantal Flip over images
    # transforms.RandomVerticalFlip(p=0.4),
    # transforms.RandomRotation(15),
    # transforms.RandomRotation(90, expand=True),
    # transforms.RandomHorizontalFlip(p=1.0),
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

filepath_busi = './data/Dataset_BUSI/Dataset_BUSI_with_GT/'
filepath_bus = './data/BUS/BUS/'
filepath_cloth = './data/archive/'
filepath_bus = './data/BUS/BUS/'
filepath_STU = './data/STU-Hospital-master/'
filepath_Polyp = './data/Kvasir-SEG/'

if args.data_name == "bus":
    filepath = filepath_bus
elif args.data_name == "polyp":
    filepath = filepath_Polyp
elif args.data_name == "stu":
    filepath = filepath_STU

imagefilepath = filepath + 'data/train/'
imagefilepath_label = filepath + 'data/trainannot/'

valfilepath = filepath + 'data/val/'
valfilepath_label = filepath + 'data/valannot/'

train_dataset = dataset.Busi(imagefilepath, imagefilepath_label, transform, transform_test)
test_dataset = dataset.Busi(valfilepath, valfilepath_label, transform, transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


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


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_dice = DiceLoss()
criterion_bce = nn.BCELoss()
criterion_focal = FocalLoss()
model = model_net.DDNet()
model.to(DEVICE)

# pretrained_model = "./log_beifen_weight/bus_2dnm_adam1e-4_epoch160_batchSize8.log.pth"
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)
optimizer = optim.Adam(model.parameters(), lr=args.LR)

pretrained = 0
if pretrained:
    pretrain_model = model_net.DDNet(DEVICE)
    pre_dic = torch.load(pretrain_model)
    pretrain_model.load_state_dict(pre_dic["model_static_dict"])
    model_dict = model.state_dict()
    pretrained_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.CRITICAL}
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


# logger = get_logger('./log/log_M10_2dnm_busi.log')
# log_name
logger = get_logger(args.log_name)
# logger = get_logger('./log/bus_RRCNet_2dnm_4_adam 1e-4.log')
logger.info('start training!')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)


def train(epoch):
    model.train()
    sum_total_loss_batch = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(7)]

    pdb.set_trace()
    for batch_idx, (img, label) in tqdm(enumerate(train_loader)):
        loss = [0 for i in range(7)]
        total_loss = 0
        img, label = img.to(DEVICE), label.to(DEVICE)
        model.zero_grad
        output = model(img)
        loss[0] = criterion_bce(F.sigmoid(output[0]), label)

        for i in range(1, 6):
            loss[i] = args.LOSSK * criterion_focal(output[i], label)
        for i in range(6):
            loss_sum[i] += loss[i].data.item()
        for i in range(6):
            total_loss += loss[i]
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
        print("\nTrain Epoch: {}".format(epoch))
        return (sum_total_loss * args.batch_size) / len(train_loader.dataset)


def calculate_metric_percase(pred, gt):
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
    return dice, jc, pre, rec, spe


def test(epoch):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    loss_sum = [0 for i in range(8)]
    for batch_idx, (img, mask_true) in tqdm(enumerate(test_loader)):
        img, label = img.to(DEVICE), mask_true.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            # mask_pred = (Out1>0.5).float()
            mask_pred = torch.where(output[0] > 0.5, 1., 0.)

            # Visualization
            # plt.imshow(transforms.ToPILImage()(mask_pred.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(output[0].squeeze()).show()  # Alternatively
            # plt.imshow(transforms.ToPILImage()(mask_true.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_true.squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(Out0[0].squeeze()).show()
            mask_pred = mask_pred.float().cpu()
            # mask_pred.argmax(dim=1)
            mask_true = mask_true.to(device=DEVICE, dtype=torch.long).float().cpu()
            loss = criterion_mse(mask_true, mask_pred)
            sum_total_loss += loss.data.item()

            dice, jc, pre, rec, spe = calculate_metric_percase(mask_pred, mask_true)
            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe

    print("Test Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(test_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(test_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(test_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(test_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(test_loader)))
    print("test_loss: \t{:.4f}".format(sum_total_loss / len(test_loader)))

    return pre_score / len(test_loader), recall_score / len(test_loader), dice_score / len(
        test_loader), jaccard_score / len(test_loader), spe_score / len(test_loader), sum_total_loss / len(test_loader)


def adjust_learning_rate(optimizer, epoch):
    if epoch % 80 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__ == '__main__':
    min = 1
    for epoch in range(1, args.EPOCH + 1):
        train_loss = train(epoch)
        pre, recall, dice, jaccard, spe, test_loss = test(epoch)
        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                    f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},test_loss={test_loss:.4f},')
        if min > test_loss:
            min = test_loss
            checkpoint = {
                "model_static_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dic": optimizer.state_dict()
            }
            torch.save(checkpoint, "./pic/model/" + args.log_name[5:-4] + '.pth')
