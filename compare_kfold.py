import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchmetrics.functional import dice_score
# from torchmetrics import Dice
# from tensorboardX import SummaryWriter
import dataset
import logging
import model_net
from model_net import *
from dataset import *
from PIL import Image
from AUUnet import AUUnet
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse
from sklearn.model_selection import KFold
from model.segnet import SegNet
from model.mbsnet import MBSNet
from model.unet_model import R2U_Net, AttU_Net, R2AttU_Net, U_Net
# from model.unext import UNext
from model.transunet_model import TransUNet
from model.sknet import SKNet26
from model.nestedUNet import NestedUNet
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
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
    transforms.Resize((256, 256), Image.BILINEAR),
    # transforms.Grayscale(num_output_channels=1),
    # transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 224
    transforms.ToTensor(),
    #Q transforms.Normalize(mean = mean_nums, std=std_nums) # Normalizes
    # transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

filepath_bus = './data/BUS/BUS/'
#filepath_covid = './archive/COVID19_Radiography_Dataset/COVID/'
# filepath_Normal = './archive/COVID19_Radiography_Dataset/Normal/'
filepath_STU = './data/STU-Hospital-master/'
# filepath_busi_m = './data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/'
# filepath_cloth = './data/archive/'
filepath_Polyp = './data/Kvasir-SEG/'

filepath = filepath_Polyp

imagefilepath = filepath + 'data_mask/images/'
imagefilepath_label = filepath + 'data_mask/masks/'

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

    for batch_idx, (img, label) in tqdm(enumerate(train_loader)):
        total_loss = 0
        img, label = img.to(DEVICE), label.to(DEVICE)
        model.zero_grad()
        output = model(img)
        output = output.float().cpu()
        label = label.cpu()
        # pdb.set_trace()
        loss = criterion_dice(output, label)
        # loss = criterion_bce(F.sigmoid(output), label)
        # loss = criterion_dice(output, label)
        total_loss += loss
        sum_total_loss += total_loss.data.item()
        sum_total_loss_batch += total_loss.data.item()
        total_loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 and batch_idx != 0:
            print(
                'Train Epoch: {} \tLoss: {:.6f}\tLR: {}'.format(
                    epoch, sum_total_loss_batch / 10,
                    optimizer.param_groups[0]['lr']))
            sum_total_loss_batch = 0
    print(args.model_name)
    print(
        'Train Epoch: {}\tLoss: {:.6f}'.format(epoch, (sum_total_loss * args.batch_size) / (len(train_loader.dataset))))
    logger.info(
        'Train Epoch: {}\tLoss: {:.6f}'.format(epoch, (sum_total_loss * args.batch_size) / (len(train_loader.dataset))))

    return (sum_total_loss * args.batch_size) / len(train_loader.dataset)


def calculate_metric_percase(pred, gt):
    # pdb.set_trace()
    if torch.is_tensor(pred):
        predict = pred.data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    pred = numpy.atleast_1d(predict.astype(numpy.bool))
    gt = numpy.atleast_1d(target.astype(numpy.bool))
    
    rec = metric.binary.recall(pred, gt)
    spe = metric.binary.specificity(pred, gt)
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    pre = metric.binary.precision(pred, gt)
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
            # pdb.set_trace()
            
            output = model(img)
            # mask_pred = (Out1>0.5).float()
            mask_pred = torch.sigmoid(output)
            mask_pred = torch.where(mask_pred > 0.5, 1., 0.)
            mask_true = mask_true.cpu()

            # transforms.ToPILImage()(mask_pred[0].squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(mask_true[0].squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(img[0]).show()  # Alternati
            loss = criterion_bce(torch.sigmoid(output).cpu(), mask_true)
            sum_total_loss += loss.data.item()
            # mask_pred.argmax(dim=1)
            # if epoch == 90:
            #     pdb.set_trace()
            dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(mask_pred, mask_true)
            pre_score += pre
            recall_score += rec
            dice_score += dice
            jaccard_score += jc
            spe_score += spe
            acc_score += acc
            f1_score += f1
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


def ChooseModel(args):
    if args.model_name == "unet":
        model = U_Net().to(DEVICE)	
    elif args.model_name == "SegNet":
        model = SegNet().to(DEVICE)
    elif args.model_name == "AAUnet":
        model = AUUnet().to(DEVICE) 
    elif args.model_name == "R2U_Net":
        model = R2U_Net().to(DEVICE)
    elif args.model_name == "AttU_Net":
        model = AttU_Net().to(DEVICE)
    elif args.model_name == "R2AttU_Net":
        model = R2AttU_Net().to(DEVICE)
    elif args.model_name == "NestedUNet":
        model = NestedUNet().to(DEVICE)
    elif args.model_name == "mbsnet":
        model = MBSNet().to(DEVICE)
    elif args.model_name == "transunet":
        model = TransUNet(img_dim=128,
                      in_channels=3,
                      out_channels=128,
                      head_num=4,
                      mlp_dim=512,
                      block_num=8,
                      patch_dim=16,
                      class_num=1).to(DEVICE)
    elif args.model_name == "sknet":
        model = SKNet26().to(DEVICE)
    elif args.model_name == "unext":
        model = UNeXt().to(DEVICE)
    else:
        raise ValueError("Invalid model name: " + args.model_name)
    return model


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--log_name", type=str, default="./log/test.log")
    parse.add_argument("--model_name", type=str, default="unet")
    parse.add_argument("--batch_size", type=int, default=6)
    parse.add_argument("--EPOCH", type=int, default=100)
    parse.add_argument("--LR", type=float, default=0.0001)
    parse.add_argument("--DEVICE", type=int, default=0)
    parse.add_argument("--data_name", type=str, default="log_Polyp")
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
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
        print("model_name: {}".format(args.model_name))
        logging.info("k_fold training : {} ".format(i))


        train_dataset = dataset.K_fold(imagefilepath, imagefilepath_label, transform,
                                       transform_test, train_idx)
        val_dataset = dataset.K_fold(imagefilepath, imagefilepath_label, transform, transform_test, val_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
        val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=False)

        model = ChooseModel(args)
        optimizer = optim.Adam(model.parameters(), lr=args.LR)

        for epoch in range(1, args.EPOCH + 1):
            train_loss = train(train_loader, model, epoch, args, DEVICE)

            pre, recall, dice, jaccard, spe, test_loss, acc, f1 = test(val_dataset, model, args)
            logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                        f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},acc={acc:.4f},f1={f1:.4f},test_loss={test_loss:.4f},')
      
