import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from model.mbsnet import MBSNet
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
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torchvision.transforms as TF

from model.segnet import SegNet
from model.unet_model import R2U_Net, AttU_Net, R2AttU_Net, U_Net
# from model.unext import UNext
from model.transunet_model import TransUNet
from model.sknet import SKNet26
from model.nestedUNet import NestedUNet
from AUUnet import *
from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--model_name", type=str, default="U_Net")
parse.add_argument("--log_name", type=str, default="./log/test.log")
parse.add_argument("--batch_size", type=int, default=6)
parse.add_argument("--EPOCH", type=int, default=100)
parse.add_argument("--LR", type=float, default=0.0001)
parse.add_argument("--DEVICE", type=int, default=0)
parse.add_argument("--M", type=int, default=10)
parse.add_argument("--DNM1", type=int, default=1)
parse.add_argument("--DNM2", type=int, default=1)
parse.add_argument("--ckpt", type=str, help="the path of model weight file")
args = parse.parse_args()
if args.DEVICE == 0:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练
else:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练

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

filepath_bus = './data/BUS/BUS/'
filepath_STU = './data/STU-Hospital-master/'
filepath_Polyp = './data/Kvasir-SEG/'

filepath = filepath_Polyp

imagefilepath = filepath + 'data/train/'
imagefilepath_label = filepath + 'data/trainannot/'

valfilepath = filepath + 'data/val/'
valfilepath_label = filepath + 'data/valannot/'

train_dataset = dataset.Busi(imagefilepath, imagefilepath_label, transform, transform_test)
test_dataset = dataset.Busi(valfilepath, valfilepath_label, transform, transform_test)
# print(train_dataset.class_to_idx)


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


# 损失函数和模型调用
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_dice = DiceLoss()

# python predicted_other_model.py --model_name U_Net

if args.model_name == "unet":
    model = U_Net().to(DEVICE)
elif args.model_name == "AAUnet":
    model = AUUnet().to(DEVICE)
elif args.model_name == "SegNet":
    model = SegNet().to(DEVICE)
elif args.model_name == "R2UNet":
    model = R2U_Net().to(DEVICE)
elif args.model_name == "AttUNet":
    model = AttU_Net().to(DEVICE)
elif args.model_name == "R2AttUNet":
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

# 预训练模型和优化器的选用：
# pretrained_model = "./log/bus_0.5loss_norm.log.pth"
# pretrained_model = "./pic/model/pic_AAUnet.pth"

pretrained_model = './pic/model/pic_polyp_'+args.model_name+'.pth'
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)
optimizer = optim.Adam(model.parameters(), lr=args.LR)

# 预训练模型加载
pretrained = 1
if pretrained:
    pretrain_model = model
    pre_dic = torch.load(pretrained_model)
    pretrain_model.load_state_dict(pre_dic["model_static_dict"])
    model_dict = model.state_dict()
    pretrained_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 选择相同的部分
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()


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
            mask_pred = torch.where(output > 0.5, 1., 0.)
            # pdb.set_trace()
            # 可视化
            # plt.imshow(transforms.ToPILImage()(mask_pred[0].squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_pred[0].squeeze()).show()  # Alternatively

            # plt.imshow(transforms.ToPILImage()(mask_true.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_true[0].squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(img[0]).show()  # Alternatively
            # _, thresh = cv2.threshold(transforms.ToPILImage()(mask_pred[0].squeeze()), 0, 255, cv2.THRESH_BINARY)

            for i in range(args.batch_size):
                # 标记mask_true的边界
                contours_true, hierarchy_true = cv2.findContours(
                    mask_true[i].cpu().squeeze().numpy().astype('uint8'),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                img_true = transforms.ToTensor()(mask_true[i].cpu().squeeze().numpy().astype('uint8'))
                img_true = img_true * 255
                img_true = img_true.permute(1, 2, 0).numpy()
                # img_true = cv2.cvtColor(img_true, cv2.COLOR_RGB2BGR)
                img_true = cv2.cvtColor(img_true, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_true, contours_true, -1, (0, 0, 255), 2)
                # for cnt in contours_true:
                #     x, y, w, h = cv2.boundingRect(cnt)
                #     cv2.rectangle(img_true, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # cv2.drawContours(img_true, contours_true, -1, (0, 0, 255), 2)

                # 标记mask_pred的边界
                contours_pred, hierarchy_pred = cv2.findContours(
                    mask_pred[i].cpu().squeeze().numpy().astype('uint8'),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                img_pred = transforms.ToTensor()(mask_pred[i].cpu().squeeze().numpy().astype('uint8'))
                img_pred = img_pred * 255
                img_pred = img_pred.permute(1, 2, 0).numpy()
                # img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
                img_pred = cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_pred, contours_pred, -1, (0, 0, 0), 1)  # 用绿色填充轮廓
                # 把img_true和img_pred叠加
                mask = img_pred + img_true
                mask[mask > 255] = 255
                # 计算交集
                intersection = cv2.bitwise_and(img_true, img_pred)
                # 将交集部分设为白色
                intersection[intersection > 0] = 255
                mask = mask + intersection
                # img_blend = cv2.addWeighted(img_true, 0.5, img_pred, 0.5, 0)
                # pdb.set_trace()
                # 显示标记mask_true和mask_pred轮廓后的图像
                # cv2.imshow('Blend', img_blend)
                # cv2.waitKey(0)
                # img_blend = mask
                # pdb.set_trace()
                cv2.imwrite('./pic/{}_{}_{}.png'.format(batch_idx, i, args.model_name), mask)

if __name__ == '__main__':
    for epoch in range(1, args.EPOCH + 1):
        test(epoch)

