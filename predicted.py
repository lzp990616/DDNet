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
import pdb
from medpy import metric
from torchvision.datasets import ImageFolder
import os
# from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torchvision.transforms as TF
from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--log_name", type=str, default="./log/test.log")
parse.add_argument("--data_name", type=str, default="bus")
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
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    transforms.ToTensor(),  # Coverts into Tensors
])

transform_test = transforms.Compose([
    transforms.Resize((384, 384), Image.BILINEAR),
    transforms.ToTensor(),
])


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
# print(train_dataset.class_to_idx)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


# train_size = len(train_loader.dataset)
# test_size = len(test_loader.dataset) #incorrect
# train_num_batches = len(train_loader)
# test_num_batches = len(test_loader)


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


# LOSS
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_dice = DiceLoss()

model = model_net.DDNet()
model.to(DEVICE)

# Selection of pre-trained models and optimizers：
# pretrained_model = "./log/bus_0.5loss_norm.log.pth"
pretrained_model = "./pic/model/pic_polyp_ours.pth"

# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)
optimizer = optim.Adam(model.parameters(), lr=args.LR)

# Pre-trained model loading
pretrained = 0
if pretrained:
    pretrain_model = model_net.DDNet()
    pre_dic = torch.load(pretrained_model)
    pretrain_model.load_state_dict(pre_dic["model_static_dict"])
    model_dict = model.state_dict()
    pretrained_dict = pretrain_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 选择相同的部分
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
logger.info('start predicteding!')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(__name__)


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
    
    
def predicted_ori2(epoch):
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
            mask_pred = torch.where(output[1] > 0.5, 1., 0.)
            mask_true = torch.where(mask_true > 0.5, 1., 0.)
            for i in range(args.batch_size):
            
           
                
                # mask_pic = mask_true[i].cpu()+torch.logical_not(mask_true[i].cpu())
                
                # transforms.ToPILImage()(torch.where(mask_pic==1.,0.,1.)).show()             
                # mask_pil = Image.fromarray((mask_pic.squeeze(0).numpy()* 255).astype(np.uint8), mode='L')
                
                 # 标记mask_true的边界
                contours_true, hierarchy_true = cv2.findContours(
                    mask_true[i].cpu().squeeze().numpy().astype('uint8'),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                img_true = transforms.ToTensor()(mask_true[i].cpu().squeeze().numpy().astype('uint8'))
                img_true = img_true * 255
                img_true = img_true.permute(1, 2, 0).numpy()
                img_true = cv2.cvtColor(img_true, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_true, contours_true, -1, (0, 0, 255), 1)
                # img_pil = Image.fromarray(img_true.transpose((1, 2, 0)))

                # 将原图从tensor类型转为ndarray类型，并将通道顺序从(C, H, W)转为(H, W, C)
                img_np = img[i].cpu().numpy().transpose((1, 2, 0))
                
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # 在原图上绘制轮廓线
                img_np = (img_np * 255).astype(np.uint8)
                cv2.drawContours(img_np, contours_true, -1, (0, 255, 0), thickness=4)     
                # pdb.set_trace()
                # img_np += img_true
                # img_np.
                # img_np = (img_np * 255).astype(np.uint8)
                # img_np[img_np>255] = 255
                # img_pil = Image.fromarray(img_np)
                
                # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                # 保存图片
                # img_pil.save('./pic/{}_{}_mask.png'.format(batch_idx, i))
                cv2.imwrite('./pic/{}_{}_mask.png'.format(batch_idx, i), img_np)


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
            mask_pred = torch.where(output[1] > 0.5, 1., 0.)
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
                cv2.drawContours(img_pred, contours_pred, -1, (0, 0, 0), 1)
                
                # 黑色：(0, 0, 0)
                # 白色：(255, 255, 255)
                # 红色：(0, 0, 255)
                # 绿色：(0, 255, 0)
                # 蓝色：(255, 0, 0)
                # 黄色：(0, 255, 255)
                # 紫色：(255, 0, 255)
                # 青色：(255, 255, 0)
                # 橙色：(0, 165, 255)
                # 粉色：(255, 192, 203)
                # 灰色：(128, 128, 128)
                
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
                cv2.imwrite('./pic/{}_{}_Ours.png'.format(batch_idx, i), mask)

if __name__ == '__main__':
    for epoch in range(1, args.EPOCH + 1):
        # test(epoch)
        predicted_ori2(epoch)
        
