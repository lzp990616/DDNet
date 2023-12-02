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

from model.segnet import SegNet
from model.unet_model import R2U_Net, AttU_Net, R2AttU_Net, U_Net
# from model.unext import UNext
from model.transunet_model import TransUNet
from model.sknet import SKNet26
from model.nestedUNet import NestedUNet

from torchmetrics.functional import precision_recall
from torchmetrics import Specificity, JaccardIndex
import argparse



image_files = [os.path.join("./pic", filename)
               for filename in os.listdir("./pic")
               if filename.endswith(".png")]
            


model_order = {"mask": 0, "Ours": 1, "unet": 2, "SegNet": 3, "NestedUNet": 4, "mbsnet": 5, "RRCnet": 6}

def sort_key(file_name):
    parts = os.path.basename(file_name).split('_')
    num_parts = tuple(map(int, parts[:-1]))
    model_name = parts[-1].split('.')[0]
    model_order_value = model_order[model_name]
    return (num_parts, model_order_value)

image_files = sorted(image_files, key=sort_key)

# pdb.set_trace()


# pdb.set_trace()
# 将图像文件划分为每七个图像的列表
image_lists = [image_files[i:i+7] for i in range(0, len(image_files), 7)]
# pdb.set_trace()
# 遍历每个子列表，并将七个图像放在同一行中
for image_list in image_lists:
    fig, axs = plt.subplots(1, 7, figsize=(18, 3))
    plt.subplots_adjust(wspace=0.1)
    for i, image_file in enumerate(image_list):
        image = plt.imread(image_file)
        axs[i].imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])
        axs[i].axis('off')
        model_name = os.path.basename(image_file).split('_')[2].split('.')[0]  # 从文件名中获取模型名称
        if model_name == "mask":
            model_name = "Mask"
        elif model_name == "unet":
            model_name = "U-Net"
        elif model_name == "mbsnet":
            model_name = "MBSnet"
        elif model_name == "NestedUNet":
            model_name = "U-Net++"
        
        axs[i].text(0.5, -0.2, model_name, size=23, ha="center", transform=axs[i].transAxes)  # 在图像下方添加模型名称
    plt.savefig('./pic_row/row_{}.png'.format(image_files.index(image_list[0])))
    plt.close()

