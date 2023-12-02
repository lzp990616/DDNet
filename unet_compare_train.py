import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import dataset
import logging
import model_net
from model_net import *
from dataset import *
from PIL import Image
import pdb
from medpy import metric
import argparse
# from model.unet import U_Net
from model.segnet import SegNet
from model.unet_model import R2U_Net, AttU_Net, R2AttU_Net, U_Net, U_Net_Dnm
from unet_parts import *
from model.transunet_model import TransUNet
from model.sknet import SKNet26
from model.mbsnet import MBSNet
from model.nestedUNet import NestedUNet
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

parse = argparse.ArgumentParser()
parse = argparse.ArgumentParser()
# parse.add_argument("action", type=str, help="train or test")
parse.add_argument("--log_name", type=str, default="./log/test.log")
parse.add_argument("--model_name", type=str, default="test")
parse.add_argument("--data_name", type=str, default="stu")
parse.add_argument("--batch_size", type=int, default=2)
parse.add_argument("--EPOCH", type=int, default=100)
parse.add_argument("--LR", type=float, default=0.001)
parse.add_argument("--DEVICE", type=int, default=0)
parse.add_argument("--ckpt", type=str, help="the path of model weight file")
args = parse.parse_args()
if args.DEVICE == 0:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练
else:
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU训练

transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    # transforms.RandomRotation(15),# Rotates the images upto Max of 10 Degrees
    # transforms.RandomHorizontalFlip(p=0.1),#Performs Horizantal Flip over images
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.ToTensor(),  
    # transforms.Grayscale(num_output_channels=1),
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
        
        
def met_fun(predict, target):  # Sensitivity, Recall, true positive rate都一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    tn = numpy.count_nonzero(~predict & ~target)
    fn = numpy.count_nonzero(~predict & target)
    fp = numpy.count_nonzero(predict & ~target)



    return tp, fp, tn, fn


criterion_bce = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_ce = nn.CrossEntropyLoss()
criterion_dice = DiceLoss()

# model = U_Net().to(DEVICE)

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






# 预训练模型和优化器的选用：
# pretrained_model = "./log_beifen_weight/bus_2dnm_adam1e-4_epoch160_batchSize8.log.pth"
# optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)
optimizer = optim.Adam(model.parameters(), lr=args.LR)

# 预训练模型加载
pretrained = 0
if pretrained:
    pretrain_model = model_net.DDNet(DEVICE)
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


logger = get_logger(args.log_name)
# logger = get_logger('./log/bus_RRCNet_2dnm_4_adam 1e-4.log')
logger.info('start training!')
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


def train(epoch):
    model.train()  # 模型的训练模式
    loss_sum = 0
    loss_sum_batch = 0
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    acc_score = 0
    f1_score = 0
    for i, (img, label) in enumerate(train_loader):
        img, label = img.to(DEVICE), label.to(DEVICE)
        model.zero_grad()
        output = model(img)
        output = output.float().cpu()
        # mask_pred = torch.where(output > 0.5, 1., 0.)
        mask_true = label.cpu()       
        loss = criterion_dice(output, mask_true)
        loss_sum += loss.data.item()
        loss_sum_batch += loss.data.item()
        loss.backward()
        optimizer.step()
        dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(torch.where(output > 0.5, 1., 0.).cpu(), mask_true)
        
        
        # print(dice, jc, pre, rec, spe, )
        pre_score += pre
        recall_score += rec
        dice_score += dice
        jaccard_score += jc
        spe_score += spe
        acc_score += acc
        f1_score += f1
        if (i+1) % 10 == 0:
            print("EPOCH: {}\tTrain_loss: {:.6f}".format(epoch, loss_sum_batch / 10))
            loss_sum_batch = 0

    print(
        'Epoch: {}\tTrain_loss: {:.6f}'.format(epoch, (loss_sum * args.batch_size) / (len(train_loader.dataset))))
    print("Train Epoch: {}".format(epoch))
    print("pre_score: \t{:.4f}".format(pre_score / len(train_loader)))
    print("recall_score: \t{:.4f}".format(recall_score / len(train_loader)))
    print("dice_score: \t{:.4f}".format(dice_score / len(train_loader)))
    print("jaccard_score: \t{:.4f}".format(jaccard_score / len(train_loader)))
    print("spe_score: \t{:.4f}".format(spe_score / len(train_loader)))
    print("acc_score: \t{:.4f}".format(acc_score / len(test_loader)))
    print("f1_score: \t{:.4f}".format(f1_score / len(test_loader)))
    return loss_sum/ len(test_loader)


# mask_pred = output.float().cpu()
# mask_true = torch.where(label==0., 0., 1.)
# transforms.ToPILImage()(mask_pred[0].squeeze()).show()  # Alternatively
# transforms.ToPILImage()(mask_true[0].squeeze()).show()  # Alternatively


def test(epoch):
    model.eval()
    dice_score = 0
    jaccard_score = 0
    pre_score = 0
    recall_score = 0
    spe_score = 0
    sum_total_loss = 0
    acc_score = 0
    f1_score = 0
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            output = model(img)
            mask_pred = torch.sigmoid(output) 
            mask_pred = torch.where(mask_pred > 0.5, 1., 0.)
            mask_true = label.cpu()
            # mask_true = label.to(device=DEVICE, dtype=torch.long).float().cpu()
            # if epoch == 10:
                # pdb.set_trace()
            # mask_true = torch.where(label == 0., 0., 1.

            # mask_true = label.to(device=DEVICE, dtype=torch.long).cpu()
            # 可视化
            # plt.imshow(transforms.ToPILImage()(mask_pred[0].squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_pred[0].squeeze()).show()  # Alternatively

            # plt.imshow(transforms.ToPILImage()(mask_true.squeeze()), interpolation="bicubic")
            # transforms.ToPILImage()(mask_true[0].squeeze()).show()  # Alternatively
            # transforms.ToPILImage()(img[0]).show()  # Alternatively

            # tp, fp, tn, fn = met_fun(mask_pred, mask_true)
            # print(tp, fp, tn, fn)
            loss = criterion_dice(output.cpu(), mask_true)
            sum_total_loss += loss.data.item()
            dice, jc, pre, rec, spe, acc, f1 = calculate_metric_percase(mask_pred, mask_true)
            # print(dice, jc, pre, rec, spe, )
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



if __name__ == "__main__":
    min = 1.
    for epoch in range(1, args.EPOCH + 1):
        train_loss = train(epoch)  # 调用训练函数
        pre, recall, dice, jaccard, spe, test_loss, acc, f1 = test(epoch)
        logger.info(f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                        f'pre={pre:.4f}, recall={recall:.4f}, dice={dice:.4f}, jaccard={jaccard:.4f},spe={spe:.4f},acc={acc:.4f},f1={f1:.4f},test_loss={test_loss:.4f},')
        if min > test_loss:
            min = test_loss
            checkpoint = {
                "model_static_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dic": optimizer.state_dict()
            }
            print(args.log_name)
            torch.save(checkpoint, "./pic/model/" + args.log_name[5:-4] + '.pth')
