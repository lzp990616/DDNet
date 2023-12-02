import os
import pdb
import random
import shutil

'''
├── data(按照7:2:1比例划分)
│   ├── train 存放用于训练的图片
│   ├── trainannot 存放用于训练的图片标注
│   ├── val 存放用于验证的图片
│   ├── valannot 存放用于验证的图片标注
│   ├── test 存放用于测试的图片
│   ├── testannot 存放用于测试的图片标注
'''

# 数据集原始图片所存放的文件夹，必须为png文件
filepath_bus = './data/BUS/BUS/'
filepath_busi = './data/Dataset_BUSI/Dataset_BUSI_with_GT/'
filepath_cloth = './data/archive/'
file_path_busi_m = './data/Dataset_BUSI_malignant/Dataset_BUSI_with_GT/'
filepath_Polyp = './data/Kvasir-SEG/'
filepath_stu = './data/STU-Hospital-master/'
filepath = filepath_stu
# filepath = file_path_busi_m


# 创建数据集文件夹
dirpath_list = [filepath+'/data/train',
                filepath+'/data/trainannot',
                filepath+'/data/val',
                filepath+'/data/valannot',
                filepath+'/data/test',
                filepath+'/data/testannot']

imagefilepath = filepath + 'data_mask/images'  

# pdb.set_trace()

# dirpath_list_busi = ['./data/Dataset_BUSI/Dataset_BUSI_with_GT/data/train',
#                 './data/Dataset_BUSI/Dataset_BUSI_with_GT/data/trainannot',
#                 './data/Dataset_BUSI/Dataset_BUSI_with_GT/data/val',
#                 './data/Dataset_BUSI/Dataset_BUSI_with_GT/data/valannot',
#                 './data/Dataset_BUSI/Dataset_BUSI_with_GT/data/test',
#                 './data/Dataset_BUSI/Dataset_BUSI_with_GT/data/testannot']


                
for dirpath in dirpath_list:
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)  # 删除原有的文件夹
        os.makedirs(dirpath)  # 创建文件夹
    elif not os.path.exists(dirpath):
        os.makedirs(dirpath)

# 训练集、验证集、测试集所占比例
train_percent = 0.8
val_percent = 0.2
test_percent = 0




total_img = os.listdir(imagefilepath)

# 所有数据集的图片名列表
total_name_list = [row.split('.')[0] for row in total_img]
num = len(total_name_list)
# pdb.set_trace()



num_list = range(num)
# 训练集、验证集、测试集所包含的图片数目
train_tol = int(num * train_percent)
val_tol = int(num * val_percent)
test_tol = int(num * test_percent)

# 训练集在total_name_list中的index
train_numlist = random.sample(num_list, train_tol)
# 验证集在total_name_list中的index
val_test_numlist = list(set(num_list) - set(train_numlist))
val_numlist = random.sample(val_test_numlist, val_tol)
# 测试集在total_name_list中的index
test_numlist = list(set(val_test_numlist) - set(val_numlist))
# 将数据集和标签图片安装分类情况依次复制到对应的文件夹
for i in train_numlist:
    img_path = filepath + 'data_mask/images/' + total_name_list[i] + '.png'
    new_path = filepath + 'data/train/' + total_name_list[i] + '.png'
    shutil.copy(img_path, new_path)
    img_path = filepath + 'data_mask/masks/' + total_name_list[i] + '.png'
    new_path = filepath + 'data/trainannot/' + total_name_list[i] + '.png'
    shutil.copy(img_path, new_path)
for i in val_numlist:
    img_path = filepath + 'data_mask/images/' + total_name_list[i] + '.png'
    new_path = filepath + 'data/val/' + total_name_list[i] + '.png'
    shutil.copy(img_path, new_path)
    img_path = filepath + 'data_mask/masks/' + total_name_list[i] + '.png'
    new_path = filepath + 'data/valannot/' + total_name_list[i] + '.png'
    shutil.copy(img_path, new_path)
for i in test_numlist:
    img_path = filepath + 'data_mask/images/' + total_name_list[i] + '.png'
    new_path = filepath + 'data/test/' + total_name_list[i] + '.png'
    shutil.copy(img_path, new_path)
    img_path = filepath + 'data_mask/masks/' + total_name_list[i] + '.png'
    new_path = filepath + 'data/testannot/' + total_name_list[i] + '.png'
    shutil.copy(img_path, new_path)
