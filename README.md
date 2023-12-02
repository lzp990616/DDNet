# DDNet

[Paper]() | [Project Page](https://github.com/lzp990616/DDNet) 

> DDNet: Dendritic Deep Learning for Medical Segmentation

> [Zhipeng Liu](https://scholar.google.com/citations?hl=zh-CN&user=2r4K18sAAAAJ), [Zhiming Zhang](https://scholar.google.com/citations?user=j5YBr3IAAAAJ&hl=zh-CN&oi=sra), [Zhenyu Lei](https://scholar.google.com/citations?user=7Ss6peAAAAAJ&hl=zh-CN&oi=sra), [Masaaki Omura](), [Rong-Long Wang](https://researchmap.jp/read0114888),[Shangce Gao](https://toyamaailab.github.io/)

DDNet is a novel network that integrates biologically interpretable dendritic neurons and employs deep supervision during training to enhance the model's efficacy. To evaluate the effectiveness of the proposed methodology, comparative trials were conducted on datasets STU, Polyp, and DatasetB. The experiments demonstrate the superiority of the proposed approach.

## Overview

![demo](D:\seg_dnm\DDnet_V5\Img\DDNet-eps-converted-to\DDNet_structure.png)

DDNet presents a novel segmentation approach that leverages dendritic neurons to tackle the challenges of medical imaging segmentation. The model enhance the segmentation accuracy based on a SegNet variant including an encoder-decoder structure, an upsampling index, and a deep supervision method. Furthermore, the model introduce a dendritic neuron-based convolutional block to enable nonlinear feature mapping, thereby further improving the effectiveness of our approach. 

![demo](D:\seg_dnm\DDnet_V5\Img\DDNet-eps-converted-to\DDNet_structure.png)

The proposed method is evaluated on medical imaging segmentation datasets, and the experimental results demonstrate that it is superior to state-of-the-art methods in terms of performance.

## Getting Started

### Data_dir

```python
├── data
│   ├── BUS
│   ├── Kvasir-SEG
│   ├── STU-Hospital-master

    ├── Bus(Kvasir-SEG, STU-Hospital-master...)
    │   ├── data_mask
    │       ├── images
    │       ├── masks

    │   ├── data (Only need to create data and data_mask)
    │       ├── train
    │       ├── trainannot
    │       ├── val
    │       ├── valannot
    │       ├── test
    │       ├── testannot
```

### Data_process:
`python data_spilt.py  # Spilt the data. Select data by changing 'filepath' in code`

### K-folder_experiment:
`python k_folder.py --log_name "./log/log_name.log" --data_name stu --batch_size 8 --EPOCH 100 --LR 0.0005 --LOSSK 0.1 --DNM 1 --M 10`

### Comparative_experiment:
`python compare_kfold.py --log_name "./log/log_name.log" --data_name stu --model_name unet --batch_size 8 --EPOCH 100 --LR 0.0005`

### Pic_experiment:
`python unet_train.py --log_name "./log/log_name.log" --data_name stu --batch_size 8 --EPOCH 100 --LR 0.0005 --LOSSK 0.1 --DNM 1 --M 10`
`python unet_compare_train.py --log_name "./log/log_name.log" --data_name stu --batch_size 8 --EPOCH 100 --LR 0.0005`

### Pic_plot:
`python predicted.py`
`python predicted_other_model.py`
`predicted_other_model_pic.py  # If the number of pictures and the name of the model are changed, the file needs to be changed.` 

## Related Projects

Our code is based on [PyTorch](https://github.com/pytorch/pytorch).

## Citing DDNet

```bib

```

