import os
from config_segmentation import config
opx = config()
opt = opx()
try:
    os.makedirs(opt.out)
except OSError:
    pass

#指定GPU训练
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
from utils import get_scheduler
from partitioner import partition_dataset
from qfedavg_train_valid_unet_segmentation import train, valid
from dataset_segmentation import MedicalImageDataset


import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn import init
import torch.distributed as dist




if opt.setting != 'central':
    dist.init_process_group(
        backend=opt.backend,  # 定义在config.py
        init_method=opt.init_method,
        rank=opt.rank,
        world_size=opt.world_size)

def init_weights(net, init_type='normal',gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)

# 数据读取
def dataload(opt_dataset):
    # dataloader_tmp = 1
    if opt_dataset == "train":
        datasets = MedicalImageDataset("train", os.path.join(opt.root,'FDG-nii-raw-'+opt.dataset_choice) if opt.dataset_choice !='all' else opt.root, transform=transforms.ToTensor(),
                                       mask_transform=transforms.ToTensor(), augment=True, equalize=False)
       # 'central':
        dataloader_tmp = torch.utils.data.DataLoader(datasets, batch_size=opt.batchSize,
                                    shuffle=True,num_workers=4)         
        if opt.setting == 'FL':
            dataloader_tmp, _ = partition_dataset(dataset=datasets, batch_size=16)

                                       
    elif opt_dataset == "valid":
        datasets = MedicalImageDataset("val", os.path.join(opt.root,'FDG-nii-raw-'+opt.dataset_choice) if opt.dataset_choice !='all' else opt.root, transform=transforms.ToTensor(),
                                       mask_transform=transforms.ToTensor(), augment=False, equalize=False)
        dataloader_tmp = torch.utils.data.DataLoader(datasets, batch_size=opt.batchSize,
                                        shuffle=False)
    
    elif opt_dataset == "test":
        datasets = MedicalImageDataset("test",os.path.join(opt.root,'FDG-nii-raw-'+opt.dataset_choice) if opt.dataset_choice !='all' else opt.root, transform=transforms.ToTensor(),
                                       mask_transform=transforms.ToTensor(), augment=False, equalize=False)
        dataloader_tmp = torch.utils.data.DataLoader(datasets, batch_size=opt.batchSize,
                                            shuffle=False)

    return dataloader_tmp


# 定义网络
# CT的模型
# if opt.named == 'petct':
#     model = Unet_petct(2, 2)
# else:
# model = Unet(1, 2)
# PetCT的训练效果
if opt.model_choice == 'best_model_embeding_attention':
    from unet_segmentation_V2 import Unet
    model = Unet(2, 2)
elif opt.model_choice == 'navie_model_concat':
    from unet_segmentation import Unet
    if opt.dataset_named == 'petct':
        model = Unet(2, 2)
    else:
        model = Unet(1, 2)

# model = nn.DataParallel(model)
# init_weights(model,init_type=opt.init_type,gain=opt.init_gain)
model.to(torch.device("cuda"))


if opt.modelWeights != '':
    checkpoint = torch.load(opt.modelWeights)
    model.load_state_dict(checkpoint["model_state_dict"])

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optim = optim.Adam(model.parameters(), lr=opt.modelLR, betas=(opt.beta1,0.999))
if opt.modelWeights != '':
    optim.load_state_dict(checkpoint["optimizer_state_dict"])

#定义学习率衰减
scheduler = get_scheduler(optim,opt)
import numpy as np

# 训练
dice_best = 0.
opt.dataset = "train"
dataloader_train = dataload(opt.dataset)
opt.dataset = "valid"
dataloader_valid = dataload(opt.dataset)

result = []
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # print("lr:{}".format(optim.param_groups[0]["lr"]))

    result.append(train(model, criterion, dataloader_train, optim, epoch))

    # dice  = valid(model, criterion, dataloader_valid, epoch)
    Valid_Epoch, all_rank_dice1,Mean_Dice, Dice1 , all_rank_voe,voe = valid(model, criterion, dataloader_valid, epoch)
    result.append('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f},voe:{:.6f},alldice:{},allvoe:{}'.format(
            Valid_Epoch, Mean_Dice, Dice1,voe,all_rank_dice1,all_rank_voe))


    dice = Mean_Dice
    
       
    strroot=opt.root
    strroot=strroot.replace('./','')



    save_model_root = f'{opt.out}s/qlwhole_body'
    save_client_model_root = f'{opt.out}s/qlwhole_body_client'

    if os.path.exists(save_model_root) != True:
        os.makedirs(save_model_root)
    if os.path.exists(save_client_model_root) != True:
        os.makedirs(save_client_model_root)


    if dice >= dice_best:
        dice_best = dice
        torch.save({"model_state_dict": model.state_dict()},
                "{}s/{}/{}_{}_{}_{}_DP_{}_{}_lr10-4_epoch200_best_{}.pth".format(opt.out, strroot,opt.dataset_choice,opt.dataset_named,opt.setting,opt.model_choice,opt.DP,opt.dp_delta,epoch))
        print(dist.get_rank())
        for rank in range(4):
            if dist.get_rank() == rank: 
                torch.save({"model_state_dict": model.state_dict()},
                        "{}/{}_{}_{}_No{}_{}_DP_{}_{}_best_{}.pth".format(save_client_model_root, opt.dataset_choice, opt.dataset_named,opt.setting, dist.get_rank(), opt.model_choice,opt.DP,opt.dp_delta,epoch))
    
    else:
        torch.save({"model_state_dict": model.state_dict()},
                "{}s/{}/{}_{}_{}_DP_{}_{}_lr10-4_epoch200_{}.pth".format(opt.out,strroot,opt.dataset_choice,opt.dataset_named,opt.setting,opt.model_choice,opt.DP,opt.dp_delta,epoch))
        
        for rank in range(4):
            if dist.get_rank() == rank: 
                torch.save({"model_state_dict": model.state_dict()},
                        "{}/{}_{}_{}_No{}_{}_DP_{}_{}_{}.pth".format(save_client_model_root, opt.dataset_choice, opt.dataset_named,opt.setting, dist.get_rank(), opt.model_choice,opt.DP,opt.dp_delta,epoch))
    

    if opt.setting != 'central':
        np.savetxt(f'whole_body_result/qqqq{opt.setting}_{opt.dataset_named}_{strroot}_{opt.dataset_choice}_{opt.model_choice}_DP_{opt.DP}_{opt.dp_delta}qfed.txt', result,delimiter=',', fmt='%s')
    else:
        np.savetxt(f'whole_body_result/qqq{opt.setting}_{opt.dataset_named}_{strroot}_{opt.dataset_choice}_{opt.model_choice}.txt', result,delimiter=',', fmt='%s')
   
    # 等间隔更新学习率
    scheduler.step()



       

