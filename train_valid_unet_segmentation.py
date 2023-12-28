import torch
import torch.nn as nn
from utils import computeDiceOneHot, getOneHotSegmentation, getTargetSegmentation, predToSegmentation, DicesToDice
import torch.distributed as dist
from config_segmentation import config
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast,GradScaler
flag = False
scale = GradScaler(enabled=flag)
opx = config()
opt = opx()

def add_gaussian(updates,delta=0.001):
    '''inject gaussian noise to a vector'''
    updates += torch.FloatTensor(np.random.normal(0, delta,updates.shape)).to(torch.device("cuda"))
    return updates

def average_weights(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if opt.DP == 'yes':
            grad_tensor = param.data
            grad_tensor = add_gaussian(grad_tensor, delta=opt.dp_delta)
        
        dist.all_reduce(grad_tensor,op=dist.ReduceOp.SUM)
        param.data /= size


def train(model, crierion, data_loader_train, optimizer, epoch):
    model.train()
    loss_train = 0.
    DiceB_train = 0.
    voe_score_train=0.
    # DiceW_train = 0.
    # DiceT_train = 0.
    # DiceZ_train = 0.
    Dice_score_train = 0.
    softmax = nn.Softmax()
    Diceloss = computeDiceOneHot()
    softmax.to(torch.device("cuda"))
    Diceloss.to(torch.device("cuda"))
    
    i = 0
    for batch_idx,(ct_image, pet_image, label) in tqdm(enumerate(data_loader_train), total=len(data_loader_train)):
        i = i + 1
        if i != len(data_loader_train):
            ct_image, pet_image, label = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")),\
                                        label.to(torch.device("cuda"))

            optimizer.zero_grad()
            with autocast(flag):
                segmentation_prediction = model(ct_image, pet_image)
                predclass_y = softmax(segmentation_prediction)
                Segmentation_planes = getOneHotSegmentation(label)
                segmentation_prediction_ones = predToSegmentation(predclass_y)
                # It needs the logits, not the softmax
                Segmentation_class = getTargetSegmentation(label)
            # print(segmentation_prediction.shape)
            # print(Segmentation_class.shape)
                loss = crierion(segmentation_prediction, Segmentation_class)
            loss_train += loss.item()
            # 计算DSC
            # 分5类
            # DiceN, DiceB, DiceW, DiceT, DiceZ = Diceloss(segmentation_prediction_ones, Segmentation_planes)
            # 分2类
            DiceN, DiceB ,voe= Diceloss(segmentation_prediction_ones, Segmentation_planes)
            DiceB = DicesToDice(DiceB)
            # DiceW = DicesToDice(DiceW)
            # DiceT = DicesToDice(DiceT)
            # DiceZ = DicesToDice(DiceZ)
            # 分5类
            # Dice_score = (DiceB + DiceW + DiceT + DiceZ) / 4
            # 分2类
            # Dice_score = (DiceB + DiceW) / 2
            Dice_score = (DiceB) / 2
            DiceB_train += DiceB.item()
            # DiceW_train += DiceW.item()
            # DiceT_train += DiceT.item()
            # DiceZ_train += DiceZ.item()
            Dice_score_train += Dice_score.item()

            voe_score_train+=voe.item()

            # loss.backward()
            # optimizer.step()
            scale.scale(loss).backward()
            scale.step(optimizer)
            scale.update()
            if opt.setting == 'FL' and opt.standalone=='false':
                average_weights(model)
            # 分5类
            # if (batch_idx+1)/(len(data_loader_train)) == 1:
            #     print('Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f}, Dice2: {:.6f}, Dice3: {:.6f}, Dice4: {:.6f}'.format(
            #         epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
            #         DiceB_train/len(data_loader_train),DiceW_train/len(data_loader_train),DiceT_train/len(data_loader_train)
            #         , DiceZ_train/len(data_loader_train)))
            # 分3类
            # if (batch_idx+1)/(len(data_loader_train)) == 1:
            #     print('Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f}, Dice2: {:.6f}'.format(
            #         epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
            #         DiceB_train/len(data_loader_train),DiceW_train/len(data_loader_train)))
            # 2分类
            if (batch_idx+1)/(len(data_loader_train)) == 1:
                print('Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f},voe:{:.6f}'.format(
                    epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
                    DiceB_train/len(data_loader_train),voe_score_train/len(data_loader_train)))
    return 'Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f}'.format(
                    epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
                    DiceB_train/len(data_loader_train))



def valid(model, criterion, data_loader_valid, epoch):
    model.eval()
    softmax = nn.Softmax()
    Diceloss = computeDiceOneHot()
    softmax.to(torch.device("cuda"))
    Diceloss.to(torch.device("cuda"))
    dice = 0.
    dice1 = 0.
    dice1_mean = 0.
    voe_score_valid=0.
    with torch.no_grad():
        for batch_idx, (ct_image, pet_image, target) in enumerate(data_loader_valid):
            ct_image, pet_image, target = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")), \
                                         target.to(torch.device("cuda"))
            segmentation_prediction = model(ct_image, pet_image)
            pred_y = softmax(segmentation_prediction)
            Segmentation_planes = getOneHotSegmentation(target)
            segmentation_prediction_ones = predToSegmentation(pred_y)
            DicesN, Dices1 ,voe= Diceloss(segmentation_prediction_ones, Segmentation_planes)
            Dice1 = DicesToDice(Dices1)
            dice_score = (Dice1) / 2
            dice += dice_score.item()
            dice1 += Dice1.item()
            voe_score_valid+=voe.item()


    # 等待所有进程完成计算
    dist.barrier()

    # 使用分布式数据并行计算模型在验证集上的结果
    dice1_tensor = torch.tensor((dist.get_rank(), dice1 / len(data_loader_valid), dice1_mean / len(data_loader_valid), voe_score_valid/len(data_loader_valid))).to(torch.device("cuda"))
    gathered_dice1 = [torch.zeros_like(dice1_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_dice1, dice1_tensor)
    dice1_list = [(d[0].item(), d[1].item(), d[2].item(), d[3].item()) for d in gathered_dice1]

    # 计算所有进程的dice1平均值和dice1_mean的平均值
    dice1_avg = sum([d[1] for d in dice1_list]) / len(dice1_list)
    dice1_mean_avg = sum([d[2] for d in dice1_list]) / len(dice1_list)
    voe_avg = sum([d[3] for d in dice1_list]) / len(dice1_list)

    # 输出所有进程的dice和dice1平均值
    all_rank_dice1 = [d[1] for d in dice1_list]
    all_rank_voe= [d[3] for d in dice1_list]
    print('Valid Epoch: {}, All_Rank_DICE1: {}'.format(epoch, all_rank_dice1))
    print('Valid Epoch: {}, All_Rank_Avg_DICE1: {:.6f}'.format(epoch, dice1_avg))
    print('Valid Epoch: {}, All_Rank_Mean_DICE: {:.6f}'.format(epoch, dice1_mean_avg))

    return epoch, all_rank_dice1,dice/len(data_loader_valid), dice1/len(data_loader_valid),all_rank_voe,voe_avg
    
    

def valid2(model, crierion, data_loader_valid, epoch):
    model.eval()
    softmax = nn.Softmax()
    Diceloss = computeDiceOneHot()
    softmax.to(torch.device("cuda"))
    Diceloss.to(torch.device("cuda"))
    dice = 0.
    dice1 = 0.
    voe_score_valid=0.
    # dice2 = 0.
    # dice3 = 0.
    # dice4 = 0.
    with torch.no_grad():
        for batch_idx, (ct_image, pet_image, target) in enumerate(data_loader_valid):
            ct_image, pet_image, target = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")), \
                                         target.to(torch.device("cuda"))
            segmentation_prediction = model(ct_image, pet_image)
            pred_y = softmax(segmentation_prediction)
            Segmentation_planes = getOneHotSegmentation(target)
            segmentation_prediction_ones = predToSegmentation(pred_y)
            # 分5类
            # DicesN, Dices1, Dices2, Dices3, Dices4 = Diceloss(segmentation_prediction_ones, Segmentation_planes)
            # 分2类
            DicesN, Dices1,voe = Diceloss(segmentation_prediction_ones, Segmentation_planes)
            Dice1 = DicesToDice(Dices1)
            # Dice2 = DicesToDice(Dices2)
            # Dice3 = DicesToDice(Dices3)
            # Dice4 = DicesToDice(Dices4)
            # 分5类
            # dice_score = (Dice1 + Dice2 + Dice3 + Dice4) / 4
            # 分2类
            # dice_score = (Dice1 + Dice2) / 2
            # dice += dice_score.item()
            # dice1 += Dice1.item()
            # dice2 += Dice2.item()
            # 2分类
            dice_score = (Dice1) / 2
            dice += dice_score.item()
            dice1 += Dice1.item()
            voe_score_valid+=voe.item()
            # dice3 += Dice3.item()
            # dice4 += Dice4.item()
            # 分5类
            # if (batch_idx + 1) / (len(data_loader_valid)) == 1:
            #     print('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f}, Dice2: {:.6f}, Dice3: {:.6f}, Dice4: {:.6f}'.format(
            #         epoch, dice / len(data_loader_valid), dice1 / len(data_loader_valid),
            #         dice2 / len(data_loader_valid), dice3 / len(data_loader_valid), dice4 / len(data_loader_valid)))
            # 分2类
            # if (batch_idx + 1) / (len(data_loader_valid)) == 1:
            #     print('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f}, Dice2: {:.6f}'.format(
            #         epoch, dice / len(data_loader_valid), dice1 / len(data_loader_valid),
            #         dice2 / len(data_loader_valid)))
            # 2分类
            if (batch_idx + 1) / (len(data_loader_valid)) == 1:
                print('Valid Epoch: {}, Mean_Dice: {:.6f}, Dice1: {:.6f}, voe: {:.6f}'.format(
                    epoch, dice / len(data_loader_valid), dice1 / len(data_loader_valid),voe_score_valid / len(data_loader_valid)))

    return epoch, dice/len(data_loader_valid), dice1/len(data_loader_valid),voe_score_valid/len(data_loader_valid),



