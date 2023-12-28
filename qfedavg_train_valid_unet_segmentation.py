import torch
import torch.nn as nn
from utils import computeDiceOneHot, getOneHotSegmentation, getTargetSegmentation, predToSegmentation, DicesToDice
import torch.distributed as dist
from config_segmentation import config
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
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


def norm_params(model):
    # Get flattened gradients for current model
    flat_params = torch.cat([param.grad.view(-1) for param in model.parameters()])

    world_size = 4

    # Get flattened gradients from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)

    # Compute the L2 norm of flattened gradients from all processes
    squared_l2_norms = torch.tensor([torch.sum(torch.square(param)) for param in all_flat_params_list])
    l2_norms = torch.sqrt(squared_l2_norms)
    # Normalize L2 norms
    l2_norms_normalized = l2_norms / torch.sum(l2_norms)

    return l2_norms_normalized

def qfedavg_weight(optimizer, loss, norm, max_loss):
    q = 0.5
    lr = optimizer.param_groups[0]['lr']
    normalized_loss = torch.tensor(loss) / torch.tensor(max_loss)
    return (q * torch.float_power(normalized_loss + 1e-10, (q - 1)) * torch.tensor(norm) + (1.0 / lr) * torch.float_power(normalized_loss + 1e-10, q)).item()

def qfedavg_aggregate_weights(model, weights):
    world_size = 4
    # Get flattened parameters for current model
    flat_params = torch.cat([param.data.clone().view(-1) for param in model.parameters()])
    
    # Get flattened parameters from all processes
    all_flat_params_list = [torch.zeros_like(flat_params) for _ in range(world_size)]
    dist.all_gather(all_flat_params_list, flat_params)
    

    # Compute the weighted average of parameters
    weighted_avg_params = torch.stack([weight * params for weight, params in zip(weights, all_flat_params_list)], dim=0).sum(dim=0)
    
    # Update the model with the weighted average parameters
    index = 0
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(weighted_avg_params[index:index + numel].view(param.shape))
        index += numel


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
    print(type(data_loader_train))
    
    for batch_idx,(ct_image, pet_image, label) in tqdm(enumerate(data_loader_train), total=len(data_loader_train)):
        i = i + 1
        if i != len(data_loader_train):
            ct_image, pet_image, label = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")),\
                                        label.to(torch.device("cuda"))

            optimizer.zero_grad()
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
            Dice_score = (DiceB) / 2
            DiceB_train += DiceB.item()
            # DiceW_train += DiceW.item()
            # DiceT_train += DiceT.item()
            # DiceZ_train += DiceZ.item()
            Dice_score_train += Dice_score.item()

            voe_score_train+=voe.item()
            
            loss.backward()
            optimizer.step()
            
            if opt.setting == 'FL':
                average_weights(model)
            
            elif opt.setting == 'QFFedAvg':
                q=0.5
                weights = [0.0 for _ in range(4)]
                losses = [0.0 for _ in range(4)]

                l2_norms = norm_params(model)
                norm = l2_norms[dist.get_rank()].item()
                loss = loss.item()
                losses[dist.get_rank()] = loss

                # Gather all losses
                all_losses_tensor = torch.tensor(losses)
                gathered_losses = [torch.zeros_like(all_losses_tensor) for _ in range(4)]
                dist.all_gather(gathered_losses, all_losses_tensor)

                # Convert gathered_losses to a flat list of scalar values
                gathered_loss_values = [loss.item() for gathered_loss in gathered_losses for loss in gathered_loss]

                # Calculate the maximum loss across all processes
                max_loss_tensor = torch.tensor(max(gathered_loss_values))
                max_loss_tensor_list = [torch.zeros_like(max_loss_tensor) for _ in range(4)]
                dist.all_gather(max_loss_tensor_list, max_loss_tensor)
                max_loss = max([x.item() for x in max_loss_tensor_list])

                weight = qfedavg_weight(optimizer, loss, norm, max_loss)
                weights[dist.get_rank()] = weight

                # 将4个进程对应的长度为4的weights，汇总成1个长度为4的权重数组，其中最后的权重数组对应的权重分别对应着进程号 0 1 2 3
                # 汇总权重数组
                all_weights_tensor = torch.tensor(weights)
                gathered_weights = [torch.zeros_like(all_weights_tensor) for _ in range(4)]
                dist.all_gather(gathered_weights, all_weights_tensor)

                # 将4个进程对应的长度为4的weights，汇总成1个长度为4的权重数组
                weights = [0.0 for _ in range(4)]
                for i, gathered_weight in enumerate(gathered_weights):
                    weights[i] = gathered_weight[i]

                # 计算所有进程对应的权重之和
                weight = weight / sum(weights)

                weight1 = torch.tensor(weight)
                weight2 = [torch.zeros_like(weight1) for _ in range(4)]
                dist.all_gather(weight2, weight1)

                qfedavg_aggregate_weights(model, weight2)

            # 2分类
            if (batch_idx+1)/(len(data_loader_train)) == 1:
                print('Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f},voe:{:.6f}'.format(
                    epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
                    DiceB_train/len(data_loader_train),voe_score_train/len(data_loader_train)))
                
                
    return 'Train Epoch: {}, Loss: {:.6f}, Mean_Dice: {:.6f}, Dice1: {:.6f},voe:{:.6f}'.format(
                epoch, loss_train/len(data_loader_train), Dice_score_train/len(data_loader_train),
                DiceB_train/len(data_loader_train),voe_score_train/len(data_loader_train))

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
    

