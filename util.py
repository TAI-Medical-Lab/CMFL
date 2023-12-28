import torch
import torch.nn as nn
from torch.autograd import Variable
# from progressBar import printProgressBar
from torch.optim import lr_scheduler

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()
    
    def topicture(self, pred_mask):
    # 把概率矩阵转化为二值图像（0，1）
        pred_mask=to_var(pred_mask)
        pred_mask = torch.where(torch.tensor(pred_mask) >= 0.5, torch.tensor(1), torch.tensor(0))
        
        return pred_mask
    
    def calculate_voe(self, pred_mask, true_mask):
        """
        计算预测分割结果和实际分割结果的体积重叠度（VOE）
        :param pred_mask: 预测分割结果，二值化的tensor，0表示背景，1表示前景
        :param true_mask: 实际分割结果，二值化的tensor，0表示背景，1表示前景
        :return: VOE指标
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        pred_mask=to_var(pred_mask)
        true_mask=to_var(true_mask)

        pred_mask = self.topicture(pred_mask)
        # pred_mask = torch.Tensor(pred_mask).float()
        # true_mask = torch.Tensor(true_mask).float()

        intersection = torch.logical_and(pred_mask, true_mask)
        union = torch.logical_or(pred_mask, true_mask)
        voe = 1 - ((torch.sum(intersection) + 1e-8) / (torch.sum(union)) + 1e-8)
        return voe

    


    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        # DiceW = to_var(torch.zeros(batchsize, 2))
        # DiceT = to_var(torch.zeros(batchsize, 2))
        # DiceZ = to_var(torch.zeros(batchsize, 2))

        Voe = to_var(torch.zeros(batchsize, 1))
        

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            # DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            # DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])
            # DiceZ[i, 0] = self.inter(pred[i, 4], GT[i, 4])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            # DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            # DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])
            # DiceZ[i, 1] = self.sum(pred[i, 4], GT[i, 4])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            pred[i,1]=pred[i,1].to(device)
            GT[i,1]=GT[i,1].to(device)
            Voe[i,0]=self.calculate_voe(pred[i,1],GT[i,1])


        return DiceN, DiceB,Voe.mean()


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    num_classes = 2
    Val = to_var(torch.zeros(num_classes))

    # Chaos MRI
    # Val[1] = 0.24705882
    # Val[2] = 0.49411765
    # Val[3] = 0.7411765
    # Val[4] = 0.9882353
    # HVSMR
    # Val[1] = 0.49803922
    # Val[2] = 1.0

    Val[1] = 1.0

    x = predToSegmentation(pred)

    out = x * Val.view(1, 2, 1, 1)

    return out.sum(dim=1, keepdim=True)

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotSegmentation(batch):
    backgroundVal = 0

    # Chaos MRI (These values are to set label values as 0,1,2,3 and 4)
    # label1 = 0.24705882
    # label2 = 0.49411765
    # label3 = 0.7411765
    # label4 = 0.9882353
    # HVSMR MRI (These values are to set label values as 0,1,2)
    # label1 = 0.49803922
    # label2 = 1.0
    label1 = 1.0
    oneHotLabels = torch.cat(
        (batch == backgroundVal, batch == label1),
        dim=1)

    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 1.0  # for Chaos MRI  Dataset this value
    return (batch / denom).round().long().squeeze()


# def inference(net, img_batch):
#     total = len(img_batch)
#
#     Dice1 = torch.zeros(total, 2)
#     Dice2 = torch.zeros(total, 2)
#     # Dice3 = torch.zeros(total, 2)
#     # Dice4 = torch.zeros(total, 2)
#
#     net.eval()
#     img_names_ALL = []
#
#     dice = computeDiceOneHot().cuda()
#     softMax = nn.Softmax().cuda()
#     for i, data in enumerate(img_batch):
#         printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
#         image, labels, img_names = data
#         img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])
#
#         MRI = to_var(image)
#         Segmentation = to_var(labels)
#
#         segmentation_prediction = net(MRI)
#
#         pred_y = softMax(segmentation_prediction)
#         Segmentation_planes = getOneHotSegmentation(Segmentation)
#
#         segmentation_prediction_ones = predToSegmentation(pred_y)
#         DicesN, Dices1, Dices2= dice(segmentation_prediction_ones, Segmentation_planes)
#
#         Dice1[i] = Dices1.data
#         Dice2[i] = Dices2.data
#         # Dice3[i] = Dices3.data
#         # Dice4[i] = Dices4.data
#
#     printProgressBar(total, total, done="[Inference] Segmentation Done !")
#
#     ValDice1 = DicesToDice(Dice1)
#     ValDice2 = DicesToDice(Dice2)
#     # ValDice3 = DicesToDice(Dice3)
#     # ValDice4 = DicesToDice(Dice4)
#
#     return [ValDice1, ValDice2]


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler