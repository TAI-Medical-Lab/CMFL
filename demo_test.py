import os
import torch
from dataset_segmentation import MedicalImageDataset

import torchvision.transforms as transforms

import torch.nn as nn
from unet_segmentation import Unet
from config_segmentation import config
from utils import computeDiceOneHot, getOneHotSegmentation, getTargetSegmentation, predToSegmentation, DicesToDice, getSingleImage
from torch.nn import init
import numpy as np

def init_weights(net, init_type='normal', gain=0.02):
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

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class MIOU_Metric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def MIoU(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection/union
        mIoU = np.nanmean(IoU)
        return IoU, mIoU

    def genConfusionMatrix(self, pred, label):
        mask = (label >= 0) & (label < self.numClass)
        truth = self.numClass * label[mask] + pred[mask]
        count = np.bincount(truth.astype(int), minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, pred, label):
        assert pred.shape == label.shape
        self.confusionMatrix += self.genConfusionMatrix(pred, label)

color_data = torch.zeros(3,3)
# color_data[0] = torch.tensor([0, 199, 140]) #土耳其色
color_data[2] = torch.tensor([255, 255, 0]) #黄
# color_data[2] = torch.tensor([138, 43, 226]) #紫罗兰
# color_data[3] = torch.tensor([64, 224, 205]) #青绿色
color_data[1] = torch.tensor([0, 255, 0]) #绿色
color_data[0] = torch.tensor([255, 0, 0]) #红色

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
opx = config()
opt = opx()
try:
    os.makedirs(opt.out)
except OSError:
    pass


datasets = MedicalImageDataset("test", opt.root,
                                       mask_transform=transforms.ToTensor(), augment=False, equalize=False)
dataloader = torch.utils.data.DataLoader(datasets, batch_size=1,
                                             shuffle=False)
model = Unet(2, 2)
#init_weights(model,'xavier', 0.02)


opt.modelWeights = "" # 模型文件的地址
checkpoint = torch.load(opt.modelWeights)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(torch.device("cuda"))

model.eval()
softmax = nn.Softmax()
Diceloss = computeDiceOneHot()
softmax.to(torch.device("cuda"))
Diceloss.to(torch.device("cuda"))


# 5054为总体测试图像数量
Dice1 = torch.zeros(5054, 2)# 157 CHAOS 5054
Pred = torch.zeros(942, 1, 256, 256)
Target = torch.zeros(942, 1, 256, 256)



with torch.no_grad():
    for batch_idx, (ct_image, pet_image, label) in enumerate(dataloader):
        ct_image, pet_image, label = ct_image.to(torch.device("cuda")), pet_image.to(torch.device("cuda")), \
                                     label.to(torch.device("cuda"))
        segmentation_prediction = model(ct_image, pet_image)
        predclass_y = softmax(segmentation_prediction)
        output = getSingleImage(predclass_y)
        output_new = output.squeeze(dim=0)

        # 该部分为显示分割结果。
        # output_new1 = output_new/0.24705885 #这是除以最小的类别。
        # output_new1 = output_new1.round().cpu()
        # r = torch.zeros(1, output_new1.shape[1], output_new1.shape[2])
        # g = torch.zeros(1, output_new1.shape[1], output_new1.shape[2])
        # b = torch.zeros(1, output_new1.shape[1], output_new1.shape[2])
        # 根据不同的类别给不同类上不同颜色。
        # for l in range(1,7):
        #     r[output_new1 ==l] = color_data[l-1, 0]
        #     g[output_new1 ==l] = color_data[l-1, 1]
        #     b[output_new1 ==l] = color_data[l-1, 2]
        # rgb = torch.zeros(3, output_new1.shape[1], output_new1.shape[2])
        # rgb[0, :, :] = r/255
        # rgb[1, :, :] = g/255
        # rgb[2, :, :] = b/255
        # transforms_new = transforms.ToPILImage()
        # image = transforms_new(rgb)
        # image.save("TEST/StructSeg/model_unet_attention_on_concat_multiloss_no_augment/{}.png".format(batch_idx+1))
        Segmentation_planes = getOneHotSegmentation(label)
        segmentation_prediction_ones = predToSegmentation(predclass_y)
        DicesN, Dices1 = Diceloss(segmentation_prediction_ones, Segmentation_planes)
        Dice1[batch_idx] = Dices1.data
        pred = output_new / 1.0  # chaos /0.24705882
        Pred[batch_idx] = pred
        target_origin = label * 255 / 255  # chaos /63
        target_origin = target_origin.squeeze(dim=0)
        Target[batch_idx] = target_origin

    pred = Pred.round().cpu().numpy()
    label = Target.round().cpu().numpy()
    Metric = MIOU_Metric(2)
    Metric.addBatch(pred, label)
    IoU, _ = Metric.MIoU()

    # MIoU = (IoU[1] + IoU[2] + IoU[3] + IoU[4] + IoU[5] + IoU[6]) / 6 #这是多分类
    MIoU = IoU[1]
    ValDice1 = DicesToDice(Dice1)

    # ValDice = (ValDice1 + ValDice2 + ValDice3 + ValDice4 + ValDice5 + ValDice6) / 6
    ValDice = ValDice1
    print('Mean_Dice: {:.6f}, Dice1: {:.6f}'.format(ValDice, ValDice1))
    print('Mean_IoU: {:.6f}, IoU1: {:.6f}'.format(MIoU, IoU[1]))







