import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config

import cv2
import SimpleITK as sitk
import numpy as np
import pydicom
import torch

# 归一化

# 融合CT数据与PET数据
def read_itk_image(dcm_dir):
    # 在终端执行 python test.py 'NA-15574' 'NA-32516' '300.000000-Segmentation-38310'
    # 其中NA-15574表ct数据，NA32516表pet数据，300.000000-Segmentation-38310表标记后的petct
    #
    # dcm_dir = 'medicine-data/300.000000-Segmentation-38310'

    reader = sitk.ImageSeriesReader()
    # seriesIDs = reader.GetGDCMSeriesIDs(dcm_dir)
    dcm_series = reader.GetGDCMSeriesFileNames(dcm_dir)
    reader.SetFileNames(dcm_series)
    itk_img = reader.Execute()
    itk_img_array = sitk.GetArrayFromImage(itk_img)
    print("img_array",itk_img_array.shape)
    return itk_img
def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled
def Fusion(path_CT, path_PET,fusion_data_path,fusion_rate):
    # 读取并转成数组形式
    # 在终端执行 python test.py 'NA-15574' 'NA-32516' '300.000000-Segmentation-38310'
    # 其中NA-15574表ct数据，NA32516表pet数据，300.000000-Segmentation-38310表标记后的petct
    print("CT_shape:")
    image_CT_ITK  = read_itk_image(path_CT)#read a series of dcms
    image_CT = sitk.GetArrayFromImage(image_CT_ITK)
    image_CT = image_CT.astype(float)
    image_CT_size = np.shape(image_CT)
    # 归一化 CT
    # image_CT = MaxMinNormalizer(image_CT)
    print("PET_shape:")
    input_PET_property  = read_itk_image(path_PET)
    image_PET = resize_image_itk(input_PET_property,image_CT_ITK)
    image_PET = sitk.GetArrayFromImage(image_PET)
    image_PET = image_PET.astype(float)

    # image_PET_size = np.shape(image_PET)
    # 归一化 CT
    # image_PET = MaxMinNormalizer(image_PET)
    #  融合比例
    # percent_list = []
    # for i in range(1, 10):
        # percent_list.append(i / 10)
    # fusion CT , PET,
    # percent_list=[0.1] # best performance
    # for percent in percent_list
    ImageFusion = np.zeros(image_CT_size)
    for num in range(image_CT_size[2]):
        image_CT_slice = image_CT[:, :, num]
        image_PET_slice = image_PET[:, :, num]
        # 按照比例融合数据
        img_mix = cv2.addWeighted(image_CT_slice, fusion_rate, image_PET_slice, 1 - fusion_rate, 0)
        ImageFusion[:, :, num] = img_mix
    # ImageFusion = np.transpose(ImageFusion, (2, 1, 0))
    print("ImageFusion",ImageFusion.shape)
    ImageFusionISO = sitk.GetImageFromArray(ImageFusion, isVector=False)

    # 获取图像扫描及空间信息
    # input_PET_property = sitk.ReadImage(path_PET)
    spacing = np.array(input_PET_property.GetSpacing())
    direction = np.array(input_PET_property.GetDirection())
    Origin = np.array(input_PET_property.GetOrigin())

    # 重新把融合后的数据设置图像信息
    ImageFusionISO.SetSpacing(spacing)
    ImageFusionISO.SetOrigin(Origin)
    ImageFusionISO.SetDirection(direction)
    # print("ImageFusionISO", ImageFusionISO)
    print("...successfully fusing...")
    return ImageFusionISO
    # return ImageFusionISO
    # pass
def dcm2nii(dcms_path):
    image = sitk.ReadImage(dcms_path)  # 读取一个含有头信息的dicom格式的医学图像
    sitk.WriteImage(image2, 'medicine-data/CT-PET.dcm')  # 把插入过头信息的图像输出为dicom格式
    return image3

def nii2dcm():
    dicom_path = 'medicine-data/NA-15574/1-001.dcm'
    nii_path = 'medicine-data/CT-PET-Fusioin.nii.gz'

    image = sitk.ReadImage(dicom_path)  # 读取一个含有头信息的dicom格式的医学图像
    keys = image.GetMetaDataKeys()  # 获取它的头信息
    image2 = sitk.ReadImage(nii_path)  # 读取要转换格式的图像
    for key in keys:
        image2.SetMetaData(key, image.GetMetaData(key))  # 把第一张图像的头信息，插入到第二张图像
    sitk.WriteImage(image2, 'medicine-data/CT-PET.dcm')  # 把插入过头信息的图像输出为dicom格式
    # return
def visual(example_filename):
    import matplotlib
    # matplotlib.use('TkAgg')
    import nibabel as nib
    from nibabel.viewers import OrthoSlicer3D
    # example_filename = 'medicine-data/CT-PET-Fusioin.nii'
    img = nib.load(example_filename)
    # print(img)
    # print(img.header['db_name'])  # 输出头信息
    # width, height, queue, C = img.dataobj.shape
    img = img.get_data().squeeze()
    OrthoSlicer3D(img).show()

class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale
        self.args = args
        self.valid_rate = args.valid_rate

    def get_start_and_end(self,seg_array, chosen):
        z = np.any(seg_array, axis=(1, 2))  # 获取z的范围

        print("or_z",z)
        y = np.any(seg_array, axis=(0, 2))  # 获取y的范围
        x = np.any(seg_array, axis=(0, 1))  # 获取x的范围
        print('x   ',x)
        print('y  ',y)
        if (z == 0).all() or (x == 0).all() or (y == 0).all():
            xend_slice=xstart_slice=0
            return xstart_slice, xend_slice

        else:
            zstart_slice, zend_slice = np.where(z)[0][[0, -1]]
            ystart_slice, yend_slice = np.where(y)[0][[0, -1]]
            xstart_slice, xend_slice = np.where(x)[0][[0, -1]]
            # print("xstart_slice, xend_slice", xstart_slice, xend_slice,xend_slice-xstart_slice)
            # print("ystart_slice, yend_slice", ystart_slice, yend_slice,yend_slice - ystart_slice)
            # print("zstart_slice, zend_slice", zstart_slice, zend_slice,zend_slice - zstart_slice)
            max_XYZ = [xend_slice - xstart_slice, yend_slice - ystart_slice, zend_slice - zstart_slice]

            xymax = max(max_XYZ[0], max_XYZ[1])
            xyup_bound = seg_array.shape[1]
            xy_size = self.xy_down_scale
            z_size = self.slice_down_scale
            print("xymax>xysize",xymax,xy_size)
            if chosen == "x":
                rema = (xy_size - max_XYZ[0]) % 2
                inte = (xy_size - max_XYZ[0]) // 2
                xlow_exp = -inte
                xhig_exp = rema + inte

                xstart_slice = xstart_slice
                xend_slice = xend_slice
            if chosen == "y":
                rema = (xy_size - max_XYZ[1]) % 2
                inte = (xy_size - max_XYZ[1]) // 2
                xlow_exp = -inte
                xhig_exp = rema + inte

                xstart_slice = ystart_slice
                xend_slice = yend_slice
            if chosen == "z":
                xyup_bound = seg_array.shape[0]
                rema = (z_size - max_XYZ[2]) % 2
                inte = (z_size - max_XYZ[2]) // 2
                xlow_exp = -inte
                xhig_exp = rema + inte
                xstart_slice = zstart_slice
                xend_slice = zend_slice

            if xstart_slice + xlow_exp < 0:
                xstart_slice = 0
                xhig_exp += xlow_exp - xstart_slice
                xend_slice += xhig_exp

            elif xend_slice + xhig_exp >= xyup_bound - 1:
                xend_slice = xyup_bound - 1
                xlow_exp += -(xyup_bound - 1 - xend_slice)
                xstart_slice += xlow_exp
            else:
                xstart_slice += xlow_exp
                xend_slice += xhig_exp
            return xstart_slice, xend_slice
    def fix_data(self,data_type1,data_type2):
        "函数说明"
        # 从原文件夹中读取data_type1和data_type2数据，并保持到self.fixed_path文件夹下

        print("process ",data_type1,data_type2)
        file_list = os.listdir(join(self.raw_root_path, data_type1))  # 输出该路径下的所有文件和目录名称
        Numbers = len(file_list)
        print('Total numbers of samples is :', Numbers)  # 输出该目录下所以文件的数量，相当于所有ct数据数
        for ct_file, i in zip(file_list, range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i + 1, Numbers))
            ct_path = os.path.join(self.raw_root_path, data_type1, ct_file)
            ctseg_path = os.path.join(self.raw_root_path, data_type2, ct_file.replace("X","Seg-Y"))
            print(ctseg_path)
            new_ct, new_seg = self.orignial_process(ct_path, ctseg_path, classes=self.classes)  # 512X512 -> 256X256
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, data_type1, ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, data_type2, ct_file.replace("X","Seg-Y")))

    def ResampleBySitk(self,imgArr,img,new_spacing):
        original_size = img.GetSize()  # 获取图像原始尺寸
        original_spacing = img.GetSpacing()  # 获取图像原始分辨率

        img = sitk.GetImageFromArray(imgArr)
        img.SetDirection(img.GetDirection())
        img.SetOrigin(img.GetOrigin())
        img.SetSpacing(img.GetSpacing())

        new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                    int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                    int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]  # 计算图像在新的分辨率下尺寸大小
        print("new_size",new_size)
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(img.GetDirection())
        resample.SetOutputOrigin(img.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(img.GetPixelIDValue())
        Resampleimage = resample.Execute(img)

        ResampleimageArray = sitk.GetArrayFromImage(Resampleimage)
        return ResampleimageArray

    def centerCrop(self,image, label, output_size):
        if image.shape[0] <= output_size[0] or image.shape[1] <= output_size[1] or image.shape[2] <= output_size[2]:
            pw = max((output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - output_size[0]) / 2.))
        h1 = int(round((h - output_size[1]) / 2.))
        d1 = int(round((d - output_size[2]) / 2.))

        # print(image.shape, output_size, get_center(label), w1, h1, d1)
        image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
        label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

        return image, label
    def ResampleByZoom(self,ct_array, spacing, new_spacing=[1, 1, 1], order=3):
        # 0 denotes remains the same
        # order=0:nearest interpolation
        # order=1:bilinear interpolation
        # order=3:cubic interpolation
        # http://shartoo.github.io/medical_image_process/
        resize_factor = spacing / new_spacing
        new_shape = ct_array.shape * resize_factor

        real_new_shape = np.round(new_shape)
        real_resize_factor = real_new_shape / ct_array.shape
        real_new_spacing = spacing / real_resize_factor

        ct_array = ndimage.zoom(ct_array, real_resize_factor, order, mode='nearest')
        # mode : Points outside the boundaries of the input are filled according to the given mode
        # (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
        return ct_array, real_new_spacing



    def orignial_process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        # Y = torch.flatten(torch.tensor(seg_array))
        # # XX = torch.bincount(X)
        # YY = torch.bincount(torch.abs(Y))
        print("Ori shape:", ct_array.shape, seg_array.shape)
        if classes == 2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            # seg_array[seg_array < 2] = 0
            seg_array[seg_array < 0] = 1
        Or_YY = np.bincount(seg_array.flatten())
        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper
        ct_array[ct_array < self.lower] = self.lower

        # 重采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale)
        new_spacing = [1, 1, 2]  # 设置图像新的分辨率为1*1*1
        ct_array =self.ResampleBySitk(ct_array,ct,new_spacing)
        seg_array = self.ResampleBySitk(seg_array,ct,new_spacing)

        print("resample_size: ",ct_array.shape,seg_array.shape)

        x = self.get_start_and_end(seg_array, "x")#这个函数是截取x轴上标签起止位置的区域，训练时，训练样本你包含标签区域就行了。
        y = self.get_start_and_end(seg_array, "y")#这个函数是截取轴上标签起止位置的区域，训练时，训练样本你包含标签区域就行了。
        z = self.get_start_and_end(seg_array, "z")#这个函数是截取z轴上标签起止位置的区域，训练时，训练样本你包含标签区域就行了。

        print('x',x)

        ct_array = ct_array[z[0]:z[1], x[0]:x[1], y[0]:y[1]] #截取病灶区域
        seg_array = seg_array[z[0]:z[1], x[0]:x[1], y[0]:y[1]]
        print("centerCrop shape:", ct_array.shape, seg_array.shape)
        # print("Preprocessed shape:", ct_array.shape, seg_array.shape)
        # print('seg_array.bincount', YY)
        # print("new_Z",np.bincount(seg_array.flatten()))
        New_YY = np.bincount(seg_array.flatten())
        print(len(New_YY))
        if len(New_YY) !=2:#截取结果区域只存在一类标签，不使用该样本作为训练样本。
            print("Not1!!!!!!!!!")
            new_ct = None
            new_seg = None
        elif New_YY[1].item()< 0.48*Or_YY[1].item():#截取结果区域的病灶标签数量小于原始病灶标签数量，截取效果不好，不使用该样本作为训练样本。，这里可以把条件设置宽泛点，比如0.5*Or_YY[1].item()。要不然训练样本会很少。
            print("Not2!!!!!!!!!")
            new_ct = None
            new_seg = None
        elif z[0]==z[1] or x[0]==x[1] or y[0]==y[1]:
            print("没有标签————————————————————————————")
            new_ct = None
            new_seg = None
        else:
            print("OK")
            # new_ct = sitk.GetImageFromArray(ct_array)
            # new_ct.SetDirection(ct.GetDirection())
            # new_ct.SetOrigin(ct.GetOrigin())
            # new_ct.SetSpacing(new_spacing)
            #
            # new_seg = sitk.GetImageFromArray(seg_array)
            # new_seg.SetDirection(ct.GetDirection())
            # new_seg.SetOrigin(ct.GetOrigin())
            # new_seg.SetSpacing(new_spacing)
            #
            new_ct = sitk.GetImageFromArray(ct_array)
            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing(ct.GetSpacing())

            new_seg = sitk.GetImageFromArray(seg_array)
            new_seg.SetDirection(ct.GetDirection())
            new_seg.SetOrigin(ct.GetOrigin())
            new_seg.SetSpacing(ct.GetSpacing())

        # new_ct = sitk.GetImageFromArray(ct_array)
        # new_ct.SetDirection(ct.GetDirection())
        # new_ct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing(new_spacing)
        #
        # new_seg = sitk.GetImageFromArray(seg_array)
        # new_seg.SetDirection(ct.GetDirection())
        # new_seg.SetOrigin(ct.GetOrigin())
        # new_seg.SetSpacing(new_spacing)

        # new_ct = sitk.GetImageFromArray(ct_array)
        # new_ct.SetDirection(ct.GetDirection())
        # new_ct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing(ct.GetSpacing())
        #
        # new_seg = sitk.GetImageFromArray(seg_array)
        # new_seg.SetDirection(ct.GetDirection())
        # new_seg.SetOrigin(ct.GetOrigin())
        # new_seg.SetSpacing(ct.GetSpacing())
        # # 保存为对应的格式
        # new_ct = sitk.GetImageFromArray(ct_array)
        # new_ct.SetDirection(ct.GetDirection())
        # new_ct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
        #                    ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        #
        # new_seg = sitk.GetImageFromArray(seg_array)
        # new_seg.SetDirection(ct.GetDirection())
        # new_seg.SetOrigin(ct.GetOrigin())
        # new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
        #                     ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_ct, new_seg
    def process(self, ct_path, seg_path, classes=None):
        # print("ct_path",ct_path)
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)  # 读取nii数据
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        # X = torch.flatten(torch.tensor(ct_array))
        Y = torch.flatten(torch.tensor(seg_array))
        # XX = torch.bincount(X)
        YY = torch.bincount(torch.abs(Y))
        # print("X",X[:50])
        # print('Y',YY)
        # X_sorted = torch.topk(X, k=10, dim=0, largest=True, sorted=True)
        # Y_sorted = torch.topk(Y, k=10, dim=0, largest=False, sorted=True)
        # print(Y_sorted)
        # X_sorted_inc = torch.topk(X, k=10, dim=0, largest=False, sorted=True)
        # X_sorted_des = torch.topk(X, k=10, dim=0, largest=True, sorted=True)
        # print("X_sorted_inc",X_sorted_inc)
        # print("X_sorted_des",X_sorted_des)
        print("Ori shape:", ct_array.shape, seg_array.shape)
        if classes == 2:
            # 将金标准中肝脏和肝肿瘤的标签融合为一个
            print("relabel")
            # seg_array[seg_array == 0] = 1
            seg_array[seg_array <0] = 1 #原数据金标准是-1，其它为0
        Or_YY = np.bincount(seg_array.flatten())


        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > self.upper] = self.upper #设置1500才不会导致图片过于模糊
        ct_array[ct_array < self.lower] = self.lower
        # X_sorted_inc = torch.topk(torch.flatten(torch.tensor(ct_array)), k=10, dim=0, largest=False, sorted=True)
        # print("X_sorted_inc",X_sorted_inc)
        # X_sorted_des = torch.topk(torch.flatten(torch.tensor(ct_array)), k=10, dim=0, largest=True, sorted=True)
        # print("X_sorted_dec",X_sorted_des)
        # X_sorted_des = torch.topk(X, k=10, dim=0, largest=True, sorted=True)
        # print("ct.GetSpacing()[-1]",ct.GetSpacing()[-1])
        x = self.get_start_and_end(seg_array, "x")#这个函数是截取x轴上标签起止位置的区域，训练时，训练样本你包含标签区域就行了。
        y = self.get_start_and_end(seg_array, "y")#这个函数是截取轴上标签起止位置的区域，训练时，训练样本你包含标签区域就行了。
        z = self.get_start_and_end(seg_array, "z")#这个函数是截取z轴上标签起止位置的区域，训练时，训练样本你包含标签区域就行了。

        ct_array = ct_array[z[0]:z[1], x[0]:x[1], y[0]:y[1]] #截取病灶区域
        seg_array = seg_array[z[0]:z[1], x[0]:x[1], y[0]:y[1]]


        print("Preprocessed shape:", ct_array.shape, seg_array.shape)
        print('seg_array.bincount', YY)
        print("new_Z",np.bincount(seg_array.flatten()))
        New_YY = np.bincount(seg_array.flatten())
        print(len(New_YY))
        if len(New_YY) !=2:#截取结果区域只存在一类标签，不使用该样本作为训练样本。
            print("Not1!!!!!!!!!")
            new_ct = None
            new_seg = None
        elif New_YY[1].item()< 0.5*Or_YY[1].item():#截取结果区域的病灶标签数量小于原始病灶标签数量，截取效果不好，不使用该样本作为训练样本。，这里可以把条件设置宽泛点，比如0.5*Or_YY[1].item()。要不然训练样本会很少。
                print("Not2!!!!!!!!!")
                new_ct = None
                new_seg = None
        else:
            print("OK")
            new_ct = sitk.GetImageFromArray(ct_array)
            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing(ct.GetSpacing())

            new_seg = sitk.GetImageFromArray(seg_array)
            new_seg.SetDirection(ct.GetDirection())
            new_seg.SetOrigin(ct.GetOrigin())
            new_seg.SetSpacing(ct.GetSpacing())

        # if len(New_YY) != 2:
        #     print("Not1!!!!!!!!!")
        #     new_ct = None
        #     new_seg = None
        #
        # else:
        #     print("OK")
        #     new_ct = sitk.GetImageFromArray(ct_array)
        #     new_ct.SetDirection(ct.GetDirection())
        #     new_ct.SetOrigin(ct.GetOrigin())
        #     new_ct.SetSpacing(ct.GetSpacing())
        #
        #     new_seg = sitk.GetImageFromArray(seg_array)
        #     new_seg.SetDirection(ct.GetDirection())
        #     new_seg.SetOrigin(ct.GetOrigin())
        #     new_seg.SetSpacing(ct.GetSpacing())


        # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        # ct_array = ndimage.zoom(ct_array,
        #                         (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
        #                         order=3)
        # seg_array = ndimage.zoom(seg_array,
        #                          (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
        #                          order=0)
        # print("zoomct_array",ct_array.shape,seg_array.shape)


        # 保存为对应的格式
        # new_ct = sitk.GetImageFromArray(ct_array)
        # new_ct.SetDirection(ct.GetDirection())
        # new_ct.SetOrigin(ct.GetOrigin())
        # new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
        #                    ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        #
        # new_seg = sitk.GetImageFromArray(seg_array)
        # new_seg.SetDirection(ct.GetDirection())
        # new_seg.SetOrigin(ct.GetOrigin())
        # new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
        #                     ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))

        # 找到病灶区域开始和结束，并各向外扩张


        # if new_ct != None and new_seg != None:
        #     sitk.WriteImage(new_ct,
        #                     os.path.join(self.fixed_path,"fusionX.nii"))
        #     sitk.WriteImage(new_seg,
        #                     os.path.join(self.fixed_path,"fusionY.nii"))
        # visual(os.path.join(self.fixed_path,"fusionX.nii"))#可视化融合的pet-ct图像

        return new_ct, new_seg
    def write_train_val_name_list(self,data_type1,data_type2):
        # data_name_list = os.listdir(join(self.fixed_path, "pet-ct-image"))
        data_name_list = os.listdir(join(self.fixed_path, data_type1))
        data_num = len(data_name_list)
        print('the fixed dataset total numbers of samples is :', data_num)
        random.shuffle(data_name_list)

        assert self.valid_rate < 1.0
        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]
        val_name_list = data_name_list[
                        int(data_num * (1 - self.valid_rate)):int(data_num * ((1 - self.valid_rate) + self.valid_rate))]

        self.write_name_list(train_name_list, str(data_type1) + "_train_path_list.txt",data_type1,data_type2)
        self.write_name_list(val_name_list, str(data_type1) + "_val_path_list.txt",data_type1,data_type2)

    def write_name_list(self, name_list, file_name,data_type1,data_type2):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, data_type1, name)
            seg_path = os.path.join(self.fixed_path, data_type2, name.replace("1.2X","Seg-Y"))
            # ct_path = os.path.join(self.fixed_path, 'pet-ct-image', name)
            # seg_path = os.path.join(self.fixed_path, 'pet-ct-seg', name.replace("1.2Fusioin-X","Seg-Y"))
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()

def load_ct_and_pet_data(root_path,saved_fusion_data_path):
    file_list = os.listdir(root_path)  # 输出该路径下的所有文件和目录名称:HNXXX1,HNXXX2,HNXXX3,...,HNXXX
    # print(len(file_list))
    if 'LICENSE' in file_list:
        file_list.remove("LICENSE")
    # print(file_list)
    for i,file_name in enumerate(file_list):
        # if i == 20:break
        path1 = os.path.join(root_path, file_name)#"./HEAD-NECK-RADIOMICS-HN1/HNXXX/"
        file_name1 = os.listdir(path1)# 输出该路径下的所有文件和目录名称:...PETCT..
        f_n = file_name1[0]#...NAPETCT..
        print("f_n",f_n)
        if f_n.find("PET-CT") == -1:
            print("PET/CT file could not be found!")
        else:
            print("Found PET/CT file.")
            path2 = os.path.join(path1, f_n)#"./HEAD-NECK-RADIOMICS-HN1/HNXXX/...PETCT../"
            file_name2 = os.listdir(path2)# 输出该路径下的所有文件和目录名称:1.0000-NA-XXXX,300.000-Segmentation,NA-XXXX,NA-XXXX
            collect_CT_PET_serios_path = {}#collect_CT_PET_name[0]对应CT，#collect_CT_PET_name[1]对应PET
            Segmentation_path = None
            modality_count = [0,0]
            print(file_name2)
            if len(file_name2)<3:continue
            for f_n1 in file_name2:
                if 'GK' in f_n1 or 'PET' in f_n1 or 'THA' in f_n1:
                # if f_n1.startswith('NA'):
                    path3 = os.path.join(path2, f_n1)  # "./HEAD-NECK-RADIOMICS-HN1/HNXXX/...PETCT../NA-XXXX"
                    all_dcm_name = os.listdir(path3)# 输出该路径下的所有文件和目录名称:1.001.dcm,1.002.dcm,...,1.00X.dcm,
                    dicm_file = pydicom.dcmread(os.path.join(path3, all_dcm_name[0]))
                    print("dicm_file.Modality1", dicm_file.Modality)
                    if dicm_file.Modality == 'CT':
                        modality_count[0] +=1
                        # print("dicm_file.Modality2", dicm_file.Modality)
                        if modality_count[0] == 1:
                            collect_CT_PET_serios_path[0] = path3
                            print('CT:',path3)
                    else:
                        # print("dicm_file.Modality2", dicm_file.Modality)
                        modality_count[1] += 1
                        if modality_count[1] == 1:
                            collect_CT_PET_serios_path[1] = path3
                            print('PET:',path3)

                if f_n1.find("Segmentation") != -1:#找到金标准
                    print("Found Segmentation")
                    path3 = os.path.join(path2, f_n1)  # "./HEAD-NECK-RADIOMICS-HN1/HNXXX/...PETCT../300.000-Segmentation"
                    Segmentation_path = path3

            print('ct',collect_CT_PET_serios_path[0])
            print('pet',collect_CT_PET_serios_path[1])
            ct_image = read_itk_image(collect_CT_PET_serios_path[0])
            pet_image = read_itk_image(collect_CT_PET_serios_path[1])
            saveCTpath_name = saved_fusion_data_path + "/ct-image" + '/{}.nii'.format(file_name)
            savePETpath_name = saved_fusion_data_path + "/pet-image" + '/{}.nii'.format(file_name)
            sitk.WriteImage(ct_image, saveCTpath_name)#保存ct图像
            sitk.WriteImage(pet_image,savePETpath_name)#保存pet图像

            seg_label = read_itk_image(Segmentation_path)  # 读取ct图像和pet图像共同的标签,
            saved_seg_path = saved_fusion_data_path + "/ct-seg" + "/{}.nii".format(file_name)

            sitk.WriteImage(seg_label, saved_seg_path)#保存为ct的标签

            saved_seg_path = saved_fusion_data_path + "/pet-seg" + "/{}.nii".format(file_name)
            seg_label = read_itk_image(Segmentation_path)
            sitk.WriteImage(seg_label, saved_seg_path) #保存为pet的标签

            print("..Successfully saving seg_label{}.nii...".format(file_name))

if __name__ == '__main__':
    '''需要添加安装包'''
    #pip install opencv-python   这是cv2安装包命令
    #pip install pydicom

    root_path = "./FDG-PET-CT-Lesions" #源数据集，存储着dicom格式的ct,pet，和它们的标签
    raw_dataset_path = "./FDG-PET-CT-Lesions-raw_data/raw_fusion_data"  # 保存未经处理的ct,pet的nii格式图像，一般ct和pet的尺寸不统一,ct:134x512x512,pet 48x128x128
    fixed_dataset_path = './FDG-PET-CT-Lesions-raw_data/fixed_fusion_data'#保存处理过的ct,pet的nii格式图像，得到统一的尺寸 48x128x128

    if os.path.exists(raw_dataset_path) != True:
        # os.makedirs(os.path.join(raw_dataset_path, "pet-ct-image"))
        # os.makedirs(os.path.join(raw_dataset_path, "pet-ct-seg"))

        os.makedirs(os.path.join(raw_dataset_path, "ct-image"))#创建文件夹存放ct图像
        os.makedirs(os.path.join(raw_dataset_path, "ct-seg"))#存放ct图像对应的标签

        os.makedirs(os.path.join(raw_dataset_path, "pet-image"))
        os.makedirs(os.path.join(raw_dataset_path, "pet-seg"))
        # print("oooeee")
    if os.path.exists(fixed_dataset_path) != True:
    # os.makedirs(os.path.join(fixed_dataset_path, "pet-ct-image"))
    # os.makedirs(os.path.join(fixed_dataset_path, "pet-ct-seg"))

        os.makedirs(os.path.join(fixed_dataset_path, "ct-image"))#存放ct图像
        os.makedirs(os.path.join(fixed_dataset_path, "ct-seg"))#存放ct图像对应的标签

        os.makedirs(os.path.join(fixed_dataset_path, "pet-image"))
        os.makedirs(os.path.join(fixed_dataset_path, "pet-seg"))
    # print("oooiii")


    #从root_path读取dicom格式的pet,ct,和标签，并保存到上面创建的raw_dataset_path. raw_dataset_path的数据集未经过重采样处理
    load_ct_and_pet_data(root_path=root_path,saved_fusion_data_path=raw_dataset_path)

    args = config.args

    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, args)

    #对raw_dataset_path中的图像重采样处理，并保存到fixed_dataset_path
    tool.fix_data("ct-image", "ct-seg")#处理ct
    tool.write_train_val_name_list("ct-image", "ct-seg")#保存为训练集和测试集

    tool.fix_data("pet-image", "pet-seg")#处理pet
    tool.write_train_val_name_list("pet-image", "pet-seg")#保存为训练集和测试集
    # for (t1,t2) in file_list:
    #     tool.fix_data(t1,t2)  # 对原始图像进行修剪并保存
    #     tool.write_train_val_name_list(t1,t2)  # 创建索引txt文件，分配测试集和训练集
