import os
import cv2
import SimpleITK as sitk
import numpy as np
import pydicom
import torch
#from config_segmentation import config


def orignial_process(root_ct, root_pet,root_seg, save_ct_root, save_pet_root, save_seg_root):
    image_list = os.listdir(root_ct)
    seg_list = os.listdir(root_seg)
    for image in image_list:
        if image not in seg_list:
            print('11')
            continue
        image_name = image.split(".nii")[0]
        print(image_name)
        ct_path = os.path.join(root_ct, image)
        pet_path = os.path.join(root_pet, image)
        seg_path = os.path.join(root_seg, image)
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        pet = sitk.ReadImage(pet_path, sitk.sitkUInt16)
        pet_array = sitk.GetArrayFromImage(pet)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt16)
        seg_array = sitk.GetArrayFromImage(seg)
        # print(seg_array)
        print(np.where(seg_array>0))

        print("Ori shape:", ct_array.shape, seg_array.shape)
        print("Ori shape:", pet_array.shape, seg_array.shape)

        # 将灰度值在阈值之外的截断掉
        # ct_array[ct_array > 200] = 200
        # ct_array[ct_array < -200] = -200
        # ct_array = (ct_array + 200) / 400
        num = len(seg_array)
        print('#'*10)
        print(num)
        for i in range(num):
            ct_image = ct_array[i]
           # print(np.all(label==0))
            pet_image = pet_array[i]
            seg_image = seg_array[i]
              # 仅筛选有标记的PET/CT图像
            if seg_image.max() > 0:
                
                # 对于PET图像还可以采取其他方式来预处理。
                # pet_image = pet_image/pet_image.max() * 255
                seg_image = seg_image * 255
                seg_image = seg_image.astype(np.uint8)
                
                seg_path = os.path.join(save_seg_root, "{}_{}.png".format(image_name, i))
                print(seg_path)
                cv2.imwrite(seg_path, seg_image)

            ct_image = ct_image * 255
            ct_image = ct_image.astype(np.uint8)
            pet_image = pet_image * 255
            pet_image = pet_image.astype(np.uint8)

            ct_path = os.path.join(save_ct_root,"{}_{}.png".format(image_name,i))
            pet_path = os.path.join(save_pet_root, "{}_{}.png".format(image_name, i))
            
            cv2.imwrite(ct_path, ct_image)
            cv2.imwrite(pet_path, pet_image)


            #   print(seg_image.max())
            
            # 仅筛选有标记的PET/CT图像
            #if seg_image.max() > 0:
            #    ct_image = ct_image * 255
            #    ct_image = ct_image.astype(np.uint8)
                # 对于PET图像还可以采取其他方式来预处理。
                # pet_image = pet_image/pet_image.max() * 255
            #    pet_image = pet_image * 255

             #   pet_image = pet_image.astype(np.uint8)
             #   seg_image = seg_image * 255
             
             #   seg_image = seg_image.astype(np.uint8)

              #  ct_path = os.path.join(save_ct_root,"{}_{}.png".format(image_name,i))
              #  pet_path = os.path.join(save_pet_root, "{}_{}.png".format(image_name, i))
              #  seg_path = os.path.join(save_seg_root, "{}_{}.png".format(image_name, i))
              #  cv2.imwrite(ct_path, ct_image)
              #  cv2.imwrite(pet_path, pet_image)
              #  cv2.imwrite(seg_path, seg_image)

if __name__ == '__main__':
    # root_ct = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/ct-image"
    # root_seg = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/ct-seg"
    # root_pet = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/pet-image"
    # save_ct_root = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/CT_2"
    # save_pet_root = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/PET_2"
    # save_seg_root = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/GT_2"
    #opx = config()
    #opt = opx()
   
    #root_ct = "./FDG-PET-CT-Lesions-raw_data/fixed_fusion_data/ct-image"
    #root_seg = "./FDG-PET-CT-Lesions-raw_data/fixed_fusion_data/ct-seg"
    #root_pet = "./FDG-PET-CT-Lesions-raw_data/fixed_fusion_data/pet-image"
    #save_ct_root = "./FDG-PET-CT-Lesions-raw_data/raw_fusion_data/CT_1"
    #save_pet_root = "./FDG-PET-CT-Lesions-raw_data/raw_fusion_data/PET_1"
    #save_seg_root = "./FDG-PET-CT-Lesions-raw_data/raw_fusion_data/GT_1"
    
    
    root_ct = "/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/CT"
    root_seg = "/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/SEG"
    root_pet = "/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/PET"
    save_ct_root = "/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/CT_1"
    save_pet_root = "/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/PET_1"
    save_seg_root = "/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/SEG_1"
    print(root_ct)

    #root_ct = f'./HEAD-NECK-HN1-raw_data/CT/{opt.dataset_choice}'
    #root_seg = f'./HEAD-NECK-HN1-raw_data/PET/{opt.dataset_choice}'
    #root_pet = f'./HEAD-NECK-HN1-raw_data/SEG/{opt.dataset_choice}'

   # save_ct_root = f'./HEAD-NECK-HN1-raw_data/CT/{opt.dataset_choice}_2'
   # save_pet_root = f'./HEAD-NECK-HN1-raw_data/PET/{opt.dataset_choice}_2'
   # save_seg_root = f'./HEAD-NECK-HN1-raw_data/SEG/{opt.dataset_choice}_2'

    orignial_process(root_ct, root_pet, root_seg, save_ct_root, save_pet_root, save_seg_root)



