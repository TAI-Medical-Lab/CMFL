import os
import cv2
import SimpleITK as sitk
import numpy as np
import pydicom
import torch
from config_segmentation import config

# 训练集包含有病灶和无病灶的
# 验证集只包含有病灶的，对比实验结果

def orignial_process(rate1,rate2,root_ct, root_pet, root_seg, save_ct_root, save_pet_root, save_seg_root,save_ct_root1,save_pet_root1,save_seg_root1,save_ct_root2,save_pet_root2,save_seg_root2):
    image_list = os.listdir(root_ct)
    file_len = len(image_list)

    train_len = int(file_len*0.8)
    # print(train_len)

    import random
    train_list =  random.sample(image_list, train_len)
    # print(train_list)

    # val_len = int(file_len*0.2)
    
    # print(val_len)
    # image_list_train = image_list[:train_len]
    # image_list_val = image_list[train_len:]

    # val_list = image_list - train_list

    val_list = []
    for i in image_list:
        if i not in train_list:
            val_list.append(i)
    # print(val_list)


    # print(image_list)
    # print(train_list)
    # print(val_list)

    image_list_train = train_list
    image_list_val = val_list
    # print(image_list_train)
    # print(image_list_val)
    # seg_list = os.listdir(root_seg)
    # print(seg_list)
    seg_list_train = train_list
    seg_list_val = val_list


    # print(image_list)
    # print(seg_list)
    for image in image_list_train:
        if image not in seg_list_train:
            continue
        image_name = image.split(".nii")[0]
        # print(image_name)

        ct_path = os.path.join(root_ct, image)
        pet_path = os.path.join(root_pet, image)
        seg_path = os.path.join(root_seg, image)
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        pet = sitk.ReadImage(pet_path, sitk.sitkUInt16)
        pet_array = sitk.GetArrayFromImage(pet)
        # print(pet_array.shape)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        print("Ori ct shape:", ct_array.shape, seg_array.shape)
        print("Ori pet shape:", pet_array.shape, seg_array.shape)
        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200
        ct_array = (ct_array + 200) / 400
        num = len(ct_array)
        
        
        trainsegnum={}
        for i in range(num):
            seg_image = seg_array[i]
            segnum=seg_image.sum()
            if segnum>0:
                trainsegnum[i]=segnum
                
        
        
        trainsegnum = dict(sorted(trainsegnum.items(), key=lambda item: item[1]))

        # 计算需要取的键值对的数量
        # start = int(len(trainsegnum) * 0.1)
        # end = int(len(trainsegnum) * 0.2)
        start = int(len(trainsegnum) * rate1)
        end = int(len(trainsegnum) * rate2)

        print(trainsegnum)
        # 取出排序后值的前10%至20%的键值对
        selected_dict =dict(list(trainsegnum.items())[start:end])

        


        for i in selected_dict.keys():
            ct_image = ct_array[i]
            pet_image = pet_array[i]
            seg_image = seg_array[i]

            # 仅筛选有标记的PET/CT图像
            if seg_image.max() > 0:
                ct_image = ct_image * 255
                ct_image = ct_image.astype(np.uint8)
                # 对于PET图像还可以采取其他方式来预处理
                pet_image = pet_image/pet_image.max() * 255
                pet_image = pet_image.astype(np.uint8)
                seg_image = seg_image * 255

     

                ct_path = os.path.join(save_ct_root,"{}_{}.png".format(image_name,i))
                pet_path = os.path.join(save_pet_root, "{}_{}.png".format(image_name, i))
                seg_path = os.path.join(save_seg_root, "{}_{}.png".format(image_name, i))
   
                cv2.imwrite(ct_path, ct_image)
                # ct_size = ct_image.shape            
                # new_pet_image = cv2.resize(pet_image, ct_size)
                cv2.imwrite(pet_path, pet_image)
                cv2.imwrite(seg_path, seg_image)

                # cv2.imwrite(neg_ct_path, ct_image_neg)
                # neg_ct_size = ct_image_neg.shape            
                # neg_new_pet_image = cv2.resize(pet_image_neg, neg_ct_size)
                # cv2.imwrite(neg_pet_path, neg_new_pet_image)
                # cv2.imwrite(neg_seg_path, seg_image_neg)

                train_ct_path = os.path.join(save_ct_root1,"{}_{}.png".format(image_name,i))
                train_pet_path = os.path.join(save_pet_root1, "{}_{}.png".format(image_name, i))
                train_seg_path = os.path.join(save_seg_root1, "{}_{}.png".format(image_name, i))

                # train_ct_path_neg = os.path.join(save_ct_root1,"{}_{}.png".format(image_name,negative_tmp))
                # train_pet_path_neg = os.path.join(save_pet_root1, "{}_{}.png".format(image_name, negative_tmp))
                # train_seg_path_neg = os.path.join(save_seg_root1, "{}_{}.png".format(image_name, negative_tmp))



                cv2.imwrite(train_ct_path, ct_image)
                # new_pet_image = cv2.resize(pet_image, ct_size)

                cv2.imwrite(train_pet_path, pet_image)
                cv2.imwrite(train_seg_path, seg_image)

                
                # cv2.imwrite(train_ct_path_neg, ct_image_neg)
                # new_pet_image = cv2.resize(pet_image_neg, neg_ct_size)

                # cv2.imwrite(train_pet_path_neg, neg_new_pet_image)
                # cv2.imwrite(train_seg_path_neg, seg_image_neg)
                
                # ct_image = ct_image * 255
                # ct_image = ct_image.astype(np.uint8)
                # # 对于PET图像还可以采取其他方式来预处理。
                
                # # pet_image = pet_image/pet_image.max() * 255
                # pet_image = pet_image * 255
                # pet_image = pet_image.astype(np.uint8)


                # seg_image = seg_image.astype(np.uint8)
    
    for image in image_list_val:
        if image not in seg_list_val:
            continue
        image_name = image.split(".nii")[0]
        # print(image_name)

        ct_path = os.path.join(root_ct, image)
        pet_path = os.path.join(root_pet, image)
        seg_path = os.path.join(root_seg, image)
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        pet = sitk.ReadImage(pet_path, sitk.sitkUInt16)
        pet_array = sitk.GetArrayFromImage(pet)
        # print(pet_array.shape)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)
        print("Ori ct shape:", ct_array.shape, seg_array.shape)
        print("Ori pet shape:", pet_array.shape, seg_array.shape)
        # 将灰度值在阈值之外的截断掉
        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200
        ct_array = (ct_array + 200) / 400
        num = len(ct_array)
        
        # valid_len = int(num*0.2)

        # print(num)
        # for i in range(train_len):

        validsegnum={}
        for i in range(num):
            seg_image = seg_array[i]
            segnum=seg_image.sum()
            if segnum>0:
                validsegnum[i]=segnum
        
        validsegnum = dict(sorted(validsegnum.items(), key=lambda item: item[1]))

        # 计算需要取的键值对的数量
        # start = int(len(validsegnum) * 0.1)
        # end = int(len(validsegnum) * 0.2)

        start = int(len(validsegnum) * rate1)
        end = int(len(validsegnum) * rate2)
        



        # 取出排序后值的前10%至20%的键值对
        selected_dict =dict(list(validsegnum.items())[start:end])



        for i in selected_dict.keys():
            ct_image = ct_array[i]
            pet_image = pet_array[i]
            seg_image = seg_array[i]
            # print(seg_image.max())
            
            # # 仅筛选有标记的PET/CT图像
            if seg_image.max() > 0:
                ct_image = ct_image * 255
                ct_image = ct_image.astype(np.uint8)
                pet_image = pet_image/pet_image.max() * 255
                pet_image = pet_image.astype(np.uint8)
                seg_image = seg_image * 255
                
            #     # pet_image = pet_image * 255
            #      # 对于PET图像还可以采取其他方式来预处理。

                # 选相同数量没有标记的PET/CT图像 
                # negative_tmp = np.random.randint(num)

                # print(negative_tmp)
                # seg_image_neg = seg_array[negative_tmp]
                # if seg_image_neg.max() == 0:
                #     ct_image_neg = ct_array[negative_tmp]
                #     pet_image_neg = pet_array[negative_tmp]
                #     seg_image_neg = seg_array[negative_tmp]
                #     print(seg_image_neg.max())
                #     ct_image_neg = ct_image_neg * 255
                #     ct_image_neg = ct_image_neg.astype(np.uint8)
                #     pet_image_neg = pet_image_neg/pet_image_neg.max() * 255
                #     pet_image_neg = pet_image_neg.astype(np.uint8)
                # else:
                #     negative_tmp = np.random.randint(num)
                #     ct_image_neg = ct_array[negative_tmp]
                #     pet_image_neg = pet_array[negative_tmp]
                #     seg_image_neg = seg_array[negative_tmp]
                #     print(seg_image_neg.max())
                #     ct_image_neg = ct_image_neg * 255
                #     ct_image_neg = ct_image_neg.astype(np.uint8)
                #     pet_image_neg = pet_image_neg/pet_image_neg.max() * 255
                #     pet_image_neg = pet_image_neg.astype(np.uint8)


                ct_path = os.path.join(save_ct_root,"{}_{}.png".format(image_name,i))
                pet_path = os.path.join(save_pet_root, "{}_{}.png".format(image_name, i))
                seg_path = os.path.join(save_seg_root, "{}_{}.png".format(image_name, i))
                
                # neg_ct_path = os.path.join(save_ct_root,"{}_{}.png".format(image_name,negative_tmp))
                # neg_pet_path = os.path.join(save_pet_root, "{}_{}.png".format(image_name, negative_tmp))
                # neg_seg_path = os.path.join(save_seg_root, "{}_{}.png".format(image_name, negative_tmp))


                cv2.imwrite(ct_path, ct_image)
                # ct_size = ct_image.shape            
                # new_pet_image = cv2.resize(pet_image, ct_size)
                cv2.imwrite(pet_path, pet_image)
                cv2.imwrite(seg_path, seg_image)

                # cv2.imwrite(neg_ct_path, ct_image_neg)
                # neg_ct_size = ct_image_neg.shape            
                # neg_new_pet_image = cv2.resize(pet_image_neg, neg_ct_size)
                # cv2.imwrite(neg_pet_path, neg_new_pet_image)
                # cv2.imwrite(neg_seg_path, seg_image_neg)

                val_ct_path = os.path.join(save_ct_root2,"{}_{}.png".format(image_name,i))
                val_pet_path = os.path.join(save_pet_root2, "{}_{}.png".format(image_name, i))
                val_seg_path = os.path.join(save_seg_root2, "{}_{}.png".format(image_name, i))

                # val_ct_path_neg = os.path.join(save_ct_root2,"{}_{}.png".format(image_name,negative_tmp))
                # val_pet_path_neg = os.path.join(save_pet_root2, "{}_{}.png".format(image_name, negative_tmp))
                # val_seg_path_neg = os.path.join(save_seg_root2, "{}_{}.png".format(image_name, negative_tmp))


                cv2.imwrite(val_ct_path, ct_image)
                # new_pet_image = cv2.resize(pet_image, ct_size)

                cv2.imwrite(val_pet_path, pet_image)
                cv2.imwrite(val_seg_path, seg_image)

                
                # cv2.imwrite(val_ct_path_neg, ct_image_neg)
                # new_pet_image = cv2.resize(pet_image_neg, neg_ct_size)

                # cv2.imwrite(val_pet_path_neg, neg_new_pet_image)
                # cv2.imwrite(val_seg_path_neg, seg_image_neg)

                
               
if __name__ == '__main__':
    # root_ct = "./test_data1/raw_fusion_data/ct-image"
    # root_seg = "./test_data1/raw_fusion_data/ct-seg"
    # root_pet = "./test_data1/raw_fusion_data/pet-image"
    # save_ct_root = "./test_data1/raw_fusion_data/CT_1"
    # save_pet_root = "./test_data1/raw_fusion_data/PET_1"
    # save_seg_root = "./test_data1/raw_fusion_data/GT_1"

    # root_ct = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/ct-image"
    # root_seg = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/ct-seg"
    # root_pet = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/pet-image"
    # save_ct_root = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/CT_2"
    # save_pet_root = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/PET_2"
    # save_seg_root = "./HEAD-NECK-HN1-raw_data/raw_fusion_data/GT_2"
    opx = config()
    opt = opx()

    # root_ct = f'./HEAD-NECK-HN1-raw_data/CT/{opt.dataset_choice}'
    # root_pet = f'./HEAD-NECK-HN1-raw_data/PET/{opt.dataset_choice}'
    # root_seg = f'./HEAD-NECK-HN1-raw_data/SEG/{opt.dataset_choice}'

    # save_ct_root = f'./HEAD-NECK-HN1-raw_data/CT/{opt.dataset_choice}_2'
    # save_pet_root = f'./HEAD-NECK-HN1-raw_data/PET/{opt.dataset_choice}_2'
    # save_seg_root = f'./HEAD-NECK-HN1-raw_data/SEG/{opt.dataset_choice}_2'

    # 全量数据 
    # if opt.dataset_choice == 'all':
    root_ct = f'/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/CT'
    root_pet = f'/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/PET'
    root_seg = f'/mnt/sevenT/ZLLing/data_process_yl/FDG-nii-raw/SEG'


    rate1=0.75
    rate2=1
    
    save_ct_root = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/CT_2'
    save_pet_root = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/PET_2'
    save_seg_root = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/SEG_2'

    if os.path.exists(save_ct_root) != True:
        os.makedirs(save_ct_root)
        os.makedirs(save_pet_root)
        os.makedirs(save_seg_root)


    save_ct_root1 = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/train/CT'
    save_pet_root1 = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/train/PET'
    save_seg_root1 = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/train_GT'

    save_ct_root2 = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/valid/CT'
    save_pet_root2 = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/valid/PET'
    save_seg_root2 = f'/mnt/sevenT/ZLLing/data_process_yl/{rate1}_{rate2}/valid_GT'
    
    
    os.makedirs(save_ct_root1)
    os.makedirs(save_pet_root1)
    os.makedirs(save_seg_root1)
    os.makedirs(save_ct_root2)
    os.makedirs(save_pet_root2)
    os.makedirs(save_seg_root2)
    

    
    orignial_process(rate1,rate2,root_ct, root_pet, root_seg, save_ct_root, save_pet_root, save_seg_root,save_ct_root1,save_pet_root1,save_seg_root1,save_ct_root2,save_pet_root2,save_seg_root2)




    # load_ct_and_pet_data(root_path=save_ct_root,saved_data_path=save_ct_root1)

  


