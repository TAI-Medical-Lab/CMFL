from PIL import Image
import os

input_folders = ['./CentralCrop/valid/CT',
                 './CentralCrop/valid/PET',
                 './CentralCrop/valid_GT']  # 输入文件夹列表
output_folders = ['./CentralCrop/all_fusion']  # 输出文件夹列表


# 如果输出文件夹不存在，就创建它
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


def fusion_pet_ct_seg(petpath,ctpath,segpath,outpath):
    # 读取灰度图像
    image2 = Image.open(petpath).convert('L')
    image1 = Image.open(ctpath).convert('L')
    image3 = Image.open(segpath).convert('L')
    
    print(type(image1))
    print(image1.getbands())

    # 创建新的RGB图像
    rgb_image = Image.new('RGB', image1.size)

    # 合并灰度图像为RGB图像
    rgb_image.putdata(list(zip(image1.getdata(), image2.getdata(), image3.getdata())))
    
    print(type(rgb_image))

    # 保存RGB图像
    rgb_image.save(outpath)

pnglist=os.listdir(input_folders[0])

for png in pnglist:
    ctpath=os.path.join(input_folders[0],png)
    petpath=os.path.join(input_folders[1],png)
    segpath=os.path.join(input_folders[2],png)
    outpath=os.path.join(output_folders[0],png)
    if os.path.exists(petpath) and os.path.exists(segpath):
        fusion_pet_ct_seg(petpath,ctpath,segpath,outpath)


# # 遍历输入文件夹中的所有png文件
# for i, folder in enumerate(input_folders):
#     for file in os.listdir(folder):
#         if file.endswith('.png'):
#             # 打开图片
#             img_path = os.path.join(folder, file)
#             img = Image.open(img_path)

#             # 缩放图片
#             img_resized = img.resize(new_size)

#             # 保存缩放后的图片
#             new_path = os.path.join(output_folders[i], file)
#             img_resized.save(new_path)

#             print(f"{file} has been resized and saved to {new_path}.")
