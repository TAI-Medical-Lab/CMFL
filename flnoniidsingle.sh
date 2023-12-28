#!/bin/bash

# dataset_choice='all'
# python data_process_ll.py
world_sizes=4
batch_sizes=4
dp_delta_1=0.0000001
dp_delta_2=0.0001
dp_delta_3=0
dp_delta_4=1e-5
dp_delta_5=1e-6
# dp_delta_2=0.0000000001
dataset_choice='all'
root='./FDG-nii-raw-MELANOMA/'
root0='./FDG-nii-raw-MELANOMA/'
root3='./FDG-nii-raw-LYMPHOMA/'
root2='./FDG-nii-raw-LUNG_CANCER/'
root1='./FDG-nii-raw-age0_55/'
epoch_count=1
noniid='noniid'
setting='ours'
weight=1
standalone='false'

for i in `seq 0 3`
do
    {
    python save_client_main_test.py \
        --setting $setting \
        --noniid $noniid \
        --root $root \
        --root0 $root0 \
        --root1 $root1 \
        --root2 $root2 \
        --root3 $root3 \
        --low_bound_ensure 0 \
        --ours_dir 0 \
        --ours_loss 0 \
        --ours_pccs 1 \
        --q 0.75 \
        --ours_dis 0 \
        --weight $weight \
        --DP no \
        --ratio 64 \
        --gpu_id '0' \
        --dataset_named 'petct' \
        --model_choice 'navie_model_concat' \
        --dataset_choice $dataset_choice \
        --backend gloo \
        --epoch_count $epoch_count \
        --init_method tcp://localhost:16146 \
        --world_size $world_sizes \
        --rank $i \
        --standalone $standalone \
        --batchSize $batch_sizes
    }&
done



# for i in `seq 0 3`
# do
#     {
#     python main_segmentation.py \
#         --gpu_id '3' \
#         --dataset_named 'petct' \
#         --model_choice 'best_model_embeding_attention' \
#         --dataset_choice $dataset_choice \
#         --dp_delta $dp_delta_2 \
#         --backend gloo \
#         --init_method tcp://localhost:12351 \
#         --world_size $world_sizes \
#         --rank $i \
#         --batchSize $batch_sizes
#     }&
# done
