#!/bin/bash

world_sizes=4
batch_sizes=2
dp_delta_1=0.0000001
dp_delta_2=0.0001
dp_delta_3=0
# dp_delta_2=0.0000000001
dataset_choice='all'
root='./Headneck'
epoch_count=1


python main_segmentation.py \
    --gpu_id '0' \
    --dataset_named 'petct' \
    --model_choice 'navie_model_concat' \
    --dataset_choice $dataset_choice \
    --root $root \
    --epoch_count $epoch_count \
    --setting central & 
    
# for i in `seq 0 3`
# do
#     {
#     python main_segmentation.py \
#         --gpu_id '1' \
#         --dataset_named 'petct' \
#         --model_choice 'navie_model_concat' \
#         --dataset_choice $dataset_choice \
#         --dp_delta $dp_delta_3 \
#         --backend gloo \
#         --init_method tcp://localhost:63526 \
#         --world_size $world_sizes \
#         --rank $i \
#         --batchSize $batch_sizes
#     }&
# done


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
#         --init_method tcp://localhost:12444 \
#         --world_size $world_sizes \
#         --rank $i \
#         --batchSize $batch_sizes
#     }&
# done

# for i in `seq 0 3`
# do
#     {
#     python main_segmentation.py \
#         --gpu_id '3' \
#         --dataset_named 'petct' \
#         --model_choice 'navie_model_concat' \
#         --dataset_choice $dataset_choice \
#         --dp_delta $dp_delta_2 \
#         --backend gloo \
#         --init_method tcp://localhost:62526 \
#         --world_size $world_sizes \
#         --rank $i \
#         --batchSize $batch_sizes
#     }&
# done
    

# for i in `seq 0 3`
# do
#     {
#     python main_segmentation.py \
#         --gpu_id '2' \
#         --dataset_named 'petct' \
#         --model_choice 'best_model_embeding_attention' \
#         --dataset_choice $dataset_choice \
#         --dp_delta $dp_delta_3 \
#         --backend gloo \
#         --init_method tcp://localhost:12351 \
#         --world_size $world_sizes \
#         --rank $i \
#         --batchSize $batch_sizes
#     }&
# done
# for i in `seq 0 3`
# do
#     {
#     python main_segmentation.py \
#         --gpu_id '2' \
#         --dataset_named 'petct' \
#         --model_choice 'best_model_embeding_attention' \
#         --dataset_choice $dataset_choice \
#         --dp_delta $dp_delta_1 \
#         --backend gloo \
#         --init_method tcp://localhost:12351 \
#         --world_size $world_sizes \
#         --rank $i \
#         --batchSize $batch_sizes
#     }&
# done



# done

    # for i in `seq 0 3`
    # do
    #     {
    #     python main_segmentation.py \
    #         --gpu_id '0' \
    #         --dataset_named 'petct' \
    #         --model_choice 'best_model_embeding_attention' \
    #         --dataset_choice $dataset_choice \
    #         --dp_delta $dp_delta_4 \
    #         --backend gloo \
    #         --init_method tcp://localhost:12344 \
    #         --world_size $world_sizes \
    #         --rank $i \
    #         --batchSize $batch_sizes
    #     }&
    # done
        

