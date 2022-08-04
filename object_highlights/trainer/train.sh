
dataset=$1


# CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 trainer/train.py \
# --dataset ${dataset} --epoch 100 --batch_size 4 --world_size 2 --datasetname hutics \
# --modelname unet_b0 --encoder_pretrain

python trainer/train.py --epoch 100 --batch_size 4 --world_size 2 \
    --datasetname hutics --modelname unet_b0 --encoder_pretrain --dataset ${dataset}
