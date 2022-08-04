datadir=$1
epoch=50
gpus=0
do_joint=$2


if [ $do_joint -eq 1 ]
then 
    echo "Training joint model"
    CUDA_VISIBLE_DEVICES=${gpus} torchrun --nproc_per_node=1 src/trainer/train.py --world_size 1 --ddp\
    --dataset ${datadir} --epoch ${epoch} --joint
else
    echo "Training Simple Classifer"
    CUDA_VISIBLE_DEVICES=${gpus} torchrun --nproc_per_node=1 src/trainer/train.py --world_size 1 --ddp\
    --dataset ${datadir} --epoch ${epoch}
fi
