conda create -n "lookhere" python=3.8 -y
conda activate lookhere
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -y
conda install pip -y 
pip install aiohttp aiortc ninja numpy opencv-python segmentation-models-pytorch tqdm tensorboard gdown

mkdir downloads
cd downloads
# resnet18_adam.pth.tar
gdown 1E0UjBSAzltB3byhmQY2_2UckGmtQuD03
cp resnet18_adam.pth.tar ../demo_app/src/ckpt/
mv resnet18_adam.pth.tar ../object_highlights/ckpt/

gdown 1iE2Xxpmu9CljX9TsQXMJncEQYHKUqEKl
mv unet-b0-bgr-100epoch.pt ../demo_app/src/ckpt/
cd ..