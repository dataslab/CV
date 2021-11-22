#单卡
#CUDA_VISIBLE_DEVICES=3 python ./tools/test.py  ./vfnet_coco.py  ./pretrain/vfnet_r50_ms_2x_44.5.pth   --eval bbox
#多卡
CUDA_VISIBLE_DEVICES=1,2,3,./tools/dist_test.sh ./vfnet_coco.py ./pretrain/vfnet_r50_ms_2x_44.5.pth 4 --eval bbox


