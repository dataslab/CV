#CUDA_VISIBLE_DEVICES=1,2,3 bash ./tools/dist_train.sh ./vfnet_coco.py  3 --no-validate  #多卡
CUDA_VISIBLE_DEVICES=2 python ./tools/train.py ./vfnet_coco.py  

#CUDA_VISIBLE_DEVICES=2 python ./tools/train.py ./mask_rcnn_coco.py  

