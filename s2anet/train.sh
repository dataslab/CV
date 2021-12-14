#CUDA_VISIBLE_DEVICES=1,2,3 bash ./tools/dist_train.sh vfnet_tianzhi.py 3 --no-validate

#CUDA_VISIBLE_DEVICES=2 python ./tools/train.py s2anet_dota.py 
#CUDA_VISIBLE_DEVICES=1,2,3 bash ./tools/dist_train.sh ./vfnet_dota_1024.py 3 --no-validate

CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_train.sh ./s2anet_tianzhi.py 4 #--validate
