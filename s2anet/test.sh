CUDA_VISIBLE_DEVICES=0 python ./tools/test.py  ./s2anet_tianzhi.py  ./work_dir/s2anet_tianzhi/latest.pth  --out work_dir/s2anet_dota/res.pkl --eval bbox 
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/dist_test.sh  ./s2anet_dota.py work_dir/s2anet_dota/s2anet_r50_fpn_1x_converted-11c9c5f4.pth 4   --out work_dirs/s2anet_r50_fpn_1x/res.pkl --eval bbox 
