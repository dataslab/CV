
#一个test的小例子
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = './vfnet_coco.py'
# Setup a checkpoint file to load
checkpoint = './pretrain/vfnet_r50_dcn_ms_2x_47.8.pth'

# initialize the detector
model = init_detector(config, checkpoint, device='cuda:0')
# Use the detector to do inference
img = 'example.jpg'
result = inference_detector(model, img)
#print(result)
# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.3,out_file='./result_example.jpg')
