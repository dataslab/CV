import argparse
import os
import os.path as osp
import pdb
import random

import cv2
import mmcv
from mmcv import Config

from mmdet.apis import init_detector, inference_detector
from mmdet.core import rotated_box_to_poly_single
from mmdet.datasets import build_dataset
import json
import time
from tqdm import tqdm
#保存一张图片的结果
def save_single_result_rbox(image_name, detections, class_names, scale=1.0, threshold=0.2):

    result_dic={}
    result_dic["image_name"]=image_name
    result_dic["labels"]=[]


    #遍历一张图片的每个类
    for j, name in enumerate(class_names):
        try:
            dets = detections[j]
        except:
   #         pdb.set_trace()
            continue #没结果的话就继续
        #一个类的多个结果
        for det in dets:
            temp_result={}
            temp_result["category_id"]=name #种类
            temp_result["confidence"]=0.99  #随便给的置信度
            temp_result["points"]=[]

            #===================bbox======================
            score = det[-1]
            det = rotated_box_to_poly_single(det[:-1])
            bbox = det[:8] * scale
            if score < threshold:
                continue
            bbox = list(map(float, bbox))
            #=============================================
            #[2482, 2230, 2550, 2239, 2542, 2301, 2474, 2292]坐标
            #四个点 
            for i in range(4):
                temp_list=[bbox[2*i],bbox[2*i+1]]
                temp_result["points"].append(temp_list)
            result_dic["labels"].append(temp_result)
    
    return result_dic


def save_det_result(config_file, out_dir, checkpoint_file=None, img_dir=None, colormap=None):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data.test
    dataset = build_dataset(data_test)
    classnames = dataset.CLASSES
   # print(classnames)
    # use checkpoint path in cfg
    if not checkpoint_file:
        checkpoint_file = osp.join(cfg.work_dir, 'latest.pth')
  
    # use testset in cfg
    if not img_dir:
        img_dir = data_test.img_prefix

    model = init_detector(config_file, checkpoint_file, device='cuda:1')

    img_list = os.listdir(img_dir)

    result_list=[]

    #遍历每张照片
    for img_name in tqdm(img_list):
        img_path = osp.join(img_dir, img_name)
        #模型推断
        result = inference_detector(model, img_path)
        #保存bounding box结果        
        single_result = save_single_result_rbox(img_name,result,classnames,scale=1.0,threshold=0.3)
        result_list.append(single_result)
    
    #将结果整成json    
    result_json=json.dumps(result_list)

    #写文件
    with open(out_dir+"/aircraft_results.json", "w", encoding='utf-8') as f:
        #json.dump(result_json,f)
        f.write(json.dumps(json.loads(result_json),indent=2))
    print("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('--config_file', help='input config file',default="s2anet_dota.py")
    parser.add_argument('--model', help='pretrain model',default="./pretrain/s2anet_r50_fpn_1x_converted-11c9c5f4.pth")
    parser.add_argument('--input_dir', help='img dir',default="./example_test/DOTA_img")
    parser.add_argument('--output_dir', help='output dir',default="./example_test/DOTA_result")
    args = parser.parse_args()
    
    #没有的话创建一下文件夹
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    start_time = time.time()
    save_det_result(args.config_file, args.output_dir, checkpoint_file=args.model, img_dir=args.input_dir)
    print('total time:', time.time() - start_time)