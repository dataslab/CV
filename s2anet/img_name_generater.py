import os
dir="/home/songhetian/dataset/airplaneDOTA/airplane/val/images"
img_name_list=[]
for root, dirs, files in os.walk(dir):
  for file in files:
   # print os.path.join(root,file)
   img_name=file.split(".")[0]
   img_name_list.append(img_name)

write_path="/home/songhetian/dataset/airplaneDOTA/airplane/val/test_image_list.txt"
#写入文本
with open(write_path,"w") as f:
    for i in range(len(img_name_list)):
        f.write(img_name_list[i]) 
        f.write("\n")
print("end")
#print(img_name_list[0:10])
    