import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

'''
制作自己的语义分割数据集需要注意以下几点：
1、我使用的labelme版本是3.16.7，建议使用该版本的labelme，有些版本的labelme会发生错误，
   具体错误为：Too many dimensions: 3 > 2
   安装方式为命令行pip install labelme==3.16.7
2、此处生成的标签图是8位彩色图，与视频中看起来的数据集格式不太一样。
   虽然看起来是彩图，但事实上只有8位，此时每个像素点的值就是这个像素点所属的种类。
   所以其实和视频中VOC数据集的格式一样。因此这样制作出来的数据集是可以正常使用的。也是正常的。
'''
if __name__ == '__main__':
    jpgs_path   = "datasets/JPEGImages"                 #
    pngs_path   = "datasets/SegmentationClass"          #mask图片保存位置
    #classes     = ["_background_","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    classes     = ["_background_","cat","human"]        #类别
    
    count = os.listdir("./datasets/before/")            #原图片以及json
    for i in range(0, len(count)):                      #循环处理每张图片
        path = os.path.join("./datasets/before", count[i])    #路径

        if os.path.isfile(path) and path.endswith('json'):    #json文件
            data = json.load(open(path))                      #加载json文件
            #"imageData": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQg
            if data['imageData']:
                imageData = data['imageData']
            else:                                                                      
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])   
                with open(imagePath, 'rb') as f:     #以二进制方式只读打开
                    imageData = f.read()             #读取所有字节   b'\xff\xd8\xff
                    imageData = base64.b64encode(imageData).decode('utf-8')  #编码、解码 /9j/4AAQSk
            #img_b64_to_arr将imagedata中的字符转化成原始图像
            img = utils.img_b64_to_arr(imageData)     
            label_name_to_value = {'_background_': 0}
            #给每个class赋值一个label
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            #label_name_to_value:{background:0,cat:1,dog:2}
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]): #按照key的值排序
                label_values.append(lv)   #0,1,2
                label_names.append(ln)    #background,cat,dog
            assert label_values == list(range(len(label_values)))    #判断是否异常
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)    
            #()放到JPEGImages文件夹里
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))
            np.set_printoptions(threshold=np.inf)
            print(np.array(lbl))
            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                #给图像像素附上对应class的label(语义分割的图像)
                new = new + index_all*(np.array(lbl) == index_json)
            print('this is new')
            print(np.array(new))
            #PIL.Image.fromarray(lbl).save(osp.join(jpgs_path, count[i].split(".")[0]+'.png'))
            #在不改变像素值的情况下，给原图像加一层调色盘
            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
