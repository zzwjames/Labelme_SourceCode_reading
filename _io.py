import os.path as osp

import numpy as np
import PIL.Image
from labelme.utils.draw import label_colormap

#lblsave的函数功能，由于我们按照class给图片加像素导致整体图片特别暗，lblsave的功能就是在不改变像素值的情况下在图像原基础上加一个调色盘使得我们标注的目标更明显

#样例输入：
#utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)

def lblsave(filename, lbl):
    if osp.splitext(filename)[1] != '.png':                    #判断是否有后缀
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:                    
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode='P')     #uint8,8位无符号整形，mode和type对应
        colormap = label_colormap(255)
        lbl_pil.putpalette((colormap * 255).astype(np.uint8).flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )
#PIL.Image.fromarray(obj,mode=None)    
##obj:Object with array interdace, mode:Mode to use (will be determined from type it None)

#Image.putpalette(data,rawmode='RGB') 
##data:调色板序列, 函数将调色板附加到此图像

#colormap见draw.py
