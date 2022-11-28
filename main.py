
import argparse
import cv2
import math
import numpy as np
parser = argparse.ArgumentParser(description="Please type the path of the image folder")
parser.add_argument('--patch_w', type=int, default=512,help='Height of input image')
parser.add_argument('--stride_w', type=int, default=256,help='Height of input image')
parser.add_argument('--H', type=int, default=64,help='Height of input image')
args = parser.parse_args()
H = args.H
def overlapping_seg(img:np.ndarray):
    '''
    重叠切片
    :param img_path: 待切图片路径
    :param img_name: 待切图片名称
    :return: [子图1,子图2,...,子图N]
    '''
    # print(f'ori input img shape:{img.shape}')
    h,w = img.shape[:2]
    # print(h,w,c)
    patch_h = args.H
    ratio = patch_h/h
    resized_w = int(w*ratio)
    img = cv2.resize(img, (resized_w, patch_h))
    # print(f'img.shape waiting for overlap resized :{img.shape}')
    h = patch_h

    patch_w = args.patch_w

    stride_w = args.stride_w

    # 以长度 patch_h 步长stride_h的方式滑动
    stride_h = H
    # print(img.shape[1],patch_w)

    if patch_w>img.shape[1] and patch_w-img.shape[1] < 30:
        rst = cv2.copyMakeBorder(img,0,0,0,64,cv2.BORDER_CONSTANT,value=(0,0,0))
        rst= cv2.resize(rst,(patch_w,H))
        # print(f'未达到长度-30，直接返回。返回形状:{rst.shape}')
        return [rst]
    if img.shape[1]<patch_w:
        rst = cv2.copyMakeBorder(img,0,0,0,patch_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
        # print(f'未达到长度，直接返回。返回形状:{rst.shape}')
        return [rst]
    # print(ratio)
    # print(img.shape)

    # print(f'after copymakeborder img shpae:{img.shape}')
    rescaled_h,rescaled_w = img.shape[:2]
    n_w = int(math.ceil((rescaled_w-patch_w)/stride_w))*stride_w+patch_w
    n_h = H

    img = cv2.copyMakeBorder(img,0,0,0,n_w-img.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
    # img = cv2.resize(img, (n_w, n_h))

    # print(f'长边自适应尺寸:{img.shape}')

    rescaled_h,rescaled_w = img.shape[:2]
    n_patch_h = (rescaled_h-patch_h)//stride_h+1
    assert n_patch_h==1,'n_patch_h!=1'
    n_patch_w = (rescaled_w-patch_w)//stride_w+1

    # print(f'n_patch_h：{n_patch_h}，n_patch_w：{n_patch_w}')
    rst = []
    for i in range(n_patch_w):
        x1 = i * stride_w
        x2 = x1 + patch_w
        roi = img[0:H,x1:x2]
        # print(f'roi.shape:{roi.shape}')
        rst.append(roi)
    if len(rst)==0:
        print('overlap len is 0, this means something could be wrong but not that so lethel')
        return [img]

    return rst


def merge_str(a:str,b:str,k=2):
    if a != '':
        key = b[1:1+k]
        # print(key)
        index = a.rfind(key) #,len(a)-k-1,len(a)
        # 如果无法合并
        if index == -1:
            # print(f'unable to merge str, return the concat of {a} and {b}')
            rst = a + b #对编辑距离来说 该操作效果更好
        else:
            rst = a[:index]+b[1:]
        return rst
    else:
        return b
def merge_strs(strs:list):
    rst = ''
    for i in strs:
        rst = merge_str(rst,i)
    return rst
if __name__ == '__main__':
    img = cv2.imread('img-path')
    patches = overlapping_seg(img)
    # Convert each patch with e2e recognizer~
    sub_strs=[]
    for patch in patches:
        sub_strs.append('recognition result')
    
    merge_strs(sub_strs)
    
