 #!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
import caffe
import cv2
import numpy as np
#from python_wrapper import *
import os

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print("bb", boundingbox)
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print('#################')
    #print('boxes', boxes)
    #print('w,h', w, h)
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]
    #其实经过rerec处理之后，tmph和tmpw很接近，基本上一样，差距±1
    #print('tmph', tmph)
    #print('tmpw', tmpw)

    dx = np.ones(numbox)#全是1
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]# 第一个方括号，取出第一列的值，第二个方括号降低位数# 左上角x
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]# 右下角x坐标
    ey = boxes[:,3:4][:,0]# what is this？
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]# 这一步相当于把pad填充的那一部分给去掉了，画张图，分析下就会明白
        ex[tmp] = w-1 # 把ex>w的这一点给置成w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])#x的相应位置全部置1

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
    #print("dy"  ,dy )
    #print("dx"  ,dx )
    #print("y "  ,y )
    #print("x "  ,x )
    #print("edy" ,edy)
    #print("edx" ,edx)
    #print("ey"  ,ey )
    #print("ex"  ,ex )


    #print('boxes', boxes)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]#get box's width
    h = bboxA[:,3] - bboxA[:,1]#get box's height
    l = np.maximum(w,h).T      #compare w and h one by one, then get the large one as output, and transpose the matrix
    
    #print('bboxA', bboxA)
    #print('w', w)
    #print('h', h)
    #print('l', l)
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    #这相当于是一个正方形了，高宽均为小L
    #关于这个调整方法，建议用实际数值带入一下，会很清晰
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T #这个np.repeat([l注意这个这不是1，而是小写的L], 2, axis = 0)将会自动根据bboxA[:,0:2]进行适配
    return bboxA

#  nms(boxes, 0.5, 'Union') 
def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:  #detect the boxes is null or something
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]  #what is this? this is score
    area = np.multiply(x2-x1+1, y2-y1+1) #很明显
    I = np.array(s.argsort()) # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到I
    
    pick = [];
    while len(I) > 0:
        # np.argsort()[-1]即输出x中最大值对应的index，np.argsort()[-2]即输出x中倒数第二大值对应的index，依此类推
        # x1[I[-1] 是x1序列中最大值，x1[I[0:-1]]是从最小到次大值。 
        # 这一步是把所有的小于x1[I[-1]]，都置为x1[I[-1]]
        #接下来的xx1，yy1，xx2，yy2 都是为了计算二者的重合那一小块面积
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        # 接下来是最小
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])#这一步是面积最小的
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]#返回的是当前小于threshold的值的index
 
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T #map是经过softmax输出的概率
    #reg的尺寸一般（4，m，n） m和n根据不同的照片而不同
    #初步估计（dx1,dy1）是左上点，（dx2,dy2)是右下点的修改向量
    dx1 = reg[0,:,:].T #dx1...dy2这四个变量的获取主要依赖于神经网络计算得出
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    #返回满足条件的数据的坐标集合
    '''
    a = np.arange(27).reshape(3,3,3)
    (x,y) =  np.where(a > 5)
    x = [2 2 2] y = [0 1 2] 对照矩阵可以看出
    '''
    (x, y) = np.where(map >= t) #t是threshold

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].Ta
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little strange, when there is only one bb created by PNet
        
        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
    #print("dx1.shape", dx1.shape)
    #print('map.shape', map.shape)
   
    #找到对应的置信度，都大于threshold
    score = map[x,y]
    '''
    d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    pirnt(d) 
    [[ 1.  2.  3.  4.]
    [ 5.  6.  7.  8.]
    [ 9. 10. 11. 12.]]
    '''
    #也就是变成四行若干列的矩阵
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])
    #正常的话reg.shape[0]其数值应该为4
    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T #注意这个转置操作
    '''
    numpy.around(a)：平均到给定的小数位数。
    numpy.round_(a)：将数组舍入到给定的小数位数。
    numpy.rint(x)：修约到最接近的整数。
    numpy.fix(x, y)：向 0 舍入到最接近的整数。
    numpy.floor(x)：返回输入的底部(标量 x 的底部是最大的整数 i)。
    numpy.ceil(x)：返回输入的上限(标量 x 的底部是最小的整数 i).
    numpy.trunc(x)：返回输入的截断值。
    '''
    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # 注意这个转置 matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to 这个是加了12个像素的移动
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print('(x,y)',x,y)
    #print('score', score)
    #print('reg', reg)

    return boundingbox_out.T



def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

from time import time
_tstart_stack = []
# 获取系统当前时间，很奇怪，time.sleep()函数使用会报错
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print(fmt % (time()-_tstart_stack.pop()))

def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()

    factor_count = 0
   
    # numpy.zeros(shape, dtype=float, order=’C’)shape：int或ints序列
    # 新数组的形状，例如（2，3 ）或2。
    # dtype：数据类型，可选
    # 数组的所需数据类型，例如numpy.int8。默认值为 numpy.float64。
    # order：{'C'，'F'}，可选
    # 是否在存储器中以C或Fortran连续（按行或列方式）存储多维数据。

    # 这里是干嘛的，生成了一个0行9列的数组
    total_boxes = np.zeros((0,9), np.float)
    points = []
    # img.shape 返回图像高（图像矩阵的行数）、宽（图像矩阵的列数）和通道数3个属性组成的元组，
    # 若图像是非彩色图，则只返回高和宽组成的元组。
    # 图像矩阵img的size属性和dtype分别对应图像的像素总数目和图像数据类型。一般情况下，图像的数据类型是uint8。
    # uint8:在此输入正文8位的无符号整形数据，取值范围从0到255
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w) #返回高和宽中小的那个
    # 很多时候我们用numpy从文本文件读取数据作为numpy的数组，默认的dtype是float64
    # 但是有些场合我们希望有些数据列作为整数, 如果直接改dtype='int'的话，就会出错！原因如上，数组长度翻倍了！！！
    # astype(type): returns a copy of the array converted to the specified type.
    # 此处写的是float 而不是np.float64, Numpy很聪明，会将python类型映射到等价的dtype上
    # 这是不是将整数uint8，给变成浮点型float
    img = img.astype(float)
    # minsize = 20
    m = 12.0/minsize#此处为什么？m=0.6
    minl = minl*m#长宽中小的那一个乘以0.6
    
    #  注意这个是python自己带的。range(start, stop[, step]) -> range object，根据start与stop指定的范围以及step设定的步长，生成一个序列。
    #  参数含义：start:计数从start开始。默认是从0开始。例如range（5）等价于range（0， 5）;
    #  end:技术到end结束，但不包括end.例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
    #  scan：每次跳跃的间距，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)
    #  函数返回的是一个range object

    # arange函数用于创建等差数组，使用频率非常高，arange非常类似range函数，会python的人肯定经常用range函数，
    # 比如在for循环中，几乎都用到了range，下面我们通过range来学习一下arange，
    # 两者的区别仅仅是arange返回的是一个array，而range返回的是list。

    # 
    #total_boxes = np.load('total_boxes.npy')
    #total_boxes = np.load('total_boxes_242.npy')
    #total_boxes = np.load('total_boxes_101.npy')

    
    # create scale pyramid
    scales = []
    #只要还大于12像素
    #这里应该是在构筑图像金字塔
    while minl >= 12:
        #scales: [0.6, 0.42539999999999994, 0.30160859999999995, 0.21384049739999994, 0.15161291265659996, 0.10749355507352938, 0.07621293054713232, 0.054034967757916816, 0.038310792140363016]
        scales.append(m * pow(factor, factor_count))#m=0.6 factor = 0.709 factor_count = 0
        minl *= factor
        factor_count += 1
    
    # first stage
    #scale 为缩放比例
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            #去均值，归一化 
            #所有点的像素值减去127.5，然后再除以128（相当于×0.0078125），就是把像素值近似归一化到(-1,1)之间。
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            #这一行代码将整个图像都成比例缩小了
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            #将图像缩小了
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print("scale:", scale)

        #为何更换不明白，这又变成RGB，归一化到（0-1）之间
        im_data = np.swapaxes(im_data, 0, 2)
        # 这个是将im_data转换浮点数，其实上一步归一化处理已经是浮点数了，还用这一步吗
        im_data = np.array([im_data], dtype = np.float)
        #PNET要求就是12*12的输入，但是输入不一定会满足这个要求啊
        #所以，注意这个函数是reshape，相当于构造了一个空的PNet.blobs['data']
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        #这一步相当于赋值了，将上一步获得的函数进行填充
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
        #   threshold的值 [0.6, 0.7, 0.7]
        #  boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            #print(boxes[4:9])
            #print('im_data', im_data[0:5, 0:5, 0], '\n')
            #print('prob1', out['prob1'][0,0,0:3,0:3])

            pick = nms(boxes, 0.5, 'Union') 

            if len(pick) > 0 :
                boxes = boxes[pick, :]#相当于将boxes中，通过nms的候选框提取出来

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         
    #np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    print("[1]total_boxes.shape[0]:\n",total_boxes.shape[0])
    #print(total_boxes)
    #return total_boxes, [] 


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        print("[2]:",total_boxes.shape[0])
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4] #t5应该是score，分数
        total_boxes = np.array([t1,t2,t3,t4,t5]).T #注意这里有一个转置
        #print("[3]:",total_boxes.shape[0])
        #print(regh)
        #print(regw)
        #print('t1',t1)
        #print(total_boxes)

        total_boxes = rerec(total_boxes) # convert box to square
        print("[4]经过rerec正方形修正后total_box.shape[0]:\n",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])#均向0靠近
        print("[4.5]:",total_boxes.shape[0])
        #print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)# 这一步是干嘛的？

    #print(total_boxes.shape)
    #print(total_boxes)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print('tmph', tmph)
        #print('tmpw', tmpw)
        #print("y,ey,x,ex", y, ey, x, ex, )
        #print("edy", edy)

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
          
            #print("dx[k], edx[k]:", dx[k], edx[k])
            #print("dy[k], edy[k]:", dy[k], edy[k])
            #print("img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape)
            #print("tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape)

            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            #print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            #print("tmp", tmp.shape)
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print('tempimg', tempimg[k,:,:,:].shape)
            #print(tempimg[k,0:5,0:5,0] )
            #print(tempimg[k,0:5,0:5,1] )
            #print(tempimg[k,0:5,0:5,2] )
            #print(k)
    
        #print(tempimg.shape)
        #print(tempimg[0,0,0,:])
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print(tempimg[0,:,0,0])
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        #print(out['conv5-2'].shape)
        #print(out['prob1'].shape)

        score = out['prob1'][:,1]
        #print('score', score)
        pass_t = np.where(score>threshold[1])[0]
        #print('pass_t', pass_t)
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        print("[5]:",total_boxes.shape[0])
        #print(total_boxes)

        #print("1.5:",total_boxes.shape)
        
        mv = out['conv5-2'][pass_t, :].T
        #print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                print("[8]:",total_boxes.shape[0])
            
        #####
        # 2 #
        #####
        print("2:",total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print('tmpw', tmpw)
            #print('tmph', tmph)
            #print('y ', y)
            #print('ey', ey)
            #print('x ', x)
            #print('ex', ex)
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            print("[9]:",total_boxes.shape[0])
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print(pick)
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    print("3:",total_boxes.shape)

    return total_boxes, points




    
def initFaceDetector():
    minsize = 20
    caffe_model_path = "/home/duino/iactive/mtcnn/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    return (minsize, PNet, RNet, ONet, threshold, factor)

def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    threshold = facedetector[4]
    factor = facedetector[5]
    
    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    
    #tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    #toc()
    containFace = (True, False)[boundingboxes.shape[0]==0]
    return containFace, boundingboxes

def main():
    #imglistfile = "./file.txt"
    # 读取图片信息
    imglistfile = "imglist.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/all.txt"
    #imglistfile = "./imglist.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/file_n.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/file.txt"
    # 估计这个是控制每一批次最多输入的图片数量
    minsize = 20

    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    # 这一句制定了使用CPU
    # caffe.set_mode_gpu();#使用GPU并指定gpu_id
    # caffe.set_device(gpu_id);
    caffe.set_mode_cpu()

    # net = caffe.Net(model, weights, 'test'); % create net and load weights
    # net = caffe.Net(网络的定义文件， caffemodel的权值保存文件，选择：caffe.TEST)，
    # 因为一个.prototxt文件中可以即定义train，也定义test，对应的caffe.TRAIN与caffe.TEST.
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)


    #error = []
    # imglistfile = "imglist.txt" 这个前面已经定义完成
    f = open(imglistfile, 'r')
    for imgpath in f.readlines():
        # file.readlines([size]) ：返回包含size行的列表, size 未指定则返回全部行。
        # split()str.split(str="", num=string.count(str)).
        # 参数
        # str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
        # num -- 分割次数。
        # 返回分割后的字符串列表。
        imgpath = imgpath.split('\n')[0]
        print("######\n", imgpath)
        #cv2.imread(path, flag)：读入图片，共两个参数，第一个参数为要读入的图片文件名，
        #第二个参数为如何读取图片，包括cv2.IMREAD_COLOR：读入一副彩色图片；
        #cv2.IMREAD_GRAYSCALE：以灰度模式读入图片；
        #cv2.IMREAD_UNCHANGED：读入一幅图片，并包括其alpha通道
        #flags = -1：imread按解码得到的方式读入图像
        #flags = 0：imread按单通道的方式读入图像，即灰白图像
        #flags = 1：imread按三通道方式读入图像，即彩色图像

        img = cv2.imread(imgpath)
        # 对图片进行反色处理rgb中，r和b调换，不太明白为何如此操作：OpenCV通道分离是按照BGR顺序
        # 而为什么opencv选择的是BGR是因为OpenCV的早期开发当时BGR颜色格式在相机制造商和软件提供商中很受欢迎。
        img_matlab = img.copy()
        #img_matlab[:,:,2] 这个是灰色的
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp

        # check rgb position
        #tic()
        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
        #toc()

        ## copy img to positive folder
        #if boundingboxes.shape[0] > 0 :
        #    import shutil
        #    shutil.copy(imgpath, '/home/duino/Videos/3/disdata/positive/'+os.path.split(imgpath)[1] )
        #else:
        #    import shutil
        #    shutil.copy(imgpath, '/home/duino/Videos/3/disdata/negetive/'+os.path.split(imgpath)[1] )

        # useless org source use wrong values from boundingboxes,case uselsee rect is drawed 
#        for i in range(len(boundingboxes)):
#            cv2.rectangle(img, (int(boundingboxes[i][0]), int(boundingboxes[i][1])), (int(boundingboxes[i][2]), int(boundingboxes[i][3])), (0,255,0), 1)    
        


        img = drawBoxes(img, boundingboxes)
        cv2.imshow('img', img)
        ch = cv2.waitKey(0) & 0xFF
        if ch == 27:
            break


        #if boundingboxes.shape[0] > 0:
        #    error.append[imgpath]
    #print(error)
    f.close()
# if __name__ == '__main__'的意思是：
# 当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
if __name__ == "__main__":
    main()
