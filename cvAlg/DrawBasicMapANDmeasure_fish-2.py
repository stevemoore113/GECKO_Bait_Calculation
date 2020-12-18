#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import tensorflow as tf
import numpy as np
import scipy 
import itertools


#h_range=[50,-100]
#w_range=[50,-50]
#讀影片與總frame數
cap = cv2.VideoCapture('C:\\Users\\10857\\Desktop\\sonar_python\\2020223B4池_1_video_Trim.mp4') 
bg=cv2.imread('C:\\Users\\10857\\Desktop\\sonar_python\\0318_640X480_mask.png') #讀圖片
mask2 = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY) #BGR TO GRAY 
kernel = np.ones((7,7),np.uint8) # 7*7,8位無符號整數
count = 0
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #frame的高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #frame的寬度
print(height,width) 
avgg = False 
frame_num = 0
total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #幀數
print(total_frame)
test_list=[[0] * 640 for i in range(480) ] #row 640個[0] * column 480

def set_frame_number(x):
    global frame_num
    frame_num = x
    return

cv2.namedWindow('video file')
cv2.createTrackbar('frame no.','video file',0,total_frame-1,set_frame_number) 
# 創建滑動條 (滑條的名稱,window的名稱,最小值,最大值,回調func)

while frame_num < total_frame:
    cv2.setTrackbarPos('frame no.','video file',frame_num) #設置滑動條位置(滑動條名稱。窗口名稱。新位置。)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num) #捕獲下一幀 
    ret, frame = cap.read() # 擷取一張影像 ret:有沒有讀到圖片
    #frame=frame[h_range[0]:h_range[1],w_range[0]:w_range[1],:]
    #灰階後擷取所需特定區塊影像
    #bitwiseAnd = cv2.bitwise_and(img, frame)
    #print(np.shape(frame))
    frame = cv2.bitwise_and(frame,bg)
    grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #frame:BGR轉gray
    grayframe = cv2.resize(grayframe,(640,480),interpolation=cv2.INTER_CUBIC) #grayframe轉640*480,立方插值
    mask2 = cv2.resize(mask2,(640,480),interpolation=cv2.INTER_CUBIC) 
    #cv2.imshow('mask2',mask2)
    #ret,thborder = cv2.threshold(grayframe,150,255,cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(grayframe,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,45,-80)
    
    height,width = grayframe.shape
    #h,w = thborder.shape
    
    #for a in range(0,height):
        #for b in range(0,width):
    
    #print(grayframe)   
    if(frame_num < 101 and avgg == False):
        for j in range(0,height):
            for i in range(0,width):
                test_list[j][i] = test_list[j][i] + grayframe[j][i]
        count += 1
    if ret==False:
        break
    #cv2.imshow('grayframe',grayframe)
    
    
    #ret,thmask = cv2.threshold(mask1,160,255,cv2.THRESH_BINARY)
    #cv2.imshow('grayframe',grayframe)
    #cv2.imshow('thborder',thborder)
    
    cv2.imshow('video file',grayframe)
    
    if(count>100):
        if(avgg == False):
            for j in range(0,height):
                for i in range(0,width):
                    #test_list[j][i] = test_list[j][i]/100
                    mask2[j][i] = test_list[j][i]/100
                    mask2[j][i] = 255 - mask2[j][i]
            avgg = True
            
        #thmask = cv2.resize(thmask,(width,height),interpolation=cv2.INTER_CUBIC)
        #thmask = cv2.cvtColor(thmask,cv2.COLOR_BGR2GRAY)
        #dithmask = cv2.dilate(thmask,kernel,iterations = 1)
        #ret,dithmask = cv2.threshold(dithmask,160,255,cv2.THRESH_BINARY_INV)
        #checkmask = cv2.bitwise_and(grayframe, thmask) 
        
        
        checkmask = cv2.bitwise_and(grayframe, mask2) #做二進制and
        #checkmask1 = cv2.cvtColor(checkmask,cv2.COLOR_GRAY2BGR)
        
        cv2.imshow('mask2',mask2)
        cv2.imshow('checkmask',checkmask)
        
        
        #cv2.imshow('thborder2',thborder2)
        #cv2.imshow('thmask',dithmask)
        #cv2.imshow('thmask',check)
   
    key = cv2.waitKey(1) & 0xFF  #對影片按ESC跳掉
    if key == 27:
        break
    frame_num += 1
   
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:


AdaptiveThreshold_C = -24
MorphologyOpen      = 3
AreaThr             = 25
pixel_to_cm         = 1000/640


def fish_length_and_width(contour):
    c   = np.squeeze(contour)  # 刪維度
    n2  = np.sum(c**2,axis=1)  # row相加
    dist = np.reshape(n2,(-1,1))+np.reshape(n2,(1,-1))-2*np.dot(c,c.T) #
    idx = np.argmax(dist.ravel())#ravel降為一維 argmax取最大值
    pt1 = idx//dist.shape[1]
    pt2 = idx%dist.shape[1]
    if pt1 > pt2:
        pt1,pt2 = pt2,pt1
        
    v1  = (c[pt1]-c[pt2]).astype(np.float)
    v1 /= np.linalg.norm(v1) 
    
    width = 0
    wpt1  = None
    wpt2  = None
    
    vec   = np.expand_dims(c,axis=0)-np.expand_dims(c,axis=1)
    vecn  = np.linalg.norm(vec,axis=2)
    vecn[np.nonzero(vecn==0)]=1
    cosine= np.abs(np.dot(vec,v1)/vecn)
    candid= np.nonzero(cosine<=0.1)
    maxid = np.argmax(vecn[candid].ravel())
    width = vecn[candid[0][maxid],candid[1][maxid]]
    wpt1  = c[candid[0][maxid]]
    wpt2  = c[candid[1][maxid]]
    return c[pt1],c[pt2],dist[pt1,pt2]**0.5,wpt1,wpt2,width

def process(a_frame):
    gray = cv2.cvtColor(a_frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray,mask2)
    cv2.imshow('gray',gray)
    thres=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,AdaptiveThreshold_C)
    thres = cv2.morphologyEx(thres,cv2.MORPH_OPEN,None,MorphologyOpen)
    cv2.imshow('thres',thres)
    _, contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contours,_= cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(a_frame,contours,-1,(255,255,255),1)
    avg_length = 0
    avg_width  = 0
    total      = 0
    #print(contours)
    for c in contours:
            
            a = cv2.contourArea(c)
            if a >= AreaThr:
                len_pt1,len_pt2,fish_length,w_pt1,w_pt2,fish_width = fish_length_and_width(c)
                if fish_length >= 8 and fish_length <= 30:
                    if fish_width > 0 and fish_width*2 < fish_length:
                        cv2.drawContours(a_frame,[c],0,(255,255,255),1)
                        cv2.line(a_frame,(len_pt1[0],len_pt1[1]),(len_pt2[0],len_pt2[1]),(255,255,255), 2)
                        #cv2.line(a_frame,(w_pt1[0],w_pt1[1]),(w_pt2[0],w_pt2[1]),(255,0,0), 2)
                        avg_length += fish_length
                        avg_width  += fish_width
                        total += 1
    #comp,num  = scipy.ndimage.measurements.label(thres)
    
    if total > 0:
        avg_length /= total
        #avg_width  /= total        
        cv2.putText(a_frame, 'avg length:{:.2f}cm'.format(pixel_to_cm*avg_length), 
                    (10, a_frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    
    #cv2.imshow('components',np.clip(comp,0,255).astype(np.uint8))
    #print(num,len(contours))
    return 
    
    
video     = 'C:\\Users\\10857\\Desktop\\sonar_python\\2020223B4池_1_video_Trim.mp4'
video_out = 'C:\\Users\\10857\\Desktop\\sonar_python\\2020223B4池_1_video_Trim_out.mp4'

#mask3 = cv2.resize(mask2,(540,330),interpolation=cv2.INTER_CUBIC) 
#cv2.imshow('mask3',mask3)
out    = None 
cap    = cv2.VideoCapture(video)
print('height:{} width:{}'.format(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
frame_num = 0
total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def set_frame_number(x):
    global frame_num
    frame_num = x
    return

cv2.namedWindow('video file')
cv2.createTrackbar('frame no.','video file',0,total_frame-1,set_frame_number)


while frame_num < total_frame:
    cv2.setTrackbarPos('frame no.','video file',frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
    ret, crop_frame = cap.read() 
    if ret==False:
        break
    #crop_frame = frame[h_range[0]:h_range[1],w_range[0]:w_range[1],:]
    crop_frame2 = cv2.bitwise_and(crop_frame,bg) #遮圖
    
    cv2.imshow('video file',crop_frame)
    cv2.imshow('crop',crop_frame2)
    
    process(crop_frame2)
    if video_out is not None and out is None:
        out = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                            (int(crop_frame.shape[1]),int(crop_frame.shape[0])))
    cv2.imshow('result',crop_frame2)
    
    out.write(crop_frame2)
    
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    frame_num += 1
    
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()


# In[ ]:





# In[4]:





# In[ ]:





# In[ ]:




