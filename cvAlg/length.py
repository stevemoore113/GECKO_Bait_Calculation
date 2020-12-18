from sklearn.cluster import AgglomerativeClustering
from matplotlib import cm
import matplotlib    as  mpl
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2
import tensorflow as tf
import numpy as np
import scipy 
import itertools
from sklearn.cluster import DBSCAN
import mysql.connector
from mysql.connector import errorcode


bg   = cv2.imread('0326_640X480_mask_echo.png')
AdaptiveThreshold_C = -24
MorphologyOpen      = 3
AreaThr             = 20
pixel_to_cm         = 900/640
square              = 4
pi                  = 3.14
#mask2 = mask2[h_range[0]:h_range[1],w_range[0]:w_range[1]]

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
    gray = np.where(gray.astype(int)-mask2.astype(int)>=8,gray,np.uint8(0))
    #gray = 255 - gray
    cv2.imshow('gray',gray)
    thres=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,AdaptiveThreshold_C)
    thres = cv2.morphologyEx(thres,cv2.MORPH_OPEN,None,MorphologyOpen)
    cv2.imshow('thres',thres)
    _, contours, hierarchy  = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contours,_= cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(a_frame,contours,-1,(255,255,255),1)
    avg_length = 0
    avg_width  = 0
    total      = 0
    centerlist =np.empty(shape=[0, 2])
    #print(contours)
    for c in contours:
            a = cv2.contourArea(c)
            if a >= AreaThr:                
                center = cv2.moments(c)                
                len_pt1,len_pt2,fish_length,w_pt1,w_pt2,fish_width = fish_length_and_width(c)
                #if fish_length >= 20 and fish_length <= 60:
                #cv2.drawContours(a_frame,[c],0,(255,255,255),1)
                cv2.line(a_frame,(len_pt1[0],len_pt1[1]),(len_pt2[0],len_pt2[1]),(255,255,255), 2)
                avg_length += fish_length
                    #avg_width  += fish_width
                total += 1
                cx = int(center["m10"]/center["m00"]) #算x座標
                cy = int(center["m01"]/center["m00"]) #算y座標
                        #print(cx,cy)
                centlist = np.array([[cx,cy]]) # 魚中心(x,y)
                        
                centerlist = np.vstack((centerlist,centlist)) # 堆疊
    if centerlist.shape[0]>=5:                
        clustering = DBSCAN(eps=50, min_samples=3).fit(centerlist)
        predicted = clustering.labels_  # 分群
        #print('center',centerlist)
        #print('predicted',predicted)
        group_max=np.max(predicted)                     # 最大群數
        times,=predicted.shape #一幀有幾個contours

        cmap = plt.get_cmap('tab20') #colormap 
        cmaplist = [cmap(i) for i in range(cmap.N)] #20種顏色 [B,G,R,1.0]
        '''
        for i in range(times):
            centerx,centery=centerlist[i] 
            cv2.circle(a_frame,(int(centerx),int(centery)), 5 ,(int(cmaplist[predicted[i]][2]*255), 
                                int(cmaplist[predicted[i]][1]*255), int(cmaplist[predicted[i]][0]*255)),-1) #魚中心畫點
        
        for j in range(group_max+1):
            Agroup ,     = np.where(predicted == j)  #把同群拿出來
            #print(Agroup)
            #print('Agroup',Agroup)
            Agroup_num , = Agroup.shape #群有幾隻
            #print(Agroup_num)
            tri_list = []
            if Agroup_num >=5: # 三角形要3點
                for k in range(Agroup_num):
                    tri_list.append(centerlist[Agroup[k]])
                tri       = Delaunay(tri_list) 
                tri_np    = tri.simplices.copy() #轉成NParray，每列有3點
                tri_num   = tri_np.shape[0]     
                for num in range(tri_num): 
                    ctx0,cty0=centerlist[Agroup[tri_np[num][0]]]
                    ctx1,cty1=centerlist[Agroup[tri_np[num][1]]]
                    ctx2,cty2=centerlist[Agroup[tri_np[num][2]]]

                    #cv2.line(a_frame,(int(ctx0),int(cty0)),(int(ctx1),int(cty1)),(255,255,255), 1)
                    #cv2.line(a_frame,(int(ctx0),int(cty0)),(int(ctx2),int(cty2)),(255,255,255), 1)
                    #cv2.line(a_frame,(int(ctx1),int(cty1)),(int(ctx2),int(cty2)),(255,255,255), 1)
         '''


                      
    density = total/(square*square)
    return  total,avg_length,density
    
def len_weight(leng):
    y=3.44*leng**2-120.2*leng+1269.74
    return y


video     = '2020622131360_video.mp4'
video_out = '2020622131360_video_out.mp4'

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
total2=0
avg2=0
i = 1

Cage_no = 'B15'
Video_Name = '2020622131360_video.mp4'
create_time = '2020_05_26'

leng = 0
while frame_num < total_frame:
    cv2.setTrackbarPos('frame no.','video file',frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
    ret, crop_frame1 = cap.read() 
    if ret==False:
        break
    #crop_frame = frame[h_range[0]:h_range[1],w_range[0]:w_range[1],:]
    crop_frame1 = cv2.bitwise_and(crop_frame1,bg) #遮圖
    #crop_frame1 = crop_frame[h_range[0]:h_range[1],w_range[0]:w_range[1],:]
    cv2.imshow('crop',crop_frame1)
    total1,avg1,dens=process(crop_frame1)
    #print(total1,avg1)
    total2 += total1
    avg2   += avg1
    if(total2!=0):
        leng    = pixel_to_cm*(avg2/total2)
    weight  = len_weight(leng)
    if total2>0:
        cv2.putText(crop_frame1, 'avg length:{:.2f}cm  amount:{:d}  avg weight:{:.2f}g'.format(leng,total1,weight), 
                    (10, crop_frame1.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(crop_frame1, 'avg length: 0 cm     amount: 0    avg weight: 0', 
                    (10, crop_frame1.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    #crop_frame1 = crop_frame1[h_range[0]:h_range[1],w_range[0]:w_range[1]]
    if video_out is not None and out is None:
        out = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                            (int(crop_frame1.shape[1]),int(crop_frame1.shape[0])))
    cv2.imshow('result',crop_frame1)
    
    out.write(crop_frame1)
    
    
    
    try:
        cnx = mysql.connector.connect(user='lu3', password='evDBVuMvYRYN2e17',
                                    host='140.121.197.165',
                                    port='3306', database='ai_fish')

        mycursor = cnx.cursor() #開啟連線
        sql = ("INSERT INTO `sonar_record`(`Cage_no`, `Video_Name`, `frame_num`, `len`, `weight`, `num`, `create_time`) VALUES (%s,%s,%s,%s,%s,%s,%s)")
        val = (Cage_no,Video_Name, frame_num, float(round(leng,2)), float(round(weight,1)), total1, create_time) # 這邊放入你的變數

        mycursor.execute(sql,val)     
        cnx.commit()
        print("1 条记录已插入, ID:", i)
        i = i + 1
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR: 
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        cnx.close()
    
    
    
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    frame_num += 1
    if frame_num == 600:
        break
    
    
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()