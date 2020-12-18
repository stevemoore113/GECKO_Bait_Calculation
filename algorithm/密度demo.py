# -*- coding: utf-8 -*-
"""
Created on Fri May 29 02:14:38 2020

@author: lab403
"""


import cv2
import tensorflow as tf
import numpy as np
import scipy 
import itertools
from websocket_server import WebsocketServer
import threading
import base64
import socket



#讀影片與總frame數
cap = cv2.VideoCapture('number.mp4') 
bg=cv2.imread('mask.png') #遮罩
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
def base64_cv2(base64_str):
        imgString = base64.b64decode(base64_str)
        nparr = np.frombuffer(imgString,np.uint8)  
        image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        return image
    
    
def recvall(sock, cou):
         buf = b''
         while cou:
             newbuf = sock.recv(cou)
             if not newbuf: return None
             buf += newbuf
             cou -= len(newbuf)
         return buf
     
def set_frame_number(x):
    global frame_num
    frame_num = x
    return


# 創建滑動條 (滑條的名稱,window的名稱,最小值,最大值,回調func)
frame_num = 12110
while frame_num < total_frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num) #捕獲下一幀 
    ret, frame = cap.read() # 擷取一張影像 ret:有沒有讀到圖片
        
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
    if(frame_num < 1001 and avgg == False):
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
    
    
    if(count>1000):
        if(avgg == False):
            for j in range(0,height):
                for i in range(0,width):
                    #test_list[j][i] = test_list[j][i]/100
                    mask2[j][i] = test_list[j][i]/1000
                    mask2[j][i] = 255 - mask2[j][i]
            avgg = True
            
        #thmask = cv2.resize(thmask,(width,height),interpolation=cv2.INTER_CUBIC)
        #thmask = cv2.cvtColor(thmask,cv2.COLOR_BGR2GRAY)
        #dithmask = cv2.dilate(thmask,kernel,iterations = 1)
        #ret,dithmask = cv2.threshold(dithmask,160,255,cv2.THRESH_BINARY_INV)
        #checkmask = cv2.bitwise_and(grayframe, thmask) 
        
        
        checkmask = cv2.bitwise_and(grayframe, mask2) #做二進制and
        #checkmask1 = cv2.cvtColor(checkmask,cv2.COLOR_GRAY2BGR)
        

        #print(mask2.shape)

        
        
        #cv2.imshow('thborder2',thborder2)
        #cv2.imshow('thmask',dithmask)
        #cv2.imshow('thmask',check)
   
    key = cv2.waitKey(1) & 0xFF  #按ESC跳掉
    if key == 27:
        break
    frame_num += 1
   
        
cap.release()
cv2.destroyAllWindows()
from sklearn.cluster import AgglomerativeClustering
from matplotlib import cm
import matplotlib    as  mpl
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

AdaptiveThreshold_C = -24
MorphologyOpen      = 3
AreaThr             = 10
pixel_to_cm         = 3400/640
#h_range=[50,-150]
#w_range=[50,-90]

group_dist = 100
def mysql_Connect(total,avg_length,avg_width):
     db  =  MySQLdb . connect ( host = "140.121.197.160" ,user = "steve" ,passwd = "123123" ,  db = "ai_fish" )         #數據庫名稱
     sql_string = "INSERT INTO `sonar_data`(`sonar_name`, `fish_num`, `insert_time`, `avg_lenth`, `avg_width`) VALUES ('sonar_a1','"+str(total)+"',now(),'"+str(avg_length)+"','"+str(avg_width)+"')"
     cur  =  db . cursor ()
#     print(sql_string)
     cur.execute (sql_string)
     time.sleep(0.1)
     db.commit()
     cur.close()
     db.close ()
# Called for every client connecting (after handshake)
def new_client(client, server):
	print("New client connected and was given id %d" % client['id'])
    
	

def base64_cv2(base64_str):
        imgString = base64.b64decode(base64_str)
        nparr = np.frombuffer(imgString,np.uint8)  
        image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        return image
    
    
def recvall(sock, cou):
         buf = b''
         while cou:
             newbuf = sock.recv(cou)
             if not newbuf: return None
             buf += newbuf
             cou -= len(newbuf)
         return buf
     
        
def client_left(client, server):
	print("Client(%d) disconnected" % client['id'])


def message_received(client, server, message):
	if len(message) > 200:
		message = message[:200]+'..'
	print("Client(%d) said: %s" % (client['id'], message))
	global camera1
	camera1 = cv2.VideoCapture(message)

def from_vedio():
	thread3 = threading.Thread(target=playsss, args=(1,))
	thread3.start()
	print('start')
    
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

    thres=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,AdaptiveThreshold_C)
    thres = cv2.morphologyEx(thres,cv2.MORPH_OPEN,None,MorphologyOpen)

    contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
                center = cv2.moments(c) #拿輪廓
                len_pt1,len_pt2,fish_length,w_pt1,w_pt2,fish_width = fish_length_and_width(c)
                if fish_length >= 10 and fish_length <= 30:
                    #if fish_width > 0 and fish_width*2 < fish_length:
                        
                        
                    cx = int(center["m10"]/center["m00"]) #算x座標
                    cy = int(center["m01"]/center["m00"]) #算y座標
                    print("9877s")
                        #print(cx,cy)
                    centlist = np.array([[cx,cy]]) # 魚中心(x,y)
                    cv2.circle(a_frame,(int(cx),int(cy)), 1 ,(int(150), 
                                                           int(207), int(87)),13) #魚中心畫點
                    centerlist = np.vstack((centerlist,centlist)) # 堆疊


    cv2.putText(a_frame, 'Quantity:{:d}'.format(centerlist.shape[0]), 
                (60, a_frame.shape[0]-60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return   a_frame , centerlist.shape[0]
    
    
def playsss(n):
	global camera1
	img = cv2.imread('mask.png')
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(('140.121.197.160', 8078))
	s.listen(True)
	conn, addr = s.accept()   
	global frame
	try:
	       while True:
		    	    length = recvall(conn,16)
		    	    length = length.decode()
		    	    length = int(length)
		    	    video_str = recvall(conn,length)
		    	    video_str = video_str.decode()
		    	    imgdata=base64_cv2(str(video_str))
		    	    prev = imgdata
		    	    image2 = prev
		    	    image2 =  cv2.resize(image2,(600,338),interpolation = cv2.INTER_AREA)  
		    	    image2 = cv2.imencode('.jpg',image2)[1]
		    	    base64_data2 = base64.b64encode(image2)
		    	    frame,ss = process(prev)
		    	    frame =  cv2.resize(frame,(600,338),interpolation = cv2.INTER_AREA)                    
		    	    image = cv2.imencode('.jpg', frame)[1]
		    	    base64_data = base64.b64encode(image)
		    	    s = base64_data.decode()
		    	    s2 = base64_data2.decode()
		    	    data = "{\"data3\":\""+s+"\",\"data4\":\""+s2+"\",\"data12\":\""+str(ss)+"\"}"            
		    	    server.send_message_to_all(data)
	except:
                    print("斷線")
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(('140.121.197.160', 8078))
                    s.listen(True)
                    conn, addr = s.accept()

# Server Port
PORT=8993
# 建立Websocket Server
server = WebsocketServer(PORT,'140.121.197.160')
from_vedio()
# 有裝置連線上了
server.set_fn_new_client(new_client)
# 斷開連線
server.set_fn_client_left(client_left)
# 接收到資訊
server.set_fn_message_received(message_received)
# 開始監聽
server.run_forever()