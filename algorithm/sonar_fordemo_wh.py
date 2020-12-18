from websocket_server import WebsocketServer
import threading
import cv2
import base64
import time
import statistics
import socket
import time
camera1=None
import numpy as np
AdaptiveThreshold_C = -24
MorphologyOpen      = 3
AreaThr             = 25
#pixel 5*200
pixel_to_cm         = 1000/640
rtsp_path="len.mp4"
import  MySQLdb
def base64_cv2(self,base64_str):
        imgString = base64.b64decode(base64_str)
        nparr = np.frombuffer(imgString,np.uint8)  
        image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        return image
# In[6]        
def recvall(self,sock, cou):
         buf = b''
         while cou:
             newbuf = sock.recv(cou)
             if not newbuf: return None
             buf += newbuf
             cou -= len(newbuf)
         return buf

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
	thread2 = threading.Thread(target=vedio_thread2, args=(1,))
	thread2.start()
	print('start')

def vedio_thread2(n):
	global camera1
	img = cv2.imread('mask.png')
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(('140.121.197.160', 8077))
	s.listen(True)
	conn, addr = s.accept()
	global frame
	time.sleep(2)

#	conn,addr = connect_socket('140.121.197.160',"8077")
	try:
		      while True:
		    	    length = recvall(conn,16)
		    	    length = length.decode()
		    	    length = int(length)
		    	    video_str = recvall(conn,length)
		    	    video_str = video_str.decode()
		    	    imgdata=base64_cv2(str(video_str))
		    	    prev = imgdata
		    	    xxx2 = cv2.resize(prev,(1280,960),interpolation = cv2.INTER_AREA)
		    	    xxx2 = xxx2[110:670,130:1025,0:3]
		    	    frame = process(prev,np.array(img))
		    	    frame,xxx2 = cv2.resize(frame,(600,338),interpolation = cv2.INTER_AREA),cv2.resize(xxx2,(600,338),interpolation = cv2.INTER_AREA)
		    	    image = cv2.imencode('.jpg', frame)[1]
		    	    image2 = cv2.imencode('.jpg', xxx2)[1]
		    	    base64_data2 = base64.b64encode(image2)
		    	    base64_data = base64.b64encode(image)
		    	    s = base64_data.decode()
		    	    s2 = base64_data2.decode()
		    	    data = "{\"data1\":\""+s+"\",\"data2\":\""+s2+"\"}"    
		    	    print("123")
		    	    server.send_message_to_all(data)
	except  Exception as e:
                    print(e)
                    time.sleep(4)
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.bind(('140.121.197.160', 8077))
                    s.listen(True)
                    conn, addr = s.accept()



                    
def fish_length_and_width(contour):
    c   = np.squeeze(contour)
    n2  = np.sum(c**2,axis=1)
    dist = np.reshape(n2,(-1,1))+np.reshape(n2,(1,-1))-2*np.dot(c,c.T)
    idx = np.argmax(dist.ravel())
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

def process(a_frame,img):
    global lens,wid
    gray = cv2.cvtColor(a_frame,cv2.COLOR_BGR2GRAY)
    thres=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,AdaptiveThreshold_C)
    thres = cv2.morphologyEx(thres,cv2.MORPH_OPEN,None,MorphologyOpen)
    #contours,_= cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(a_frame,contours,-1,(255,255,255),1)
    avg_length = 0
    avg_width  = 0
    total      = 0
    list_fish = []
#    list_show = []
    fs = 305
    xs = 70
    loss123 = 0
    for c in contours:
        a = cv2.contourArea(c)
        if a >= AreaThr-50:
            len_pt1,len_pt2,fish_length,w_pt1,w_pt2,fish_width = fish_length_and_width(c)
            if fish_length >= 8 and fish_length <= 35 and fish_length>fish_width:
                if (fish_width > 0 and fish_width*2 < fish_length):
                    cv2.drawContours(a_frame, [c], -1, (255,255,255), thickness=1)
                    cv2.line(a_frame,(len_pt1[0],len_pt1[1]),(len_pt2[0],len_pt2[1]),(255,255,255), 1)
                    cv2.line(a_frame,(w_pt1[0],w_pt1[1]),(w_pt2[0],w_pt2[1]),(255,255,255), 1)
                    (x, y, w, h) = cv2.boundingRect(c)
#                    cv2.rectangle(a_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    print(a_frame.shape)
                    print(a_frame[fs:fs+h,xs:xs+w,:].shape,xs+w,a_frame[y:y+h,x:x+w,:].shape)
                    loss123 = loss123+1
                    if(xs+w<=580 and loss123>=10 and (y+h+3)/(x+w+3)<=1):
                      cv2.rectangle(a_frame, (x-5, y-5), (x + w+3, y + h+3), (0, 255, 255), 1)
#                      cv2.line(a_frame,(x + w+3, y + h+3),(xs,fs),(255,200,25), 1)
                      a_frame[fs:fs+h,xs:xs+w,:] =  a_frame[y:y+h,x:x+w,:]
                      xs =xs +w +5
                    avg_length += fish_length
                    avg_width  += fish_width
                    total += 1
#                    x_cc = np.array([x,y,w,h])
                    list_fish.append(fish_length)
#                    list_show.append(x_cc)
                    lens = 5
                    
    if(total>=10):
       list_fish.sort()
       total_all = total
       total2 =(int)(total/8)   
       list_fish  =  list_fish[total_all - total2:]
#       print(list_fish)
       lens = statistics.mean(list_fish) 


            
    #comp,num  = scipy.ndimage.measurements.label(thres)
    
    if total > 0:
        avg_length /= total
        avg_width  /= total        
        cv2.putText(a_frame, 'avg length:{:.2f}cm avg width:{:.2f}cm'.format(pixel_to_cm*avg_length,pixel_to_cm*avg_width),\
                    (10, a_frame.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1, (25, 255, 255), 2, cv2.LINE_AA)
#        total取向幾隻魚
    lens = 5
    mysql_Connect(total,lens,avg_width)      
    a_frame = cv2.bitwise_and(a_frame,img)
    #cv2.imshow('components',np.clip(comp,0,255).astype(np.uint8))
    #print(num,len(contours))
#    print(a_frame.shape)
    a_frame = cv2.resize(a_frame,(1280,960),interpolation = cv2.INTER_AREA)
    a_frame = a_frame[110:670,130:1025,0:3]
    
    fs,xs = 0,0
#    fs = 130
#    xs = 500
#    for i in list_show:     
##        print(i[0].shape)
##        print(a,b,c,d)
#        if(xs<590 and fs<(1025-130)):
#            a,b,cs,d =i[0],i[1],i[2],i[3]
#            if(a_frame[xs:xs+cs,fs:fs+d,:].shape ==  a_frame[a:a+cs,b:b+d,:].shape):
#                 a_frame[xs:xs+cs,fs:fs+d,:] =  a_frame[a:a+cs,b:b+d,:]
#                 cv2.rectangle(a_frame, (a, b), (a + cs, b + d), (0, 255, 0), 1)
#                 fs = fs+100
#                 print(xs,fs)
#    list_show  = []     
    return  a_frame

# Server Port
PORT=8999
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
