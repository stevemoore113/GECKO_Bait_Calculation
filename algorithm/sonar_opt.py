import cv2
import numpy as np
cap = cv2.VideoCapture("1.mp4")
#cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-200)
 
def flow(frame1,frame2):
    AdaptiveThreshold_C = -24
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    
    frame1 = cv2.medianBlur(frame1, 3)
    frame2 = cv2.medianBlur(frame2, 3)
    
    # list(set(a).intersection(set(b)))
    thres1= cv2.adaptiveThreshold(frame1,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,AdaptiveThreshold_C)
    thres2= cv2.adaptiveThreshold(frame2,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,AdaptiveThreshold_C)
    flow = cv2.calcOpticalFlowFarneback(frame1*thres1,frame2*thres2, None, 0.5, 4, 25, 5, 5, 1.2, 0)
   
    cv2.imshow('thr',thres1*255)
    flow = abs(flow[:,:,0]) + abs(flow[:,:,1])
    
    flow = flow.ravel()    
    flow = np.mean(np.sort(flow)[flow.size//100*95:])
    #print(int(flow[2]*100))
    #ans = flow[2]
 
    return flow

cc = 0
lst = []

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('total frame:', total_frames)

segment = 20

for seq in range(1,3):

    starting_frame = total_frames//segment*seq-100
    if starting_frame < 0:
        starting_frame = 0
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    ret, frame1 = cap.read()
    
    frame1 = frame1[50:200,120:430,:]
            
    for idx in range(300):
        
        ret, frame2 = cap.read()
        if not ret:
            break
        
        frame2 = frame2[50:200,120:430,:]
    
        flowss = flow(frame1,frame2)
        print(seq, flowss)
    
        disp = frame2.copy()
        disp = cv2.medianBlur(disp,3)
        cv2.putText(disp,'{} {:.2f}'.format(starting_frame+idx,flowss),(40,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,100),1,cv2.LINE_AA)
        cv2.imshow('frame2',disp)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalhsv.png',rgb)
            
        lst.append(flowss)        
        cc+=1
        
        frame1 = frame2
    
cap.release()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt

avg = np.zeros((len(lst),))
acc = np.zeros((len(avg)+1,))
for i in range(1,len(acc)):
    acc[i] = acc[i-1] + lst[i-1]
    
for i in range(len(avg)):
    if i >= 5 and i < len(acc)-6:
        avg[i] = (acc[i+6]-acc[i-5])/11
    elif i < 5:
        avg[i] = (acc[i+6]-acc[0])/(i+6)
    else:
        avg[i] = (acc[-1]-acc[i-5])/(5+len(acc)-i)
        
plt.figure(figsize=(5,5))
plt.plot([i for i in range(len(avg))],avg)
plt.show()
