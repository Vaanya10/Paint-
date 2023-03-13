import numpy as np
import cv2
from collections import deque

blueLower = np.array([100,60,60])
blueUpper = np.array([140,255,255])

kernel = np.ones((5,5),np.uint8)

bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0 
gindex = 0
rindex = 0
yindex = 0

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorindex = 0

paintWindow = np.zeros((471,636,3))+255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65),(0,0,0),2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65),colors[0],-1)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65),colors[1],-2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65),colors[2],-1)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65),colors[3],-1)

cv2.putText(paintWindow, "Clear all", (49,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Blue", (185,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Green", (298,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Red", (420,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Yellow", (520,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (155,155,155),2,cv2.LINE_AA)
cv2.namedWindow('Paint',cv2.WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)


while True:
    (hasFrame, Frame) = camera.read()
    Frame = cv2.flip(Frame,1)
    hsv = cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)

    Frame = cv2.rectangle(Frame, (40,1), (140,65),(0,0,0),2)
    Frame = cv2.rectangle(Frame, (160,1), (255,65),colors[0],-1)
    Frame = cv2.rectangle(Frame, (275,1), (370,65),colors[1],-2)
    Frame = cv2.rectangle(Frame, (390,1), (485,65),colors[2],-1)
    Frame = cv2.rectangle(Frame, (505,1), (600,65),colors[3],-1)

    cv2.putText(Frame, "Clear all", (49,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(Frame, "Blue", (185,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(Frame, "Green", (298,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(Frame, "Red", (420,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(Frame, "Yellow", (520,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (155,155,155),2,cv2.LINE_AA)

    if not hasFrame:
        break 
    blueMask = cv2.inRange(hsv,blueLower,blueUpper)
    blueMask = cv2.erode(blueMask,kernel,iterations=2)
    blueMask = cv2.morphologyEx(blueMask,cv2.MORPH_OPEN,kernel)
    blueMask = cv2.dilate(blueMask,kernel,iterations=1)

    (cnts,_)= cv2.findContours(blueMask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cntr = None
    if len(cnts)>0:
        cnt= sorted(cnts,key=cv2.contourArea,reverse=True) [0]
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(Frame,(int(x),int(y)),int(radius),(0,255,255),2)
        M = cv2.moments(cnt)
        cntr = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) 
        if cntr[1]<= 65:
            if 40<=cntr[0]<=140:
                bpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                bindex = 0
                rindex = 0
                yindex = 0
                gindex = 0
                paintWindow[67:,:,:] = 255
            elif 160<= cntr[0] <= 255:
                cntrindex=0
            elif 275<= cntr[0] <= 370:
                colorindex = 1
            elif 390<= cntr[0] <=485:
                colorindex =2
            elif 505<= cntr[0] <= 600:
                colorindex =3
        else:
            if colorindex ==0:
                bpoints[bindex].appendleft (cntr)
            elif colorindex == 1: 
                gpoints[gindex].appendleft (cntr)
            elif colorindex == 2:
                rpoints[rindex].appendleft (cntr)
            elif colorindex == 3:
                ypoints[yindex].appendleft (cntr)
    points = [bpoints,gpoints,rpoints,ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(Frame,points[i][j][k-1], points[i][j][k], colors[i],2)
                cv2.line(paintWindow,points[i][j][k-1], points[i][j][k], colors[i],2)



    cv2.imshow('Frame',Frame)
    cv2.imshow('paint', paintWindow)
    if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cv2.destroyAllWindows()