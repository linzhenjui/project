import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# set orange thresh
lower_orange=np.array([0,30,30])
upper_orange=np.array([100,255,255])
while(1):
    # get a frame and show
    ret, frame = cap.read()

    # change to hsv model
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv2,(11,11),0)
	
	# get mask
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    blur = cv2.GaussianBlur(mask, (11,11), 0)
    edge = cv2.Canny(blur, 20, 160)
    _, cnts, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
      M= cv2.moments(c)
      if M["m00"] != 0:
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
        area = cv2.contourArea(c)
        if area > 1500:
          cv2.circle(frame,(cX,cY), 5, (1,1,254), -1)
          print(cX,cY)
    # detect orange
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
