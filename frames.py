import numpy as np
import cv2
import keyboard
cap = cv2.VideoCapture(0)
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    if count==15:
    	if keyboard.press_and_release('p,e'):
    		cv2.imwrite("frame%d.jpg" % count, frame)
    		count=0
    	# cv2.imshow("frame",frame)
    count += 1
    print(count)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()