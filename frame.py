import numpy as np
import cv2
# import upload
cap = cv2.VideoCapture(0)
count = 0
i = 0
# import glob
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count +=1
    # Display the resulting frame
    while(count==15):
        cv2.imwrite('test'+str(i)+'.jpg', frame)
        count=0
        i += 1
        # cv2.imshow("frame",frame)
        # count += 1
    print(count)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# for name in glob.glob('/home/akhilesh/Server/frame?.jpg'):
    # print(name)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
