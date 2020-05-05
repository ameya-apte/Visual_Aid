import pyrealsense2 as rs
import cv2
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imshow("frame",frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    
cap.release()
cv2.destroyAllWindows()

ctx = rs.context()
if len(ctx.devices) > 0:
    for d in ctx.devices:

        print ('Found device: ', \

            d.get_info(rs.camera_info.name), ' ', \

            d.get_info(rs.camera_info.serial_number))

else:

    print("No Intel Device connected")