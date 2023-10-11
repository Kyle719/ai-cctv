import cv2

GITHUB_EX_RTSP2 = 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=0_stream=0.sdp'
MY_RTSP = 'rtsp://192.168.219.102/stream2'
MY_RTSP = 'rtsp://prezzie77:1q2w3e4r5t@192.168.219.102:554/stream2'

cap = cv2.VideoCapture(MY_RTSP)

ret, frame = cap.read()
asdf = True
if cap.isOpened():
    _,frame = cap.read()
    cap.release() #releasing camera immediately after capturing picture
    if _ and frame is not None and asdf==True:
        cv2.imwrite('images/latest.jpg', frame)
        asdf = False