from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
GITHUB_EX_RTSP = 'rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream'
GITHUB_EX_RTSP2 = 'rtsp://mamun:123456@101.134.16.117:554/user=mamun_password=123456_channel=0_stream=0.sdp'
MY_RTSP = 'rtsp://prezzie77:1q2w3e4r5t@192.168.219.102:554/stream2'
camera = cv2.VideoCapture(MY_RTSP)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)




def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame

        print('type(frame):{}'.format(type(frame)))     # type(frame):<class 'numpy.ndarray'>
        print('frame.shape:{}'.format(frame.shape))     # frame.shape:(360, 640, 3)
        print('frame:{}'.format(frame))

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index2.html')


if __name__ == '__main__':
    print('########## started!!!!!!!')
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=7000)