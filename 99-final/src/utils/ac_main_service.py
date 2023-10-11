"""
🚀
🚀
🚀

■ 01 redis 서버 실행
vi /etc/redis/redis.conf
stop-writes-on-bgsave-error
redis-server /etc/redis/redis.conf

■ 02 celery 워커 기동
celery --app=ac_infer_task worker -l INFO --concurrency=1

■ 03 Flask 실행
gunicorn -b 0.0.0.0:7000 ac_main_service:detect_app -k gthread -w 1 --threads 2
gunicorn -b 0.0.0.0:7000 ac_main_service:detect_app

🚀
🚀
🚀
"""
from datetime import datetime
print('{} # # 01 Start service..'.format(datetime.now()))
from flask import Flask, render_template, Response, request, redirect, url_for

from ac_infer_task import StreamInfer, BackgroundInferTask

detect_app = Flask(__name__)

# 실시간 detection 결과 이미지 스트리밍하는 클래스의 인스턴스
stream_infer = StreamInfer()
stream_infer.load_model_n_get_dataloader()



# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# record
# task 실행
# 얘는 post 로 하자. 저장 시간(9to18), 주기(5초) 등 값 받기
# task.delay 로 celery worker 실행시키기
@detect_app.route('/record')
def record():
    print('{} @ @ 1 record request:{}'.format(datetime.now(), request))

    # 백그라운드 detection 추론 결과 이미지 저장하는 celery 클래스의 task 인스턴스
    background_infer_task = BackgroundInferTask()
    background_infer_task.param_init()
    background_infer_task.load_model_n_get_dataloader()
    result = background_infer_task.delay('asdfasdf')
    return 'Started Background Infer !! task id : {}'.format(result)

# task status 값 리턴
# @detect_app.route('/record')
# def video_stream():

# task 죽이기
# @detect_app.route('/record')
# def video_stream():


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# video stream
# video_stream 의 index2 로 계속 feed
@detect_app.route('/video_feed')
def video_feed():
    print('{} @ @ 2 video_feed request:{}'.format(datetime.now(), request))
    stream_infer_gen = stream_infer.run()
    return Response(stream_infer_gen, mimetype='multipart/x-mixed-replace; boundary=frame')

# main 페이지에서 stream 버튼 클릭 시 redirect
@detect_app.route('/video_stream')
def video_stream():
    print('{} @ @ 1 video_stream request:{}'.format(datetime.now(), request))
    """Video streaming home page."""
    return render_template('index2.html')

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# main page
@detect_app.route('/')
def index():
    print('{} @ @ 0 main page request:{}'.format(datetime.now(), request))
    """Main page."""
    return render_template('index.html')


if __name__ == '__main__':
	detect_app.run(host='0.0.0.0', port=7000)














