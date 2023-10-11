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
from flask import Flask, render_template, Response, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json

import subprocess

from ac_infer_task import StreamInfer, BackgroundInfer, get_task_info, get_folder_file_list, get_image_bytedata, kill_executing_task

detect_app = Flask(__name__, static_folder='static', template_folder='templates')

detect_app.secret_key = "My_Key"

# 실시간 detection 결과 이미지 스트리밍하는 클래스의 인스턴스
stream_infer = StreamInfer()
stream_infer.load_model_n_get_dataloader()


# 로그인 기능
USER_ID = 'jhs'
USER_PWD = 'jhs123'
login_manager = LoginManager()
login_manager.init_app(detect_app)
login_manager.login_view = 'login'

# 사용자 모델 정의 (예: User 클래스)
class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {USER_ID: {'password': USER_PWD}}  # 실제 프로젝트에서는 데이터베이스를 사용하세요.

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


# flask 전역변수
file_list = []





# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# view recorded images
folder_name = None
RECORDED_IMG_ROOT_DIR = './runs/detect/'
@detect_app.route('/view_folder_list')
@login_required
def view_folder_list():
    print('{} @ @ view_folder_list request:{}'.format(datetime.now(), request))
    folder_list = get_folder_file_list(RECORDED_IMG_ROOT_DIR)
    return render_template('index5.html', folder_list=folder_list)

@detect_app.route('/view_image_list')
@login_required
def view_image_list():
    print('{} @ @ view_image_list request:{}'.format(datetime.now(), request))
    # <Request 'http://192.168.219.103:7000/view_image_list?folder_name=2023-09-25' [GET]>
    folder_name = request.args.get("folder_name")
    global file_list
    file_list = get_folder_file_list('{}{}'.format(RECORDED_IMG_ROOT_DIR, folder_name))
    print('file_list:{}'.format(file_list))
    # file_list:['06:21:01.png', '06:30:48.png', '06:34:25.png', '06:43:43.png', '06:44:03.png', '06:44:11.png', '06:46:22.png', '06:46:28.png', '06:46:43.png', '06:56:54.png', '06:57:05.png', '09:25:05.png', '09:25:12.png', '09:43:21.png', '10:37:05-첫번째추론이미지는무조건저장됩니다.png', '11:05:34-첫번째추론이미지는무조건저장됩니다.png', '11:08:20-첫번째추론이미지는무조건저장됩니다.png']
    return render_template('index6.html', folder_name=folder_name, file_list=file_list)

@detect_app.route('/view_image/<folder_name>/<file_name>')
@login_required
def view_image(folder_name, file_name):
    # 사용자가 이미지 파일 이름을 선택했을때 실행되는곳.
    # 이미지 파일 경로를 받았고, 실제 파일을 읽어서 보여줘야되는곳.
    print('{} @ @ view_image request:{}'.format(datetime.now(), request))
    # <Request 'http://192.168.219.103:7000/view_image/2023-09-25/06:21:01.png' [GET]>
    data = [None, None]
    # data.append(folder_name)
    # data.append(file_name)
    data[0] = folder_name
    data[1] = file_name
    global file_list
    currnet_num = 0
    for i, fi_na in enumerate(file_list):
        if fi_na == file_name :
            currnet_num = i
    if currnet_num < len(file_list) - 1:
        before_file_name, next_file_name = file_list[currnet_num-1], file_list[currnet_num+1]
        print('data:{}'.format(data))   # data:['2023-09-25', '06:21:01.png']
        return render_template('index7.html', img_path=str(data), folder_name=str(folder_name), file_name=str(file_name), bef_img_path=str(before_file_name), nex_img_path=str(next_file_name))
    elif currnet_num >= len(file_list) - 1 :
        before_file_name, next_file_name = file_list[currnet_num-1], file_list[currnet_num+1 - len(file_list)]
        print('data:{}'.format(data))   # data:['2023-09-25', '06:21:01.png']
        return render_template('index7.html', img_path=str(data), folder_name=str(folder_name), file_name=str(file_name), bef_img_path=str(before_file_name), nex_img_path=str(next_file_name))


@detect_app.route('/image_feed/<string:img_dir>')
@login_required
def image_feed(img_dir):
    print('{} @ @ 2 image_feed request:{}'.format(datetime.now(), request))
    # <Request "http://192.168.219.103:7000/image_feed/%5B'2023-09-25',%20'06:21:01.png'%5D" [GET]>
    print('img_dir:{}'.format(img_dir))
    # img_dir:['2023-09-25', '06:21:01.png']
    res_image = get_image_bytedata(img_dir)
    return Response(res_image, mimetype='multipart/x-mixed-replace; boundary=frame')


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# record
# task 실행
# 얘는 post 로 하자. 저장 시간(9to18), 주기(5초) 등 값 받기
# task.delay 로 celery worker 실행시키기
@detect_app.route('/start_recording')
@login_required
def start_recording():
    print('{} @ @ start_recording request:{}'.format(datetime.now(), request))

    active_tasks, _ = get_task_info()
    active_tasks_values = list(active_tasks.values())
    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    if active_task_num > 0 :
        flash("이미 AI가 감시 중입니다")
        return redirect(url_for('recording'))
    else :
        first_infer_flag = True
        result2 = BackgroundInfer.delay('# # Start Background Inference # #', first_infer_flag)
        if len(str(result2)) > 0 :
            flash("AI가 CCTV 영상 감시를 시작하였습니다")
            return redirect(url_for('recording'))

@detect_app.route('/stop_recording')
@login_required
def stop_recording():
    print('{} @ @ stop_recording request:{}'.format(datetime.now(), request))

    active_tasks, _ = get_task_info()
    active_tasks_values = list(active_tasks.values())

    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    if active_task_num > 0 :
        active_tasks_values_id = active_tasks_values[0][0]['id']
        print(f'Start killing executing task id:{active_tasks_values_id}')
        _ = kill_executing_task(active_tasks_values_id)
        flash("AI가 CCTV 영상 감시를 중지하였습니다")
        return redirect(url_for('recording'))
    else :
        flash("감시중인 AI가 없습니다")
        return redirect(url_for('recording'))

# task state 값 리턴
# recording 잘 돌고 있다/아니다
@detect_app.route('/get_recording_task_state')
@login_required
def get_recording_task_state():
    print('{} @ @ get_recording_task_state request:{}'.format(datetime.now(), request))
    active_tasks, reserved_tasks = get_task_info()
    # print('# # worker_ping_res : {}'.format(worker_ping_res))

    active_tasks_values = list(active_tasks.values())
    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    reserved_tasks_values = list(reserved_tasks.values())
    reserved_task_num = len(reserved_tasks_values[0])
    print('# # reserved_task_num : {}'.format(reserved_task_num))

    if active_task_num > 0 :
        return render_template('index4.html', value1='AI가 CCTV 영상을 감시 중입니다', value2=str(active_tasks_values))
    else :
        flash("감시중인 AI가 없습니다")
        return redirect(url_for('recording'))


@detect_app.route('/edit_task_parameters')
@login_required
def edit_task_parameters():
    print(f'{datetime.now()} @ @ edit_task_parameters request:{request}')
    return render_template('index8.html')


@detect_app.route('/save_editted_task_parameters', methods=['POST'])
@login_required
def save_editted_task_parameters():
    task_parameters = {}
    task_parameters['checkbox_person'] = str(request.form.get('checkbox_person'))
    task_parameters['checkbox_car'] = str(request.form.get('checkbox_car'))
    task_parameters['checkbox_dog'] = str(request.form.get('checkbox_dog'))
    task_parameters['checkbox_cat'] = str(request.form.get('checkbox_cat'))
    task_parameters['checkbox_bird'] = str(request.form.get('checkbox_bird'))
    task_parameters['save_photos'] = str(request.form.get('save_photos'))
    task_parameters['send_msg'] = str(request.form.get('send_msg'))
    task_parameters['redio_speed_accu'] = str(request.form.get('redio_speed_accu'))
    task_parameters['min_save_period'] = str(request.form.get('min_save_period'))
    task_parameters['detc_threshold'] = str(request.form.get('detc_threshold'))

    yy_mm_dd = str((datetime.now()).strftime('%Y-%m-%d'))
    hh_mm_ss = str((datetime.now()).strftime('%H:%M:%S'))
    json_file_nm = f'./record_configs/task_parameters_{yy_mm_dd}_{hh_mm_ss}.json'
    with open(json_file_nm,'w') as f:
        json.dump(task_parameters, f, ensure_ascii=False, indent=4)

    # 기존 AI감시 task 죽이기
    active_tasks, _ = get_task_info()
    active_tasks_values = list(active_tasks.values())

    active_task_num = len(active_tasks_values[0])
    print('# # active_task_num : {}'.format(active_task_num))

    if active_task_num > 0 :
        active_tasks_values_id = active_tasks_values[0][0]['id']
        print(f'Start killing executing task id:{active_tasks_values_id}')
        _ = kill_executing_task(active_tasks_values_id)

    # 새로운 설정값으로 AI감시 task 시작
    first_infer_flag = True
    result2 = BackgroundInfer.delay('# # Start Background Inference # #', first_infer_flag)
    if len(str(result2)) > 0 :
        flash("AI가 CCTV 영상 감시를 시작하였습니다")
        return redirect(url_for('recording'))


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# video stream
@detect_app.route('/video_feed')
@login_required
def video_feed():
    print('{} @ @ 2 video_feed request:{}'.format(datetime.now(), request))
    stream_infer_gen = stream_infer.run()
    return Response(stream_infer_gen, mimetype='multipart/x-mixed-replace; boundary=frame')


@detect_app.route('/video_stream')
@login_required
def video_stream():
    print('{} @ @ 1 video_stream request:{}'.format(datetime.now(), request))
    return render_template('index2.html')


# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# main page
@detect_app.route('/')
def main():
    print('{} @ @ 0 main page request:{}'.format(datetime.now(), request))
    """Main page."""
    return render_template('index1.html')

@detect_app.route('/recording')
@login_required
def recording():
    print('{} @ @ 0 recording page request:{}'.format(datetime.now(), request))
    return render_template('recording.html')


# dashboard
@detect_app.route('/dashboard')
@login_required
def dashboard():
    print('{} @ @ 0 main page request:{}'.format(datetime.now(), request))
    """Main page."""
    return render_template('dashboard.html')

# 로그인 뷰
@detect_app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('로그인 성공', 'success')
            return redirect(url_for('main'))
        else:
            flash('로그인 실패. 아이디 또는 비밀번호를 확인하세요.', 'error')
    return render_template('login.html')


if __name__ == '__main__':
	detect_app.run(host='0.0.0.0', port=7000)

