from flask import Flask
import datetime

from celery_task import TrainingTask

training_app = Flask(__name__)

@training_app.route('/')
def hello_world():
    return '--- Hello Web App with Python Flask! ---'

@training_app.route('/celery_task')
def asdf():
    task = TrainingTask()
    result = task.delay()
    return '--- task is queued! celery worker will execute it ---'


if __name__ == '__main__':
	training_app.run(host='0.0.0.0', port=9876)
