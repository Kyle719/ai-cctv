from celery import Celery, Task

class TrainingTask(Task):
    name = 'Traninig Task'
    concurrency = 1

    def __init__(self):
        self.my_five = 5

    def run(self):
        self.__init__()

        for num in range(10):
            res_num = num * self.my_five
            print(res_num)

BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
training_task_app = Celery('my_celery_task', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)
training_task_app.conf.broker_transport_options = {'visibility_timeout':259200}
training_task_app.register_task(TrainingTask())








