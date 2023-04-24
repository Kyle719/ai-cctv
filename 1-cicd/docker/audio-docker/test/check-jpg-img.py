import os
import time

my_img_dir = '/home/wasadmin/workspace/ai-cctv/99-final/dog.jpg'

num = 0
while True:
    num += 1
    print(num)
    if os.path.isfile(my_img_dir):
        print('@@@@@@@@@@@@@@@@@@@@@@')
        break
    time.sleep(1)
