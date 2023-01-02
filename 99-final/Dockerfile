FROM pytorch/pytorch

WORKDIR /home/wasadmin/workspace/ai-cctv/99-final

# opencv 에러 때문에 추가
# https://somjang.tistory.com/entry/Docker-opencv-%EC%84%A4%EC%B9%98-%EC%8B%9C-ImportError-libGLso1-cannot-open-shared-object-file-No-such-file-or-directory-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
# 아래 테스트 필요
RUN apt-get update
RUN apt-get -y install libglib2.0-0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip uninstall opencv-python
RUN pip install opencv-python

RUN groupadd -g 600 mw
RUN useradd wasadmin -u 601 -g 600
RUN id
USER wasadmin
RUN id

# 아래 테스트 필요
RUN alias ll='ls --color=auto -alF' 
RUN alias c='clear'
RUN alias h='history'

ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]

# docker build -t rtsp-yolov5-webstreaming-img:v01 .
# rtsp-yolov5-webstreaming-img:v01 이름으로 도커이미지 만들겠다
# . -> 현제 디렉토리에 있는 기준으로 dockerfile 읽어서 해라

# 아래처럼 docker run 할때 -v
# docker run -idt -v /home/wasadmin/workspace:/home/wasadmin/workspace -p 7000:7000 --name rtsp-yolov5-webstreaming-ctnr rtsp-yolov5-webstreaming-img:v01


# 컨테이너 실행

