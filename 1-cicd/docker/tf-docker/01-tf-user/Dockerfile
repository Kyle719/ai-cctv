FROM tensorflow/tensorflow:latest

RUN groupadd -g 600 mw
RUN useradd wasadmin -u 601 -g 600
RUN id
USER wasadmin
RUN id
ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]

# docker build -t tf-user-img:latest .
# tf-user-img:latest 이름으로 도커이미지 만들겠다
# . -> 현제 디렉토리에 있는 기준으로 dockerfile 읽어서 해라

# 아래처럼 docker run 할때 -v
# docker run -it -v /home/wasadmin/workspace/ai-cctv/1-cicd/tf-docker/01-tf-user:/home/wasadmin/workspace/ai-cctv/1-cicd/tf-docker/01-tf-user --name tf-user-cont tf-user-img:latest

# 컨테이너 실행

