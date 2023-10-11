FROM ubuntu:22.04

RUN groupadd -g 1001 hsjeong
RUN useradd hsjeong -u 1001 -g 1001
RUN id
USER hsjeong
RUN id
ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]


# docker build -t audio-mung-img:latest .
# audio-mung-img:latest 이름으로 도커이미지 만들겠다
# . -> 현제 디렉토리에 있는 기준으로 dockerfile 읽어서 해라

# docker run -d --name web -p 8080:8080 hellojs:latest
# 컨테이너 실행

# docker run -idt -v /home/hsjeong:/home/hsjeong --name audio-mung audio-mung-img:latest
# 컨테이너 실행

