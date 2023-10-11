FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        alsa-base \
        alsa-utils \
        libsndfile1-dev && \
    apt-get clean

# RUN groupadd -g 600 mw
# RUN useradd wasadmin -u 601 -g 600
# RUN id
# USER wasadmin
# RUN id
# ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]


# docker build -f Dockerfile --no-cache --tag=audio:0.0.1 .
# docker run -it --rm --device /dev/snd -v /home/wasadmin/workspace/test_temp/audio-docker:/audio-docker  audio:0.0.1

# 컨테이너 실행

# 오디오 장치 확인
# arecord -l

# 실행
# aplay /tmp/test-mic.wav 


