FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        alsa-base \
        alsa-utils \
        libsndfile1-dev && \
    apt-get clean

RUN groupadd -g 600 mw
RUN useradd wasadmin -u 601 -g 600
RUN id
USER wasadmin
RUN id
ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]


# docker build --no-cache --tag=audio:0.0.1 .
# docker run -it -v /home/wasadmin/workspace:/home/wasadmin/workspace --rm --device /dev/snd --name audio-cont audio:0.0.1 

# 컨테이너 실행

