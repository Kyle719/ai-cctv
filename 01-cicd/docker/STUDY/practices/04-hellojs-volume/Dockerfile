FROM node:12

RUN groupadd -g 600 mw
RUN useradd wasadmin -u 601 -g 600
RUN id
USER wasadmin
RUN id
COPY hello.js /
CMD ["node", "/hello.js"]

# docker build -t hellojs:latest .
# hellojs:latest 이름으로 도커이미지 만들겠다
# . -> 현제 디렉토리에 있는 기준으로 dockerfile 읽어서 해라

# 아래처럼 docker run 할때-v
# docker run -d --name web -p 8080:8080 hellojs:latest
# docker run -d -v /home/wasadmin/workspace/04-hellojs-volume:/home/wasadmin/workspace/04-hellojs-volume --name web -p 8080:8080 hellojs:latest

# 컨테이너 실행

