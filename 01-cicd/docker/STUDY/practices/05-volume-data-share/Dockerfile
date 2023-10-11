FROM ubuntu:18.04

ADD df.sh /bin/df.sh
RUN chmod +x /bin/df.sh
ENTRYPOINT ["/bin/df.sh"]

# docker build -t df:latest .
# docker run -d -v /webdata:/webdata --name df df:latest

# -> 계속 disk 사용량 정보를 /webdata/index.html 에다가 업데이트하는 df컨테이너

# docker run -d --name web -v /webdata:/usr/share/nginx/html:ro -p 80:80 nginx:1.14

# -> df컨테이너가 써놓은 html 로 웹띄워주는 nginx 서버컨테이너


# --> 컨테이너끼리 호스트의 같은 데이터 공유
