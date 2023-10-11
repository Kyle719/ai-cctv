FROM ubuntu:18.04
LABEL maintainer="hsjeong <prezzie77@gmail.com>"
RUN apt-get update \
    && apt-get install -y apache2
RUN echo "TEST WEB" > /var/www/html/index.html
EXPOSE 80
CMD ["/usr/sbin/apache2ctl", "-DFOREGROUND"]

# docker build -t webserver:v1 .
# docker run -d -p 80:80 --name web-apache webserver:v1
# -d -> 백그라운드 데몬 모드로 실행 옵션

