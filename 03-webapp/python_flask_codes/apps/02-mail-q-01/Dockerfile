FROM python

RUN pip install flask

RUN groupadd -g 601 mw
RUN useradd wasadmin -u 600 -g 601
RUN id
USER wasadmin
RUN id
WORKDIR /home/wasadmin/workspace/flaskbook
ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]


# . -> 현제 디렉토리에 있는 기준으로 dockerfile 읽어서 해라
# docker build -t minimalapp-img:latest .

# 컨테이너 실행
# docker run -idt -p 5001:5001 -v /home/wasadmin/workspace/flaskbook:/home/wasadmin/workspace/flaskbook --name minimalapp-test minimalapp-img:latest

