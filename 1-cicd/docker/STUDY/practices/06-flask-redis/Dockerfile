FROM python:3.7-alpine

WORKDIR /code
# 이렇게 작업디렉토리 생성하면 COPY 하는것들 모두 여기를 디폴트 경로로 복사됨. RUN 할때도 디폴트 경로에서 하는것 같음.

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /code

CMD ["flask", "run"]

# 궁금
# docker-compose.yml 보면 web 도커 이미지 빌드할때 여기 dockerfile 내용처럼 requirements.txt 에서 파이썬 라이브러리 flask 랑 redis 둘다 설치를 하는데 docker-compose.yml 에서 redis 도커 이미지를 가지고 컨테이너를 또 띄우는 이유는?
# app.py 보면 redis = Redis(host='redis', port=6379) 코드가 있는데 redis 컨테이너를 쓰는거 같긴함. 근데 꼭 따로 컨테이너를 써야하나? DAP Vision 에서는 따로 있었나?

