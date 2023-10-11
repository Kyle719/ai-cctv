import logging
import os

from flask import (
    Flask,
    current_app,
    flash,
    g,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

# Flask 클래스를 인스턴스화한다
app = Flask(__name__)

# ■01■
# URL과 실행하는 함수를 매핑한다
@app.route("/")
def index():
    return "Hello, Flaskbook!"


# ■02■
@app.route("/hello/<name>", methods=["GET"], endpoint="hello-endpoint")
def hello(name):
    return f"Hello, {name}"
# endpoint 지정해서 이 라우팅의 이름을 지정.
# url 에 변수를 쓸 수 있다.
# methods=["GET", "POST"] 이런식으로 이 라우팅에서 허가할 HTTP 메소드를 지정할수있다
# Flask2부터는 @app.get("/hello"), @app.post("/hello")라고 기술하는 것이 가능
# @app.get("/hello")
# @app.post("/hello")
# def hello():
#   return "Hello, World!"


# ■03■
# show_name 엔드포인트를 작성한다
@app.route("/name/<name>")
def show_name(name):
    # 변수를 템플릿 엔진에게 건넨다
    return render_template("index.html", name=name)
# render_template -> 템플릿엔진. 플라스크의 템플릿엔진은 Jinja2 인데
# 얘의 render_template 함수를 이용하여 HTML을 렌더링함.


# ■04■
with app.test_request_context():
    print(url_for("index"))
    # /
    print(url_for("hello-endpoint", name="world"))
    # /hello/world
    print(url_for("show_name", name="AK", page="1"))
    # /name/AK?page=1
# url_for 함수에 라우팅 함수명을 넣거나 endpoint 이름을 넣어서 url 정보를 가져온다


# ■05■
# 지금까지 app = Flask(__name__)로 취득한 app 인스턴스에 접근했음.
# 근데 앱 규모 커지면 상호 참조해서 순환참조 문제가 발생할 수 있어서
# 직접 app 을 참조하지 않고 current_app 에 접근함.

# 여기에서 호출하면 오류가 된다
# print(current_app)

# 애플리케이션 컨텍스트를 취득하여 스택에 push한다
ctx = app.app_context()
ctx.push()

# current_app에 접근이 가능해진다
print(current_app.name)
# >> apps.minimalapp.app

# 전역 임시 영역에 값을 설정한다
g.connection = "connection"
print(g.connection)
# >> connection

with app.test_request_context("/users?updated=true"):
    # true가 출력된다
    print(request.args.get("updated"))



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
