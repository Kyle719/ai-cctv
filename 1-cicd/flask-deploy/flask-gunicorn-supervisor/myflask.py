"""
number = 0
for number in range(100):
    with open('/home/wasadmin/workspace/flask-deploy/test1.txt','a') as f:
        print(number, file=f)
"""

from flask import Flask
app_with_gunicorn_supervisor = Flask(__name__)

@app_with_gunicorn_supervisor.route("/")
def home():
    return 'flask! gunicorn! supervisor!'

if __name__ == "__main__":
    app_with_gunicorn_supervisor.run(host="0.0.0.0", port=5000, debug=True)