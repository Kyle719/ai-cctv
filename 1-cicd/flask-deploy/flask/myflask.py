"""
number = 0
for number in range(100):
    with open('/home/wasadmin/workspace/flask-deploy/test1.txt','a') as f:
        print(number, file=f)
"""

from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return 'hiaaaaaaaaaaaaaaaaaa'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)