
from flask import Flask
app_with_gunicorn = Flask(__name__)

@app_with_gunicorn.route("/")
def home():
    return 'hi with gunicornnnnnnnnnnnnnnnnnnnnn'

if __name__ == "__main__":
    app_with_gunicorn.run(host="0.0.0.0", port=5000, debug=True)