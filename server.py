from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def init():
    return render_template('index.html')

@app.route('/shark2')
def shark2():
    return 'ok'


if __name__ == "__main__":
    app.run()
