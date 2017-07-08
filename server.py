import requests
import json
import datetime
import time

from TheFinalBuild import get_sentiment
from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS, cross_origin

def factory():
    app = Flask(__name__, static_url_path="/static")
    return app

app = factory()
cors = CORS(app)


@cross_origin()
@app.route('/message', methods=['POST'])
def reply():
    hashtag = request.json['tag']
    print (hashtag)
    json_reply= get_sentiment(hashtag)
    print (json_reply)
    return jsonify(json_reply)

    # return jsonify( { "reply": [{"text": "hi"},{"text": "Hey there"}] } )

def time_stamp():
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    t_stamp=time.mktime( datetime.datetime.strptime(date, "%Y-%m-%d").timetuple())
    t_stamp = int(t_stamp)
    t_stamp *= 1000

    return str(t_stamp)


@cross_origin()
@app.route("/")
def index():
    return render_template("index.html")

# start app
if __name__ == "__main__":

    app.run(host='0.0.0.0', port = 8080)
