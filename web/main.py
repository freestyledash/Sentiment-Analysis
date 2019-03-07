# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       zhangyanqi
   date：          2019/1/2
-------------------------------------------------
   Change Activity:
                   2019/1/2:
-------------------------------------------------
"""
import json

from flask import request

from analyze_single_comment import init, analyze_after_init

__author__ = 'zhangyanqi'

global encoder
global voc

# app.py
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    comment = str(request.args.get('comment', ''))
    result = analyze_after_init(comment, encoder, voc)
    string = json.dumps(result)
    return string


if __name__ == "__main__":
    print("begin init phase")
    global encoder
    global voc
    encoder, voc = init("../")
    print("finish init phase")
    app.run()
