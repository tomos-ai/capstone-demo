import os
import sys

print(' * Import Flask')
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

print(' * Import util')
# Some utilites
import numpy as np
from util import base64_to_pil

print(' * Import model')
from model import *

print(' * Create Flask app')
# Declare a flask app
app = Flask(__name__)
app.config[
    "TEMPLATES_AUTO_RELOAD"] = True  # https://stackoverflow.com/a/54852798/1071459

print(' *** Imports done: ready! ***')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predictwithheatmap/<model_name>', methods=['GET', 'POST'])
def predict_with_heatmap(model_name):
    # print(f'model_name: {model_name}')
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        class_name, probability, img_with_heatmap = model_predict_with_heatmap(
            img, model_name)
        # print(f'class_name: {class_name}, probability: {probability}')
        # print('img_with_heatmap')
        # print(type(img_with_heatmap))
        # print(img_with_heatmap.shape)

        # Serialize the result, you can add additional fields
        return jsonify(class_name=class_name,
                       probability=str(probability),
                       img_with_heatmap=np_to_base64(img_with_heatmap))

    return ''


if __name__ == '__main__':
    # # app.run(port=5002, threaded=False)
    # app.run(debug=True, extra_files=['templates'])

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 80), app)
    http_server.serve_forever()
    # run_server()
