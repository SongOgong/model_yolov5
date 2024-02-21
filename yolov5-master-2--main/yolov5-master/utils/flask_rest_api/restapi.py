# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""

import argparse
import io
import os
from PIL import Image

import torch
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection/yolov5"
model = None  # 모델을 담을 변수를 None으로 초기화합니다.

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the YOLOv5 object detection API!"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=640)  # 모델 호출 시 전역 변수 model을 사용합니다.
        return jsonify(results.pandas().xyxy[0].to_dict(orient="records"))

# GET 요청을 처리하는 뷰 함수를 작성합니다.
@app.route(DETECTION_URL, methods=['GET'])
def get_detection():
    return "GET request for object detection endpoint"

# 서버의 상태를 반환하는 라우트를 추가합니다.
@app.route('/status', methods=['GET'])
def get_status():
    # 서버의 상태를 dictionary 형태로 반환합니다.
    server_status = {
        'status': 'running',
        'message': 'The server is up and running.'
    }
    # JSON 형태로 변환하여 반환합니다.
    return jsonify(server_status)

# favicon.ico 라우트를 추가합니다.
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--model', default='yolov5s', help='model name, i.e. --model yolov5s')
    args = parser.parse_args()

    # 모델을 로드하는 부분을 수정합니다. 딕셔너리 대신 단일 모델 객체를 사용합니다.
    model = torch.hub.load("ultralytics/yolov5", 'custom', 'C:/Users/samsung/Downloads/laundry3003-main/laundry3003-main/weights/best.pt', force_reload=True)

    app.run(host="0.0.0.0", port=args.port, debug=True)
