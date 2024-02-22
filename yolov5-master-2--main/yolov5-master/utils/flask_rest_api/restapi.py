import torch
from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
import boto3
import uuid

# YOLOv5 모델 로드 (경로는 EC2에 클론한 리포지토리의 위치에 맞게 설정)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/cloned/yolov5/repo/best.pt', force_reload=True)

app = Flask(__name__)

@app.route("/v1/object-detection/yolov5", methods=["POST"])
def predict():
    data = request.json
    if 'image_url' not in data:
        return jsonify({'error': 'No image URL provided'}), 400

    # 이미지 URL로부터 이미지 로드
    image_url = data['image_url']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # 이미지 분석
    results = model(img, size=640)

    # 분석 결과 이미지를 S3에 업로드
    s3_client = boto3.client('s3')
    bucket_name = 'laundrycoachbucket75411-staging'  # S3 버킷 이름 설정
    file_name = f"analyzed_images/{uuid.uuid4()}.jpg"  # 고유한 파일 이름 생성
    img_byte_arr = BytesIO()

    # YOLOv5 분석 결과 이미지를 바이트 배열로 변환
    results_img = Image.fromarray(results.render()[0] if results.render() else results.imgs[0])
    results_img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    s3_client.upload_fileobj(img_byte_arr, bucket_name, file_name, ExtraArgs={'ContentType': 'image/jpeg'})
    analyzed_image_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"

    # 분석 결과(클래스 이름)와 업로드된 이미지 URL 반환
    response_data = {
        'detections': results.pandas().xyxy[0].to_dict(orient='records'),
        'analyzed_image_url': analyzed_image_url
    }
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
