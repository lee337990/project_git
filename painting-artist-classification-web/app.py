import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import pymysql

# 현재 파이썬 파일 기준으로 경로를 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
# 업로드 폴더를 static/uploads로 설정 => 업로드한 이미지를 이 경로에 저장
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')


# DB 연결 함수
# - 로컬 DB에 접속하는 함수
# - 아이디/비번/DB명 지정
# - DictCursor는 SQL 결과를 딕셔너리 형태로 반환 => HTML 템플릿에서 쓰기 편리함
def get_db_connection():
    conn = pymysql.connect(
        host='localhost',
        user='tree',
        password='1234',
        db='flask_project2',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

# ==================== 🔥 추가: 여러 모델 준비 ====================
# - 다양한 모델을 쓸 수 있도록 세팅
# - GPU가 있으면 cude, 없으면 cpu로 설정
# - 클래스는 5명 화가 => 출력 노드 수는 5개개
from torchvision.models import resnet18, vgg16, densenet121

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 5


# 모델 로딩 함수
# - 사용자가 선택한 모델 이름에 따라 구조를 정의하고, .pth 가중치 파일을 불러옴
# - .eval()은 추론 모드 설정
# - 학습은 되어있지 않고, 사전학습된 모델 구조 + 내 가중치로 분류만 수행행
def load_model(model_name):
    if model_name == 'vgg16':
        model = vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'densenet121':
        model = densenet121(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'resnet18':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 모델 가중치 불러오기
    model_path = f'models/{model_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# 클래스 이름
# - 모델이 출력한 예측 결과를 사람이 읽을 수 있게 매핑핑
class_names = ["Albrecht_Du_rer", "Edgar_Degas", "Pablo_Picasso", "Pierre_Auguste_Rennoir", "Vincent_van_Gogh"]

# Transform 이미지 전처리
# - 입력 이미지를 모델이 기대하는 형태로 변환:
# - 크기 통일 (224x224)
# - 정규화 (ImageNet 기준)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 예측 함수
# - 이미지를 불러와서 모델에 입력
# - 가장 확률 높은 클래스(softmax 최대값)의 인덱스를 추출
# - pred_class_name으로 이름 반환환
def predict_artist(image_path, selected_model):
    model = load_model(selected_model)
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred_class_idx = torch.max(outputs, 1)

    pred_class_name = class_names[pred_class_idx.item()]
    return pred_class_name

# 메인 페이지 라우트
@app.route('/', methods=['GET', 'POST'])
def index():
    # 첫 페이지: 이미지 업로드 & 모델 선택
    result = None
    filename = None
    pred_artist = None
    if request.method == 'POST':
        file = request.files['file']
        selected_model = request.form.get('model_name')  # 🔥 추가: 모델 선택값 받기
        if file and selected_model:
            # secure_filename()을 통해 경로 탈출 방지(보안) 처리
            # 업로드된 파일을 서버에 저장장
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 예측 결과 추출출
            pred_artist = predict_artist(filepath, selected_model)
            result = f"이 그림은 {pred_artist} 작품입니다."

    # 결과와 이미지 파일명을 템플릿으로 전달달 
    return render_template('index.html', filename=filename, result=result, pred_artist=pred_artist)

# 게시판 공통 처리 함수 (공통화)
# - 화가별 게시판에서 댓글 등록 및 보기 기능을 처리
def handle_board(artist):
    if request.method == 'POST':
        nickname = request.form['nickname']
        content = request.form['content']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = get_db_connection()
        with conn.cursor() as cursor:
            # 게시판 테이블명은 화가 이름 기반(van_Gogh_comments 등)
            # - 댓글 INSERT => 시간도 포함함
            sql = f"INSERT INTO {artist}_comments (nickname, content, timestamp) VALUES (%s, %s, %s)"
            cursor.execute(sql, (nickname, content, timestamp))
            conn.commit()
        conn.close()

        return redirect(url_for('board', artist=artist))

    conn = get_db_connection()
    with conn.cursor() as cursor:
        # 최신순 정렬해서 댓글 가져오기
        sql = f"SELECT * FROM {artist}_comments ORDER BY timestamp DESC"
        cursor.execute(sql)
        posts = cursor.fetchall()
    conn.close()

    return render_template('board.html', artist=artist, posts=posts)

# 게시판 라우트
# - URL에 따라 화가별 게시판 접근 가능
# - /board/Vincent_van_Gogh => 고흐 게시판판
@app.route('/board/<artist>', methods=['GET', 'POST'])
def board(artist):
    return handle_board(artist)

# 앱 실행 코드
# - 업로드 폴더가 없으면 자동 생성
# - Flask 서버 실행
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

