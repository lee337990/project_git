import streamlit as st
import torch
from utils import *
from model.lstm_model import *

# 예측 함수 정의
def predict(text, model, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    input_tensor = encode_and_pad([tokens], vocab)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()

    label_map = {0: "None", 1: "Offensive", 2: "Hate"}
    return label_map[pred]

# Streamlit 앱 정의
def app():
    # 웹 페이지 제목
    st.title("욕설 분류 모델")

    # 사용자가 텍스트 입력
    user_input = st.text_area("분류할 문장을 입력하세요:")

    # 모델과 vocab 로드
    df_train, df_val = load_data()
    vocab = build_vocab(df_train["tokens"].tolist())
    model = LSTMClassifier(vocab_size=len(vocab))

    # 미리 학습된 모델 로드
    model.load_state_dict(torch.load(r"C:\Users\KDP-31-\Desktop\Miniproject11_real\models\model_acc0.5563_epoch5_20250415_111243.pth"))

    # 예측 버튼 클릭 시 결과 출력
    if st.button("분류하기"):
        if user_input:
            result = predict(user_input, model, vocab)
            st.write("욕설 레벨:", result)
        else:
            st.warning("입력된 문장이 없습니다. 문장을 입력해 주세요.")

if __name__ == "__main__":
    app()
