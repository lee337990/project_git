# clean_text, tokenize, encode_and_pad, build_vocab

## 모듈 로딩

# 데이터 로딩과 전처리 관련 모듈
import pandas as pd

# 토큰화와 텍스트 전처리 관련 모듈
import re
from konlpy.tag import Okt
import nltk
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence

# LSTM 모델 구축 관련 모듈
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 텍스트 임베딩
from gensim.models import KeyedVectors

from collections import Counter

## 모듈 로딩
import pandas as pd
import re
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence

# Okt 객체 생성
okt = Okt()                     # 한국어 형태소 분석기, 텍스트를 형태소 단위로 분리할 때 사용

# 텍스트 클리닝
def clean_text(text):
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]" , "" , text)   # 한글만 남기기
    text = re.sub(r"\s+", " ", text)                   # 공백 정리
    return text.strip()                                # 앞뒤 공백 제거


# 토큰화 + 품사 정보 + 불용어 제거 + 길이가 짧은 토큰 제거
def tokenize(text):
    stopwords = [
    # 조사
    '은', '는', '이', '가', '을', '를', '에', '의', '도', '으로', '로', '와', '과', '에게', '께', '께서', 
    '에서', '부터', '까지', '마다', '밖에', '한테', '이나', '나', '이며', '라도', '처럼', '만큼', '조차',

    # 접속사/의미 없는 표현
    '그리고', '그러나', '하지만', '또는', '그런데', '그래서', '그러므로', '즉', '게다가', '따라서', '또',

    # 의존명사 및 형식명사
    '것', '거', '분', '수', '등', '데', '중', '자', '때', '의미', '내용', '경우', '정도',

    # 시간 관련 어휘 (정보성 낮은 경우)
    '지금', '오늘', '내일', '어제', '항상', '언제', '매우', '별로', '자주',

    # 기타
    '있다', '없다', '되다', '하다', '이다', '아니다', '하면', '되면', '같다', '그', '이런', '저런', '그런', '저',
    '좀', '더', '많이', '정말', '진짜', '그냥', '거의', '너무', '혹시', '자꾸', '그때', '이제', '또한', '그래도'
]
    tokens = okt.morphs(text, stem=True)
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens
    # return okt.morphs(text)     # 형태소 단위로 토큰화


# 패딩 및 인코딩 
# - 리스트(text) => 인덱스 변환
# - 길이 맞추기
def encode_and_pad(tokenized_texts, vocab, max_len=100):
    encoded = [[vocab.get(token, vocab["<UNK>"]) for token in tokens] for tokens in tokenized_texts]    # 토큰을 인덱스로 변환
    padded = pad_sequence([torch.tensor(seq) for seq in encoded], batch_first=True, padding_value=vocab["<PAD>"])   # 패딩
    return padded[:, :max_len]  # 최대 길이로 자르기

# 데이터 로드
def load_data(train_file=r"C:\Users\KDP-31-\Desktop\Miniproject11_real\data\train.tsv", val_file=r"C:\Users\KDP-31-\Desktop\Miniproject11_real\data\dev.tsv"):
    # 훈련 데이터 로드
    df_train = pd.read_csv(train_file, sep='\t')
    df_train = df_train[df_train["hate"].isin(["hate", "offensive", "none"])]
    label_map = {"none":0, "offensive":1, "hate":2}
    df_train["label"] = df_train["hate"].map(label_map)
    df_train["comments"] = df_train["comments"].apply(clean_text)
    df_train["tokens"] = df_train["comments"].apply(tokenize)

    # 검증 데이터 로드
    df_val = pd.read_csv(val_file, sep='\t')
    df_val = df_val[df_val["hate"].isin(["hate", "offensive", "none"])]  # 'hate', 'offensive', 'none' 필터링
    df_val["label"] = df_val["hate"].map(label_map)
    df_val["comments"] = df_val["comments"].apply(clean_text)
    df_val["tokens"] = df_val["comments"].apply(tokenize)

    return df_train, df_val


def build_vocab(tokenized_texts, min_freq=2):
    counter = Counter(token for sentence in tokenized_texts for token in sentence)
    vocab = { "<PAD>":0, "<UNK>":1 }
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df_train, df_val = load_data()

# 훈련 데이터에서 각 클래스의 개수 확인
train_label_counts = df_train["label"].value_counts().sort_index()
train_label_names = {0: "None", 1: "Offensive", 2: "Hate"}

# 검증 데이터에서 각 클래스의 개수 확인
val_label_counts = df_val["label"].value_counts().sort_index()
val_label_names = {0: "None", 1: "Offensive", 2: "Hate"}

# 훈련 데이터 클래스 분포 시각화
plt.figure(figsize=(10, 5))
plt.bar(train_label_names.values(), train_label_counts)
plt.title("Training Data Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 검증 데이터 클래스 분포 시각화
plt.figure(figsize=(10, 5))
plt.bar(val_label_names.values(), val_label_counts)
plt.title("Validation Data Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 훈련 데이터 클래스 분포 출력
print("훈련 데이터 클래스 분포:")
print(train_label_counts)

# 검증 데이터 클래스 분포 출력
print("검증 데이터 클래스 분포:")
print(val_label_counts)


from collections import Counter
import torch

# train 데이터의 label 컬럼에서 각 클래스별 샘플 수 확인
label_counts = Counter(df_train["label"])
total = sum(label_counts.values())

# GPU가 있을 경우 CUDA를, 없으면 CPU를 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 가중치 계산: 전체 샘플 수 / 각 클래스 샘플 수
weights = [total / label_counts[i] for i in range(3)]
weights = torch.tensor(weights, dtype=torch.float).to(device)

# 이제 클래스 가중치를 텐서로 변환하여 device로 이동
weights = torch.tensor(weights, dtype=torch.float).to(device)

# 클래스별 가중치 출력
print("클래스별 가중치:", weights)

# 손실함수 정의 (가중치 적용)
loss_fn_train = nn.CrossEntropyLoss(weight=weights)