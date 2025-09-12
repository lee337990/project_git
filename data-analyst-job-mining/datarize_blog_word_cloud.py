from bs4 import BeautifulSoup
import requests
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import platform
import numpy as np
from PIL import Image


# 뉴스 본문 크롤링 함수
def get_blog_data(urls):
    news_texts = []
    for url in urls:
        try:
            req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            time.sleep(0.2)  # 요청 간격 조절
            if req.ok:
                soup = BeautifulSoup(req.text, 'html.parser')
                
                # 제목 크롤링
                title = soup.find('h1')
                title_text = title.text.strip() if title else ""

                # 본문 크롤링 (사이트마다 다를 수 있음)
                paragraphs = soup.find_all('p')  # 일반적으로 본문은 <p> 태그에 포함됨
                body_text = " ".join(p.text.strip() for p in paragraphs)

                # 제목 + 본문 결합
                full_text = title_text + " " + body_text
                news_texts.append(full_text)
        except Exception as e:
            print(f"크롤링 실패: {url}, 오류: {e}")

    return blog_texts

# 워드 클라우드 생성 함수
def make_wordcloud(text_list, stopwords, word_count):
    okt = Okt()
    word_list = []

    for text in text_list:
        morphs = okt.pos(text)
        for word, tag in morphs:
            # if tag in ['Noun', 'Adjective']:  # 명사, 형용사만 사용
            if tag in ['Noun']:
                word_list.append(word)

    # 단어 빈도 계산
    counts = Counter(word_list)
    for stopword in stopwords:
        if stopword in counts:
            del counts[stopword]
    tags = counts.most_common(word_count)
    print('-' * 80)
    print('tags', tags)


    # 워드 클라우드 생성
    if platform.system() == 'Windows':
        font_path = r'c:\Windows\Fonts\malgun.ttf'
    else:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

    wordcloud = WordCloud(font_path=font_path, width=800, height=600, background_color="white").generate_from_frequencies(counts)
    
    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 실행 코드
if __name__ == '__main__':
    # 고정된 뉴스 링크들
    urls = ['https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88',
    'https://blog.datarize.ai/%EC%98%A8%EC%82%AC%EC%9D%B4%ED%8A%B8%EC%BA%A0%ED%8E%98%EC%9D%B8%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8',
    'https://blog.datarize.ai/%EB%A7%88%EC%BC%80%ED%8C%85%EC%9B%A8%EB%B9%84%EB%82%98',
    'https://blog.datarize.ai/%EA%B0%9C%EB%B0%9C%EC%9E%90%EB%B0%8B%EC%97%85',
    'https://blog.datarize.ai/%EB%A7%88%ED%86%A0%EC%BD%98',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88_1',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88%EC%B1%84%EC%9A%A9',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88%EC%B1%84%EC%9A%A9-0',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88%EC%B1%84%EC%9A%A9-1',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88meetup1',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8',
    'https://blog.datarize.ai/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%9D%BC%EC%9D%B4%EC%A6%88meetup',
    'https://blog.datarize.ai/%EC%9B%A8%EB%B9%84%EB%82%9811',
    'https://blog.datarize.ai/%EC%87%BC%ED%95%91%EB%AA%B0%EB%B0%B0%EB%84%881',
    'https://blog.datarize.ai/20241224',
    'https://blog.datarize.ai/25017'
    ]


    blog_texts = get_blog_data(urls)  # 뉴스 데이터 크롤링
    stopwords = ['데이터', '라이즈', '있어요']  # 불필요한 단어 제거
    make_wordcloud(blog_texts, stopwords, 50)  # 워드클라우드 생성
