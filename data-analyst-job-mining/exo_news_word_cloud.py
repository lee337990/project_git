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
def get_news_data(urls):
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

                # 본문 크롤링 (뉴스 사이트마다 다를 수 있음)
                paragraphs = soup.find_all('p')  # 일반적으로 본문은 <p> 태그에 포함됨
                body_text = " ".join(p.text.strip() for p in paragraphs)

                # 제목 + 본문 결합
                full_text = title_text + " " + body_text
                news_texts.append(full_text)
        except Exception as e:
            print(f"크롤링 실패: {url}, 오류: {e}")

    return news_texts

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
    urls = ['https://biz.heraldcorp.com/article/3827960',
    'https://www.kgnews.co.kr/news/article.html?no=643541',
    'https://www.venturesquare.net/840927',
    'https://sports.khan.co.kr/article/202111030700003?pt=nv',
    'https://www.newsmp.com/news/articleView.html?idxno=219669',
    'https://www.ajunews.com/view/20211225205130704',
    'https://www.mk.co.kr/news/it/10252819',
    'https://www.edaily.co.kr/news/read?newsId=01151286632294152&mediaCodeNo=257',
    'http://m.pharmstock.co.kr/news/articleView.html?idxno=45364',
    'https://www.medifonews.com/mobile/article.html?no=167727',
    'https://www.chosun.com/special/special_section/2022/12/21/53UZ5DEEHZD7ZD6SFD4KELFOHU/?utm_source=facebook&utm_medium=share&utm_campaign=news&fbclid=IwAR3wWEM1O_xcIbtVLAleT9zUPmemEWGlUP4m-K8s57TYwrSMoJRh_afw-68',
    # 'https://www.startupforest.net/53/?q=YToxOntzOjEyOiJrZXl3b3JkX3R5cGUiO3M6MzoiYWxsIjt9&bmode=view&idx=14153746&t=board',
    'https://www.yakup.com/news/index.html?mode=view&cat=12&nid=278941',
    'https://www.hankyung.com/economy/article/2023022254401',
    'https://www.venturesquare.net/877265',
    'https://news.mt.co.kr/mtview.php?no=2023032916495343891',
    'https://www.docdocdoc.co.kr/news/articleView.html?idxno=3006347',
    'https://news.mt.co.kr/mtview.php?no=2023083016415931583',
    'https://news.mt.co.kr/mtview.php?no=2023101213295416817',
    'https://www.docdocdoc.co.kr/news/articleView.html?idxno=3010866',
    'https://www.etoday.co.kr/news/view/2303004',
    'https://news.tvchosun.com/site/data/html_dir/2024/01/15/2024011590040.html',
    'https://www.chosun.com/english/industry-en/2024/03/01/IHHFHV44YJAT5GMCEG7RQKBJ4E/',
    'https://www.etoday.co.kr/news/view/2378652',
    'https://kormedi.com/1715088/',
    'https://www.etoday.co.kr/news/view/2403182',
    'https://zdnet.co.kr/view/?no=20241105063822'
    ]

    news_texts = get_news_data(urls)  # 뉴스 데이터 크롤링
    stopwords = ['데이터', '엑소', '기사', '뉴스','시스','템즈', '대표']  # 불필요한 단어 제거
    make_wordcloud(news_texts, stopwords, 50)  # 워드클라우드 생성
