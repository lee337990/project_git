'''
[ 체감 계절 기간 - 6, 7, 8월 여름 ]
1. 계절 기간이 나눠짐(애초 평균기온으로 나눠짐)
2. 평균온도 기준: new계절별 기간 정함
	- 기존계절기간 데이터 뽑기
	- 새로 정한 계절기간 뽑기
3. 기존계절기간과 새로 정한 계절기간과 비교
	- 계절별 기후 특성(ex: 강수량을 통해 확인 )
3. 기존계절과 기후특성과 새기준과 기후 특성 비교
파이랑 막대그래프로 표현
'''


## ------------------------------------------------------------------------------------------------------
## [1] 모듈 로딩 및 데이터 준비
## ------------------------------------------------------------------------------------------------------
## [1-1] 모듈 로딩
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import numpy as np




## ----------------------------------------------------------------------------
## 그래서 여름은?
## ----------------------------------------------------------------------------
file = '../mini_project/전국_평균_기온_94_24.xlsx'
# file = r"C:\Users\KDT-13\Desktop\KDT 7\project\TP3_public\전국_평균_기온_94_24.xlsx"
df = pd.read_excel(file)

df.columns = ['날짜', '지점', '평균기온', '최저기온', '최고기온']
df = df.drop(labels=['지점'], axis=1)
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
df['년'] = pd.to_datetime(df['날짜']).dt.year # 연 컬럼 뽑기
df['월'] = pd.to_datetime(df['날짜']).dt.month # 월 컬럼 만들기


# 6월 이후 데이터만 남김
# df = df[df['날짜'].dt.month > 6]

# 연도와 월 추가
# df['년'] = df['날짜'].dt.year
# df['월'] = df['날짜'].dt.month

# 온도 조건 (5~20도) 충족 여부 (1 = 충족, 0 = 미충족)
df['온도_충족'] = (df['평균기온'] >= 20 ).astype(int)

# 각 연도-월별 온도 충족 일수 계산
monthly_counts = df.groupby(['년', '월'])['온도_충족'].sum().reset_index()

# 임계값 설정 (예: 15일 이상이면 가을로 인정)
threshold = 15
monthly_counts['여름_여부'] = monthly_counts['온도_충족'] >= threshold

# 원본 데이터에 '여름 여부' 병합
df = df.merge(monthly_counts[['년', '월', '여름_여부']], on=['년', '월'], how='left')

# 여름로 판정된 데이터 필터링
df_new = df[df['여름_여부']]




## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------

df_new = df_new.copy()  # 🔴 : 슬라이스 문제 방지

# 1-2    여름(6~8월)에 해당하는 데이터 필터링 = df_old
df_old=df[df['날짜'].dt.month.isin([6, 7, 8])] 
df_old = df_old.copy()  # 복사본 생성


df_new_result = df_new.groupby('년')['월'].value_counts().to_frame() # 년에 해당하는 월의 일들을 카운트
df_new_result = df_new_result.sort_index() # 년도,월별 오름차순 정렬 



### df_old 전처리
df_old['년'] = pd.to_datetime(df_new['날짜']).dt.year # datetime에 년을 뽑아냄  
df_old['월'] = pd.to_datetime(df_new['날짜']).dt.month # datetime에 월을 뽑아냄 


df_old_result = df_old.groupby('년')['월'].value_counts().to_frame() # 년에 해당하는 월의 일들을 카운트
df_old_result = df_old_result.sort_index() # 년도, 월별 오름차순 정렬

df_new_result = df_new[df_new['온도_충족'] == 1].groupby('년')['월'].value_counts().to_frame()
df_new_result=df_new_result.sort_index()

## 연도별 월별 카운트 합
df_old_Cyear = df_old_result.groupby(level=0)['count'].sum().to_frame()
df_new_Cyear = df_new_result.groupby(level=0)['count'].sum().to_frame()

# 그래프 그리기
plt.figure(figsize=(8,5))
plt.plot(df_old_Cyear, marker='o', label='df_old_Cyear')
plt.plot(df_new_Cyear, marker='s', label='df_new_Cyear')
plt.ylabel("일수")
plt.xlabel("연도")
plt.xticks(range(1994, 2025, 3))
plt.legend()
plt.show()

## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------

# 차이 계산
df_diff = df_new_Cyear - df_old_Cyear

# 그래프 설정
x = df_diff.index # 연도
y = df_diff.values.flatten() # 차이 값

# 선형 회귀를 이용한 추세선 계산
coefficients = np.polyfit(x, y, 1)  # 1차 방정식 (직선) 추세선
trend_line = np.poly1d(coefficients)  # 방정식 생성

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', label='차이')
plt.plot(x, trend_line(x), linestyle='--', color='orange', label='추세선')  # 추세선 추가
# plt.axhline(0, color='red', linestyle='--', label='기준선')
plt.xlabel('연도')
plt.ylabel('차이')
plt.title('기존 여름(df_old)과 새로운 여름(df_new) 기간의 차이')
plt.legend()
plt.grid(True)
plt.show()


## ----------------------------------------------------------------------------
## ----------------------------------------------------------------------------
## --------------------------------------------------------------------------------------------------------------
## [1] 데이터 로딩
## --------------------------------------------------------------------------------------------------------------
file_h = '../mini_project/전국폭염일수_94_24.csv'
df_h = pd.read_csv(file_h, encoding='cp949')

## --------------------------------------------------------------------------------------------------------------
## [2] 데이터 전처리
## --------------------------------------------------------------------------------------------------------------

# [2-1] 컬럼명 정리
df_h.columns = ['날짜', '폭염일수', '평균최고기온']

# [2-2] 날짜 변환
df_h['날짜'] = pd.to_datetime(df_h['날짜'], format='%b-%y')
df_h['연월'] = df_h['날짜'].dt.strftime('%Y-%m')
df_h['월'] = df_h['날짜'].dt.month  # 필터링용


df_strang_h = df_h[df_h['월'].isin([1,2,3,4,5, 6,7,8,9, 10,11,12])]
# 폭염 발생이 있는 데이터만 필터링
df_nonzero = df_strang_h[df_strang_h['폭염일수'] > 0]

## --------------------------------------------------------------------------------------------------------------
## [3] 시각화 (X축 정밀하게 수정)
## --------------------------------------------------------------------------------------------------------------
## [3] 시각화 (막대그래프 + X축 눈금 20개만 출력 + 추세선 추가)

plt.figure(figsize=(25, 6))

# X축을 숫자로 변환 (연월을 인덱스로 사용)
x_values = np.arange(len(df_nonzero))
y_values = df_nonzero['폭염일수']

# 🔹 막대그래프 (폭염 일수)
plt.bar(x_values, y_values, color='red', alpha=0.7, label='폭염일수')

# 🔹 X축 눈금 20개만 출력 (균등 간격으로 선택)
num_ticks = 20
tick_indices = np.linspace(0, len(df_nonzero) - 1, num_ticks, dtype=int)  # 균등 간격 선택
tick_labels = df_nonzero['연월'].iloc[tick_indices]  # 해당 인덱스의 연월만 선택

plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90)

# 🔹 🔥 추세선 추가 (선형 회귀)
z = np.polyfit(x_values, y_values, 1)  # 1차 회귀 (선형)
p = np.poly1d(z)  # 회귀선 함수로 변환
plt.plot(x_values, p(x_values), color='blue', linestyle='--', linewidth=2, label='추세선')

plt.xlabel('날짜')
plt.ylabel('폭염일수')
plt.title('폭염 발생일수 변화 (1994~2024) + 추세선')
plt.legend()
plt.grid(True)

# 그래프 출력
plt.show()


## [3] 시각화 (막대그래프 + X축 눈금 20개만 출력)

# plt.figure(figsize=(25, 6))

# # X축을 숫자로 변환 (연월을 인덱스로 사용)
# x_values = np.arange(len(df_nonzero))
# y_values = df_nonzero['폭염일수']

# # 🔹 막대그래프 (폭염 일수)
# plt.bar(x_values, y_values, color='red')

# # 🔹 X축 눈금 20개만 출력 (균등 간격으로 선택)
# num_ticks = 20
# tick_indices = np.linspace(0, len(df_nonzero) - 1, num_ticks, dtype=int)  # 균등 간격 선택
# tick_labels = df_nonzero['연월'].iloc[tick_indices]  # 해당 인덱스의 연월만 선택

# plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90)

# plt.xlabel('날짜')
# plt.ylabel('폭염일수')
# plt.title('폭염 발생일수 변화 (1994~2024)')
# plt.grid(True)

# # 그래프 출력
# plt.show()
# plt.figure(figsize=(25, 6))
# plt.bar(df_nonzero['연월'], df_nonzero['폭염일수'], color='red')

# # 📌 **X축 눈금 & 라벨 재설정**
# tick_labels = df_nonzero['연월'].tolist()  # 폭염 발생한 연월 리스트
# tick_indices = range(len(df_nonzero))  # 연속적인 인덱스 생성

# # X축 눈금 설정 (폭염 발생한 연월만 표시)
# plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=90)

# plt.xlabel('날짜')
# plt.ylabel('폭염일수')
# plt.title('폭염 발생일수 변화 (1994~2024)')
# plt.grid(True)

# # 그래프 출력
# plt.show()

## --------------------------------------------------------------------------
## --------------------------------------------------------------------------
## --------------------------------------------------------------------
## 여름 제외 폭염일수만
## --------------------------------------------------------------------
## --------------------------------------------------------------------------------------------------------------
## [1] 데이터 로딩
## --------------------------------------------------------------------------------------------------------------
file_h = '../mini_project/전국폭염일수_94_24.csv'
df_h = pd.read_csv(file_h, encoding='cp949')

## --------------------------------------------------------------------------------------------------------------
## [2] 데이터 전처리
## --------------------------------------------------------------------------------------------------------------

# [2-1] 컬럼명 정리
df_h.columns = ['날짜', '폭염일수', '평균최고기온']

# [2-2] 날짜 변환
df_h['날짜'] = pd.to_datetime(df_h['날짜'], format='%b-%y')
df_h['연월'] = df_h['날짜'].dt.strftime('%Y-%m')
df_h['월'] = df_h['날짜'].dt.month  # 필터링용

# 6~8월(여름) 제외
# df_strang_h = df_h[~df_h['월'].isin([6, 7, 8])]
df_strang_h = df_h[df_h['월'].isin([1,2,3,4,5, 9, 10,11,12])]
# 폭염 발생이 있는 데이터만 필터링
df_nonzero = df_strang_h[df_strang_h['폭염일수'] > 0]

## --------------------------------------------------------------------------------------------------------------
## [3] 시각화 (X축 정밀하게 수정)
## --------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.bar(df_nonzero['연월'], df_nonzero['폭염일수'], color='red')
# 📌 **추세선 추가**
x = np.arange(len(df_nonzero))  # X축 인덱스 (연월을 숫자로 변환)
y = df_nonzero['폭염일수'].values  # Y축 (폭염일수)

# 선형 회귀 (1차 다항식)
z = np.polyfit(x, y, 1)  # 기울기와 절편 구하기
p = np.poly1d(z)  # 다항식 생성
plt.plot(df_nonzero['연월'], p(x), color='blue', linestyle='--', linewidth=2, label="추세선")


# 📌 **X축 눈금 & 라벨 재설정**
tick_labels = df_nonzero['연월'].tolist()  # 폭염 발생한 연월 리스트
tick_indices = range(len(df_nonzero))  # 연속적인 인덱스 생성

# X축 눈금 설정 (폭염 발생한 연월만 표시)
plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)

plt.xlabel('날짜')
plt.ylabel('폭염일수')
plt.title('여름(6~8월) 제외한 폭염 발생일수 변화 (1994~2024)')
plt.grid(True)

# 그래프 출력
plt.show()


## --------------------------------------------------------------------------
## --------------------------------------------------------------------------
## --------------------------------------------------------------------
## 여름 폭염일수도 늘어나고 있는지 
## --------------------------------------------------------------------

## --------------------------------------------------------------------------------------------------------------
## [1] 데이터 로딩
## --------------------------------------------------------------------------------------------------------------
file_h = '../mini_project/전국폭염일수_94_24.csv'
df_h = pd.read_csv(file_h, encoding='cp949')

## --------------------------------------------------------------------------------------------------------------
## [2] 데이터 전처리
## --------------------------------------------------------------------------------------------------------------

# [2-1] 컬럼명 정리
df_h.columns = ['날짜', '폭염일수', '평균최고기온']

# [2-2] 날짜 변환
df_h['날짜'] = pd.to_datetime(df_h['날짜'], format='%b-%y')
df_h['연월'] = df_h['날짜'].dt.strftime('%Y-%m')
df_h['월'] = df_h['날짜'].dt.month  # 필터링용

# 6~8월(여름) 제외
# df_strang_h = df_h[~df_h['월'].isin([6, 7, 8])]
df_strang_h = df_h[df_h['월'].isin([6,7,8])]
# 폭염 발생이 있는 데이터만 필터링
df_nonzero = df_strang_h[df_strang_h['폭염일수'] > 0]

## --------------------------------------------------------------------------------------------------------------
## [3] 시각화 (X축 정밀하게 수정)
## --------------------------------------------------------------------------------------------------------------
## [3] 시각화 (꺾은선 그래프 + 추세선 추가)

# plt.figure(figsize=(12, 6))

# # X축을 숫자로 변환 (연월을 인덱스로 사용)
# x_values = np.arange(len(df_nonzero))
# y_values = df_nonzero['폭염일수']

# # 🔹 꺾은선 그래프
# plt.plot(x_values, y_values, marker='o', linestyle='-', color='red', label='폭염 일수')

# # 🔹 추세선 추가 (선형 회귀)
# z = np.polyfit(x_values, y_values, 1)  # 1차 다항식(선형 회귀)
# p = np.poly1d(z)  
# plt.plot(x_values, p(x_values), linestyle="dashed", color="blue", label='추세선')

# # X축 눈금 설정 (연월 표시)
# plt.xticks(ticks=x_values, labels=df_nonzero['연월'], rotation=45)

# plt.xlabel('날짜')
# plt.ylabel('폭염일수')
# plt.title('여름(6~8월) 폭염 일수 (1994~2024)')
# plt.legend()  # 범례 추가
# plt.grid(True)

# # 그래프 출력
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.bar(df_nonzero['연월'], df_nonzero['폭염일수'], color='red')

# # 📌 **X축 눈금 & 라벨 재설정**
# tick_labels = df_nonzero['연월'].tolist()  # 폭염 발생한 연월 리스트
# tick_indices = range(len(df_nonzero))  # 연속적인 인덱스 생성

# # X축 눈금 설정 (폭염 발생한 연월만 표시)
# plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)

# plt.xlabel('날짜')
# plt.ylabel('폭염일수')
# plt.title('여름(6~8월) 폭염 일수 (1994~2024)')
# plt.grid(True)

# # 그래프 출력
# plt.show()
## [3] 시각화 (막대그래프 + 추세선)

plt.figure(figsize=(12, 6))

# X축을 숫자로 변환 (연월을 인덱스로 사용)
x_values = np.arange(len(df_nonzero))
y_values = df_nonzero['폭염일수']

# 🔹 막대그래프 (폭염 일수)
plt.bar(x_values, y_values, color='red', label='폭염 일수')

# 🔹 추세선 추가 (선형 회귀)
z = np.polyfit(x_values, y_values, 1)  # 1차 다항식(선형 회귀)
p = np.poly1d(z)
plt.plot(x_values, p(x_values), linestyle="dashed", color="blue", label='추세선')

# 🔹 X축 눈금 13개만 출력 (균등 간격으로 선택)
num_ticks = 13
tick_indices = np.linspace(0, len(df_nonzero) - 1, num_ticks, dtype=int)  # 균등 간격 선택
tick_labels = df_nonzero['연월'].iloc[tick_indices]  # 해당 인덱스의 연월만 선택

plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=45)

plt.xlabel('날짜')
plt.ylabel('폭염일수')
plt.title('여름(6~8월) 폭염 일수 (1994~2024)')
plt.legend()  # 범례 추가
plt.grid(True)

# 그래프 출력
plt.show()

## ----------------------------------------------------------------------
## ----------------------------------------------------------------------
## ---------------------------------------
## 연도별로 폭염일수 합치기
## ---------------------------------------
## --------------------------------------------------------------------------------------------------------------
## [1] 데이터 로딩
## --------------------------------------------------------------------------------------------------------------
file_h = '../mini_project/전국폭염_94_24.csv'
df_h = pd.read_csv(file_h, encoding='cp949')


## --------------------------------------------------------------------------------------------------------------
## [3] 시각화 (X축 정밀하게 수정)
## --------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(25, 6))
plt.bar(df_h['연도'], df_h['연합계'], color='red')


# 📌 **추세선 추가**
z = np.polyfit(df_h['연도'], df_h['연합계'], 1)  # 1차 다항식(직선) 피팅
p = np.poly1d(z)  # 다항식 생성
plt.plot(df_h['연도'], p(df_h['연도']), color='blue', linestyle='--', linewidth=2, label='추세선')

# 📌 **X축 눈금 & 라벨 재설정**
# tick_labels = df_nonzero['연월'].tolist()  # 폭염 발생한 연월 리스트
# tick_indices = range(len(df_nonzero))  # 연속적인 인덱스 생성

# # X축 눈금 설정 (폭염 발생한 연월만 표시)
# plt.xticks(ticks=df_h['연도'], labels=df_h['연도'], rotation=90)

# 📌 **X축 눈금 & 라벨 재설정**
plt.xticks(ticks=df_h['연도'], labels=df_h['연도'], rotation=90)


plt.xlabel('연도')
plt.ylabel('폭염일수 연합계')
plt.title('폭염일수 연합계 변화 (1994~2024) 및 추세선')
plt.legend()
plt.grid(True)

# 그래프 출력
plt.show()
