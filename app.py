import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# 데이터 로드
data = {
    "Team": list(range(1, 67)),
    "Strategy and Organization": [31.3, 42.2, 60.9, 55.6, 62.5, 69.9, 63.5, 72.3, 66.3, 68.1, 70.6, 75.2, 75.4, 64.4, 76.0, 74.6, 71.4, 79.9, 77.1, 66.9, 77.1, 78.1, 78.4, 69.5, 76.6, 72.9, 79.9, 78.4, 77.9, 78.6, 78.7, 76.0, 80.2, 77.6, 80.0, 80.6, 71.9, 81.7, 81.3, 83.3, 79.7, 79.7, 78.1, 81.9, 82.8, 65.6, 84.0, 77.5, 79.0, 87.1, 82.6, 84.1, 86.2, 86.7, 83.7, 91.8, 85.9, 87.5, 83.3, 88.5, 90.6, 89.9, 90.6, 93.0, 89.8, 100.0],
    "Work Method": [35.4, 66.1, 57.3, 64.6, 78.6, 72.9, 75.0, 71.1, 75.8, 75.0, 76.0, 71.8, 73.8, 76.3, 74.3, 75.6, 75.9, 75.0, 76.3, 77.1, 80.6, 82.6, 75.8, 83.3, 80.2, 77.8, 78.3, 81.9, 79.4, 80.6, 79.4, 79.5, 81.1, 83.3, 82.0, 80.3, 86.1, 81.8, 83.3, 83.3, 84.4, 84.6, 85.4, 82.8, 90.1, 95.8, 83.4, 90.8, 85.4, 85.6, 89.3, 86.2, 86.4, 86.8, 92.6, 90.8, 89.1, 91.7, 95.8, 87.8, 93.8, 92.4, 93.1, 96.9, 97.4, 100.0],
    "Leadership": [34.4, 67.2, 70.3, 68.1, 68.8, 67.5, 75.0, 68.2, 83.1, 75.0, 77.5, 72.3, 71.2, 78.8, 81.3, 82.1, 84.4, 74.6, 76.0, 84.4, 78.8, 83.3, 81.4, 80.5, 79.7, 86.5, 82.3, 79.4, 86.5, 84.4, 85.1, 81.3, 83.1, 83.9, 84.0, 87.5, 89.6, 87.0, 83.3, 82.3, 86.7, 89.1, 87.5, 90.7, 92.2, 94.8, 86.5, 91.3, 93.8, 85.8, 91.5, 91.2, 90.1, 92.3, 92.4, 89.3, 96.1, 94.4, 100.0, 97.9, 92.7, 99.3, 91.7, 96.1, 99.2, 100.0],
    "Competency Development": [50.0, 61.9, 56.3, 63.0, 55.9, 66.3, 63.3, 68.7, 62.5, 69.5, 64.8, 70.2, 71.5, 74.5, 67.5, 66.8, 68.9, 72.9, 73.4, 75.5, 70.6, 63.3, 72.2, 76.3, 74.6, 75.0, 72.5, 73.0, 73.2, 75.0, 75.4, 82.5, 75.2, 75.8, 76.4, 75.3, 76.7, 75.6, 78.3, 78.3, 78.1, 76.6, 80.0, 76.0, 67.5, 76.7, 79.7, 76.5, 78.9, 81.4, 77.5, 81.8, 82.9, 81.8, 80.6, 78.0, 81.3, 80.5, 75.8, 82.1, 86.7, 85.8, 92.5, 88.8, 90.0, 97.5]
}

# DataFrame 생성
df = pd.DataFrame(data)

# 데이터 타입 변환
df['Team'] = df['Team'].astype(int)
for col in ['Strategy and Organization', 'Work Method', 'Leadership', 'Competency Development']:
    df[col] = df[col].astype(float)

# 분석할 컬럼 리스트 정의
columns = ['Strategy and Organization', 'Work Method', 'Leadership', 'Competency Development']

# 한글-영문 매핑 딕셔너리
column_mapping = {
    'Strategy and Organization': '전략 및 조직',
    'Work Method': '일하는 방식',
    'Leadership': '리더십',
    'Competency Development': '역량 개발'
}

# Streamlit 앱 시작
st.title('팀 평가 데이터 분석')

# 각 항목별로 표준 정규분포를 그리기
for column in columns:
    try:
        # 평균과 표준편차 계산
        mean = df[column].mean()
        std_dev = df[column].std()

        # 하위 10% 기준점 계산
        bottom_10_percent = stats.norm.ppf(0.1)
        bottom_10_percent_score = mean + bottom_10_percent * std_dev

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 히스토그램 그리기
        ax.hist(df[column], bins=20, density=True, alpha=0.6, color='b')

        # 정규분포 곡선 그리기
        x = np.linspace(df[column].min(), df[column].max(), 100)
        p = stats.norm.pdf(x, mean, std_dev)
        ax.plot(x, p, 'k', linewidth=2)

        # 하위 10% 영역 표시
        ax.fill_between(x, 0, p, where=(x < bottom_10_percent_score), color='red', alpha=0.3)

        # 하위 10% 기준선 표시
        ax.axvline(x=bottom_10_percent_score, color='red', linestyle='--')

        # 그래프 세부 설정
        ax.set_title(f"Score Distribution of {column}")
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')

        # x축 눈금 설정
        start = int(df[column].min() // 10) * 10
        end = int(df[column].max() // 10 + 1) * 10
        ax.set_xticks(range(start, end + 1, 10))

        # 범례 추가
        ax.text(0.95, 0.95, 'Bottom 10%', transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

        # Streamlit에 그래프 표시 (제목은 한글로)
        st.subheader(f"{column_mapping[column]} 점수 분포")
        st.pyplot(fig)

        # 하위 10% 팀 수 계산
        bottom_10_percent_teams = df[df[column] < bottom_10_percent_score]
        st.write(f"하위 10% 팀 수: {len(bottom_10_percent_teams)}")
        st.write(f"하위 10% 기준 점수: {bottom_10_percent_score:.2f}")

    except Exception as e:
        st.error(f"Error processing {column}: {str(e)}")

st.success("분석이 완료되었습니다.")
