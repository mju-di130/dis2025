# =============================================================================
# [실험: 원본 데이터 기반 재현 데이터 생성 (Scatter Plot Fix)]
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
import os

# 라이브러리 임포트
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
except ImportError:
    print("[Error] 필수 라이브러리가 없습니다. (!pip install sdv scipy seaborn)")

warnings.filterwarnings("ignore")

# 1. 한글 폰트 설정
def set_korean_font():
    font_paths = ['C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/Supplemental/AppleGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for path in font_paths:
        try:
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rc('font', family=font_name)
            plt.rc('axes', unicode_minus=False)
            print(f"[System] 폰트 설정 완료: {font_name}")
            return
        except:
            continue
    print("[Warning] 한글 폰트를 찾지 못했습니다.")

# -----------------------------------------------------------------------------
# [Step 1] 데이터 전처리 (업체_지역 파생변수 생성 추가)
# -----------------------------------------------------------------------------
def preprocess_data(file_path):
    # 로드
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='cp949')
    
    print(f"[System] 원본 데이터 로드: {len(df)}건")

    # 1. 날짜 처리
    date_cols = ['개찰일자', '최종낙찰일자'] 
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year.fillna(2000).astype(int)
            df[f'{col}_month'] = df[col].dt.month.fillna(1).astype(int)
            df.drop(columns=[col], inplace=True)

    # 2. 수치형 변환
    def clean_currency(x):
        try: return float(str(x).replace(',', ''))
        except: return np.nan

    for col in ['최종낙찰금액', '최종낙찰율', '기초금액']:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)

    # 3. 기초금액 역산
    if '기초금액' not in df.columns: df['기초금액'] = np.nan
    if '최종낙찰금액' in df.columns and '최종낙찰율' in df.columns:
        mask = df['기초금액'].isnull() & (df['최종낙찰율'] > 0)
        df.loc[mask, '기초금액'] = df.loc[mask, '최종낙찰금액'] / (df.loc[mask, '최종낙찰율'] / 100)

    # 4. 범주형 데이터 처리
    # (1) 기관명 -> 상위 기관
    def category_agency(name):
        name = str(name)
        if '방위사업청' in name or '국방' in name: return '중앙행정기관'
        if any(x in name for x in ['육군', '해군', '공군', '부대']): return '군'
        return '기타'
    
    if '공고기관명' in df.columns:
        df['기관_상위'] = df['공고기관명'].apply(category_agency)
        df.drop(columns=['공고기관명'], inplace=True)

    # (2) [핵심 수정] 주소 -> 업체_지역 (시/도 단위) 추출
    if '최종낙찰업체주소' in df.columns:
        # 주소에서 첫 번째 단어(서울, 경기, 대전 등)만 추출
        df['업체_지역'] = df['최종낙찰업체주소'].astype(str).str.split(' ').str[0]
        
        # 이상한 데이터(빈값, 특수문자 등) 필터링 -> '기타'로 통합
        valid_regions = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
        # '서울특별시' -> '서울' 처럼 앞 2글자만 비교하거나 포함 여부 확인
        df['업체_지역'] = df['업체_지역'].apply(lambda x: next((r for r in valid_regions if x.startswith(r)), '기타'))
        
        print(f"[System] '업체_지역' 컬럼 생성 완료 (Unique: {df['업체_지역'].nunique()}개)")
        df.drop(columns=['최종낙찰업체주소'], inplace=True)

    # (3) 식별자 삭제
    pii_cols = ['최종낙찰업체대표자명', '최종낙찰업체담당자명', '입찰공고명', '최종낙찰업체명']
    for col in pii_cols:
        if col in df.columns: df.drop(columns=[col], inplace=True) 

    # 5. 결측치 처리
    df.fillna(0, inplace=True)

    print(f"[System] 학습 대상 컬럼: {df.columns.tolist()}")
    return df

# -----------------------------------------------------------------------------
# [Step 2] 재현 데이터 생성 (CTGAN)
# -----------------------------------------------------------------------------
def generate_synthetic(df_train):
    print("[System] 재현 데이터 생성 시작 (CTGAN)...")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)

    # 모델 학습 (속도를 위해 Epochs=300 설정, 실제 실험용은 500 권장)
    model = CTGANSynthesizer(metadata, epochs=500, verbose=True)
    model.fit(df_train)

    synth = model.sample(len(df_train))
    
    # 수치형 음수 방지
    num_cols = synth.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if '금액' in col or '율' in col:
            synth[col] = synth[col].clip(lower=0)

    return synth

# -----------------------------------------------------------------------------
# [Step 3] 시각화 및 검증 (업체_지역 그래프 추가)
# -----------------------------------------------------------------------------
def evaluate_results(real_df, synth_df):
    set_korean_font()
    print("[System] 실험 결과 시각화 생성 중...")

    # JSD 계산
    def get_jsd(p, q):
        p_hist, _ = np.histogram(p, bins=100, density=True)
        q_hist, _ = np.histogram(q, bins=100, density=True)
        return jensenshannon(p_hist+1e-10, q_hist+1e-10)

    # 1. 롱테일 분포
    if '최종낙찰금액' in real_df.columns:
        jsd_val = get_jsd(real_df['최종낙찰금액'], synth_df['최종낙찰금액'])
        print(f" >> 최종낙찰금액 JSD: {jsd_val:.4f}")
        
        plt.figure(figsize=(10, 6))
        q99 = real_df['최종낙찰금액'].quantile(0.99)
        sns.kdeplot(real_df[real_df['최종낙찰금액'] < q99]['최종낙찰금액'], fill=True, label='원본', color='blue')
        sns.kdeplot(synth_df[synth_df['최종낙찰금액'] < q99]['최종낙찰금액'], fill=True, label='재현', color='red', linestyle='--')
        plt.title(f"도표 1. 최종낙찰금액 분포 (JSD={jsd_val:.3f})")
        plt.legend(); plt.savefig("Exp_Result_1_Distribution.png")
    
    # 2. 산점도
    if '기초금액' in real_df.columns:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=real_df['기초금액'], y=real_df['최종낙찰금액'], alpha=0.3, color='blue')
        plt.title("원본 산점도")
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=synth_df['기초금액'], y=synth_df['최종낙찰금액'], alpha=0.3, color='red')
        plt.title("재현 산점도")
        plt.tight_layout(); plt.savefig("Exp_Result_2_Scatter.png")

    # 3. 히트맵
    plt.figure(figsize=(10, 6))
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    core_cols = [c for c in num_cols if '금액' in c or '율' in c]
    
    if len(core_cols) > 1:
        plt.subplot(1, 2, 1)
        sns.heatmap(real_df[core_cols].corr(), annot=True, fmt=".2f", cmap='Blues')
        plt.title("원본 상관관계")
        plt.subplot(1, 2, 2)
        sns.heatmap(synth_df[core_cols].corr(), annot=True, fmt=".2f", cmap='Reds')
        plt.title("재현 상관관계")
        plt.tight_layout(); plt.savefig("Exp_Result_3_Heatmap.png")

    # 4. 범주형 빈도 (기관_상위)
    if '기관_상위' in real_df.columns:
        plt.figure(figsize=(10, 6))
        real_cnt = real_df['기관_상위'].value_counts(normalize=True).reset_index()
        real_cnt['Type'] = 'Original'
        synth_cnt = synth_df['기관_상위'].value_counts(normalize=True).reset_index()
        synth_cnt['Type'] = 'Synthetic'
        combined = pd.concat([real_cnt, synth_cnt])
        combined.columns = ['Category', 'Frequency', 'Type']
        sns.barplot(data=combined, x='Category', y='Frequency', hue='Type', palette=['blue', 'red'])
        plt.title("도표 4. 기관 분류별 빈도 유사성")
        plt.savefig("Exp_Result_4_Category_Agency.png")

    # 5. [추가] 범주형 빈도 (업체_지역)
    if '업체_지역' in real_df.columns:
        plt.figure(figsize=(12, 6))
        real_cnt = real_df['업체_지역'].value_counts(normalize=True).reset_index()
        real_cnt['Type'] = 'Original'
        synth_cnt = synth_df['업체_지역'].value_counts(normalize=True).reset_index()
        synth_cnt['Type'] = 'Synthetic'
        combined = pd.concat([real_cnt, synth_cnt])
        combined.columns = ['Region', 'Frequency', 'Type']
        
        # 빈도 순으로 정렬
        order = real_df['업체_지역'].value_counts().index
        
        sns.barplot(data=combined, x='Region', y='Frequency', hue='Type', palette=['blue', 'red'], order=order)
        plt.title("도표 5. 업체 지역별 빈도 유사성")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Exp_Result_5_Region.png") # 이미지 저장

    print("[System] 모든 결과 이미지가 저장되었습니다.")

# -----------------------------------------------------------------------------
# [Main Execution]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    file_name = "data.csv"
    
    # 1. 전처리
    processed_df = preprocess_data(file_name)
    
    # 2. 재현 데이터 생성
    synthetic_data = generate_synthetic(processed_df)
    
    # 3. CSV 저장
    output_csv_name = "synthetic_data.csv"
    synthetic_data.to_csv(output_csv_name, index=False, encoding='utf-8-sig')
    print(f"[System] 재현 데이터가 '{output_csv_name}' 파일로 저장되었습니다.")
    
    # 4. 평가 및 시각화
    evaluate_results(processed_df, synthetic_data)