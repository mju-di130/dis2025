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

# 라이브러리 임포트 (SDV, Scipy 등)
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
except ImportError:
    print("[Error] 필수 라이브러리가 없습니다. (!pip install sdv scipy seaborn)")

warnings.filterwarnings("ignore")

# 1. 한글 폰트 설정 (깨짐 방지)
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
# [Step 1] 데이터 전처리 (모든 컬럼 학습 가능하도록 변환)
# -----------------------------------------------------------------------------
def preprocess_data(file_path):
    # 인코딩 자동 감지 로드
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='cp949')
    
    print(f"[System] 원본 데이터 로드: {len(df)}건, 컬럼 수: {len(df.columns)}")

    # 1. 날짜 처리: 연/월/일로 분해
    date_cols = ['개찰일자', '최종낙찰일자'] 
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year.fillna(2000).astype(int)
            df[f'{col}_month'] = df[col].dt.month.fillna(1).astype(int)
            df.drop(columns=[col], inplace=True)

    # 2. 수치형 변환 (콤마 제거)
    def clean_currency(x):
        try:
            return float(str(x).replace(',', ''))
        except:
            return np.nan

    num_cols = ['최종낙찰금액', '최종낙찰율', '기초금액']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)

    # 3. [핵심] 기초금액 역산 (데이터 복원)
    # 원본에 '기초금액' 컬럼이 없으면 생성
    if '기초금액' not in df.columns:
        df['기초금액'] = np.nan

    # 기초금액이 비어있고 낙찰금액/낙찰율이 있으면 역산 수행
    if '최종낙찰금액' in df.columns and '최종낙찰율' in df.columns:
        mask = df['기초금액'].isnull() & (df['최종낙찰율'] > 0)
        df.loc[mask, '기초금액'] = df.loc[mask, '최종낙찰금액'] / (df.loc[mask, '최종낙찰율'] / 100)
        print("[System] 기초금액 역산 완료 (Scatter Plot 준비됨)")
    
    # 4. 텍스트/범주형 데이터 처리
    # (1) 공고기관명 -> 상위 기관으로 범주화
    def category_agency(name):
        name = str(name)
        if '방위사업청' in name or '국방' in name: return '중앙행정기관'
        if any(x in name for x in ['육군', '해군', '공군', '부대']): return '군'
        if any(x in name for x in ['연구소', '연구원', '과학']): return '연구기관'
        return '기타'
    
    if '공고기관명' in df.columns:
        df['기관_상위'] = df['공고기관명'].apply(category_agency)
        df.drop(columns=['공고기관명'], inplace=True)

    # (2) 주소 -> 시/도 단위로 범주화
    if '최종낙찰업체주소' in df.columns:
        df['업체_지역'] = df['최종낙찰업체주소'].astype(str).str.split(' ').str[0] 
        df.drop(columns=['최종낙찰업체주소'], inplace=True)

    # (3) 식별자성 텍스트 삭제
    pii_cols = ['최종낙찰업체대표자명', '최종낙찰업체담당자명', '입찰공고명', '최종낙찰업체명']
    for col in pii_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True) 

    # 5. 수치형 데이터 결측치 처리
    df.fillna(0, inplace=True)

    # 학습 대상 컬럼 확인
    print(f"[System] 전처리 완료. 학습 컬럼: {df.columns.tolist()}")
    return df

# -----------------------------------------------------------------------------
# [Step 2] 재현 데이터 생성 (DP-CTGAN Simulation)
# -----------------------------------------------------------------------------
def generate_synthetic(df_train):
    print("[System] 재현 데이터 생성 시작 (CTGAN)...")
    
    # 메타데이터 자동 감지
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)

    # 모델 학습 (속도를 위해 Epochs=200 설정, 실제 논문용은 500 권장)
    model = CTGANSynthesizer(metadata, epochs=500, verbose=True)
    model.fit(df_train)

    # 데이터 생성
    synth = model.sample(len(df_train))
    
    # 수치형 컬럼 음수 방지
    num_cols = synth.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if '금액' in col or '율' in col:
            synth[col] = synth[col].clip(lower=0)

    return synth

# -----------------------------------------------------------------------------
# [Step 3] 시각화 및 검증 (Visualization) - [수정] 강제 출력
# -----------------------------------------------------------------------------
def evaluate_results(real_df, synth_df):
    set_korean_font()
    print("[System] 실험 결과 시각화 생성 중...")

    # JSD 계산 함수
    def get_jsd(p, q):
        p_hist, _ = np.histogram(p, bins=100, density=True)
        q_hist, _ = np.histogram(q, bins=100, density=True)
        return jensenshannon(p_hist+1e-10, q_hist+1e-10)

    if '최종낙찰금액' in real_df.columns:
        jsd_val = get_jsd(real_df['최종낙찰금액'], synth_df['최종낙찰금액'])
        print(f" >> 최종낙찰금액 JSD: {jsd_val:.4f} (목표: < 0.1)")

        # 1. 롱테일 분포 비교 (KDE Plot)
        plt.figure(figsize=(10, 6))
        q99 = real_df['최종낙찰금액'].quantile(0.99)
        sns.kdeplot(real_df[real_df['최종낙찰금액'] < q99]['최종낙찰금액'], fill=True, label='원본', color='blue')
        sns.kdeplot(synth_df[synth_df['최종낙찰금액'] < q99]['최종낙찰금액'], fill=True, label='재현', color='red', linestyle='--')
        plt.title(f"도표 1. 최종낙찰금액 분포 유사성 (JSD={jsd_val:.3f})")
        plt.xlabel("금액 (원)"); plt.legend()
        plt.savefig("Exp_Result_1_Distribution.png")
    
    # 2. 산점도 (Scatter Plot) - [수정] 조건 완화 및 강제 출력
    # 전처리된 real_df에는 '기초금액'이 있으므로 조건 충족됨
    if '기초금액' in real_df.columns and '최종낙찰금액' in real_df.columns:
        print("[System] 산점도(Scatter Plot) 생성 중...")
        plt.figure(figsize=(12, 5))
        
        # 데이터가 너무 많으면 1000개만 샘플링 (속도 및 가독성)
        n_sample = min(1000, len(real_df))
        real_sample = real_df.sample(n=n_sample)
        synth_sample = synth_df.sample(n=n_sample)
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=real_sample['기초금액'], y=real_sample['최종낙찰금액'], alpha=0.5, color='blue')
        plt.title("원본: 기초금액 vs 낙찰금액")
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=synth_sample['기초금액'], y=synth_sample['최종낙찰금액'], alpha=0.5, color='red')
        plt.title("재현: 기초금액 vs 낙찰금액")
        
        plt.tight_layout()
        plt.savefig("Exp_Result_2_Scatter.png")
    else:
        print("[Warning] '기초금액' 컬럼이 없어 산점도를 그릴 수 없습니다.")

    # 3. 히트맵 (Correlation)
    plt.figure(figsize=(10, 6))
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    # 날짜 등 불필요한 수치형 제외하고 핵심 변수만 선택
    core_cols = [c for c in num_cols if '금액' in c or '율' in c]
    
    if len(core_cols) > 1:
        plt.subplot(1, 2, 1)
        sns.heatmap(real_df[core_cols].corr(), annot=True, fmt=".2f", cmap='Blues')
        plt.title("원본 상관관계")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(synth_df[core_cols].corr(), annot=True, fmt=".2f", cmap='Reds')
        plt.title("재현 상관관계")
        plt.tight_layout()
        plt.savefig("Exp_Result_3_Heatmap.png")

    # 4. 범주형 빈도 (Bar Chart) - 기관_상위
    if '기관_상위' in real_df.columns:
        real_cnt = real_df['기관_상위'].value_counts(normalize=True).reset_index()
        real_cnt['Type'] = 'Original'
        synth_cnt = synth_df['기관_상위'].value_counts(normalize=True).reset_index()
        synth_cnt['Type'] = 'Synthetic'
        
        combined = pd.concat([real_cnt, synth_cnt])
        combined.columns = ['Category', 'Frequency', 'Type']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=combined, x='Category', y='Frequency', hue='Type', palette=['blue', 'red'])
        plt.title("도표 4. 기관 분류별 빈도 유사성")
        plt.savefig("Exp_Result_4_Category.png")

    print("[System] 모든 결과 이미지가 저장되었습니다.")

# -----------------------------------------------------------------------------
# [Main Execution]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    file_name = "data.csv"
    
    # 1. 전처리 (여기서 '기초금액'이 생성됨)
    processed_df = preprocess_data(file_name)
    
    # 2. 재현 데이터 생성 (전처리된 데이터로 학습)
    synthetic_data = generate_synthetic(processed_df)
    
    # 3. 생성된 재현 데이터 CSV 파일로 저장
    output_csv_name = "synthetic_data.csv"
    synthetic_data.to_csv(output_csv_name, index=False, encoding='utf-8-sig')
    print(f"[System] 재현 데이터가 '{output_csv_name}' 파일로 저장되었습니다.")
    
    # 4. 평가 및 시각화 (전처리된 원본 데이터와 비교)
    evaluate_results(processed_df, synthetic_data)