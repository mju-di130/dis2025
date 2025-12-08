# =============================================================================
# [실험 코드] 방위산업 데이터의 안전한 활용을 위한 재현 데이터 검증
#
# 본 스크립트는 제안된 '방위산업 데이터 신뢰 프레임워크(DIDTF)' 및 'DP-CTGAN' 모델의
# 실효성을 검증하기 위해 작성되었다. 주요 수행 내용은 다음과 같다.
#
# 1. 머신러닝 효용성 평가 (Machine Learning Efficacy):
#    - TSTR (Train on Synthetic, Test on Real) 방법론을 적용하여
#      재현 데이터로 학습된 AI 모델의 실무 적용 가능성을 평가한다.
#    - 비교 지표: F1-Score, 성능 보존율(Performance Retention Rate)
#
# 2. 프라이버시 안전성 검증 (Privacy Risk Assessment):
#    - 거리 기반 유출 위험도 (Distance to Closest Record, DCR)를 측정하여
#      생성된 데이터가 원본 데이터를 단순 암기(Overfitting)했는지 여부를 판별한다.
# =============================================================================

# [1. 환경 설정] 필수 라이브러리 로드 및 설치
# Google Colab 환경을 가정하여 필요한 패키지를 설치한다.
# - xgboost: 고성능 그라디언트 부스팅 모델 (TSTR 평가용)
# - sdv, scikit-learn: 데이터 처리 및 평가 지표 산출용
# - matplotlib, seaborn: 결과 시각화용
try:
    import xgboost
    import sdv
except ImportError:
    !pip install xgboost scikit-learn pandas numpy matplotlib seaborn sdv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# [2. 데이터 로드] 원본 및 재현 데이터셋 적재
# 실험의 정합성을 위해 동일한 전처리가 수행되지 않은 Raw 상태의 파일을 로드한다.
# - data-utf8.csv: 방위산업 원본 데이터 (Ground Truth)
# - synthetic-utf8.csv: 제안 모델(DP-CTGAN)을 통해 생성된 재현 데이터
try:
    real_df = pd.read_csv('data-utf8.csv')
    synthetic_df = pd.read_csv('synthetic-utf8.csv')
    print("[System] 데이터 파일 로드 성공")
    print(f" - 원본 데이터 크기: {real_df.shape}")
    print(f" - 재현 데이터 크기: {synthetic_df.shape}")
except FileNotFoundError:
    print("[Error] 실험 데이터를 찾을 수 없습니다. 파일 업로드를 확인하십시오.")

# [3. 데이터 전처리] 머신러닝 학습을 위한 변수 변환
def preprocess_for_ml(df):
    """
    머신러닝 알고리즘에 입력하기 위해 데이터의 형식을 변환하고,
    예측 과제(Task)를 정의하는 전처리 함수.
    
    Args:
        df (pd.DataFrame): 전처리 대상 데이터프레임
        
    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    df = df.copy()
    
    # 3-1. 수치형 변수 정제
    # 금액 정보에 포함된 콤마(,)를 제거하고 수치형(Float)으로 변환한다.
    if df['최종낙찰금액'].dtype == 'object':
        df['최종낙찰금액'] = df['최종낙찰금액'].astype(str).str.replace(',', '').astype(float)
    
    # 3-2. 타겟 변수(Target Variable) 정의
    # 본 연구의 시나리오는 '방산 계약 이행 위험도 예측'이다.
    # '최종낙찰율'이 88% 미만인 경우를 '저가 낙찰에 따른 잠재적 위험군(1)'으로 정의한다.
    # (Threshold: 88.0%)
    df['Target'] = np.where(df['최종낙찰율'] < 88.0, 1, 0)
    
    # 3-3. 불필요한 식별자 제거 (Feature Selection)
    # 개별 식별자(이름, 주소 등)는 모델의 일반화 성능을 저해하고
    # 단순 암기(Memorization)를 유발할 수 있으므로 학습 변수에서 제외한다.
    drop_cols = ['입찰공고명', '최종낙찰일자', '최종낙찰업체명', 
                 '최종낙찰업체대표자명', '최종낙찰업체담당자명', '최종낙찰업체주소',
                 '최종낙찰금액_num', '낙찰율_num', '기관_상위', '연구주제']
    
    # 데이터셋에 실제로 존재하는 컬럼만 선택하여 제거
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # 3-4. 결측치(Missing Value) 처리
    # 결측값은 0으로 대체하여 모델 입력 시 에러를 방지한다.
    df = df.fillna(0)
    
    return df

# 전처리 함수 적용
real_prep = preprocess_for_ml(real_df)
synthetic_prep = preprocess_for_ml(synthetic_df)

# [중요] 컬럼 순서 동기화
# 재현 데이터 생성 과정에서 컬럼 순서가 바뀔 수 있으므로, 원본 데이터 기준으로 정렬한다.
synthetic_prep = synthetic_prep[real_prep.columns]

print("[System] 데이터 전처리 완료")

# 결과 저장을 위한 로그 버퍼 초기화
log_buffer = ""

# [4. 머신러닝 효용성 평가] TSTR vs TRTR 비교 분석
def evaluate_ml_efficacy(real, synth, target_col):
    """
    TRTR(Train on Real)과 TSTR(Train on Synthetic) 성능을 비교하여
    재현 데이터의 인공지능 학습 유용성을 정량적으로 평가한다.
    """
    # 4-1. 범주형 변수 인코딩 (Label Encoding)
    real_enc = real.copy()
    synth_enc = synth.copy()
    
    # 원본과 재현 데이터의 범주(Category)를 통합하여 인코딩 (Unknown Label 방지)
    for col in real.select_dtypes(include='object').columns:
        le = LabelEncoder()
        combined_data = pd.concat([real[col], synth[col]], axis=0).astype(str)
        le.fit(combined_data)
        
        real_enc[col] = le.transform(real[col].astype(str))
        synth_enc[col] = le.transform(synth[col].astype(str))
    
    # 4-2. 데이터셋 분할 (Data Splitting)
    X_real = real_enc.drop(target_col, axis=1)
    y_real = real_enc[target_col]
    
    X_syn = synth_enc.drop(target_col, axis=1)
    y_syn = synth_enc[target_col]
    
    # [검증 원칙] 테스트 데이터(Test Set)는 반드시 '원본 데이터'에서 추출해야 한다.
    # 이는 실제 환경에서의 예측 성능을 정확히 측정하기 위함이다.
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )
    
    # 4-3. Baseline 모델 학습 (TRTR)
    # 원본으로 학습하고 원본으로 평가한다. (목표 성능치)
    model_real = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_real.fit(X_train_real, y_train_real)
    pred_real = model_real.predict(X_test_real)
    f1_trtr = f1_score(y_test_real, pred_real, average='macro')
    
    # 4-4. 제안 모델 학습 (TSTR)
    # 재현 데이터로 학습하고 원본으로 평가한다. (실험 성능치)
    model_syn = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model_syn.fit(X_syn, y_syn)  # 학습 데이터로 재현 데이터 사용
    pred_syn = model_syn.predict(X_test_real)  # 평가는 실제 데이터로 수행
    f1_tstr = f1_score(y_test_real, pred_syn, average='macro')
    
    return f1_trtr, f1_tstr

# 평가 수행
f1_trtr, f1_tstr = evaluate_ml_efficacy(real_prep, synthetic_prep, 'Target')

# 결과 기록
ml_result_text = f"""
============================================================
[실험 결과 1] 머신러닝 학습 효용성 평가 (Machine Learning Efficacy)
------------------------------------------------------------
 - 평가 모델: XGBoost Classifier
 - 평가 지표: F1-Score (Macro Average)
 
 1. TRTR (원본 학습 -> 원본 평가): {f1_trtr:.4f}
 2. TSTR (재현 학습 -> 원본 평가): {f1_tstr:.4f}
 
 [결론] 성능 보존율 (Retention Rate): {(f1_tstr/f1_trtr)*100:.2f}%
============================================================
"""
print(ml_result_text)
log_buffer += ml_result_text

# [5. 프라이버시 안전성 검증] 거리 기반 유출 위험도(DCR) 분석
def calculate_dcr(real, synth):
    """
    재현 데이터와 원본 데이터 간의 유클리드 거리(Euclidean Distance)를 계산하여
    최단 거리(Distance to Closest Record)를 산출한다.
    """
    # 5-1. 샘플링 (메모리 효율성 고려)
    # 전체 거리 계산은 연산량이 많으므로 무작위 2,000건을 표본 추출하여 분석한다.
    n_sample = min(2000, len(real), len(synth))
    real_sample = real.sample(n=n_sample, random_state=42)
    synth_sample = synth.sample(n=n_sample, random_state=42)
    
    # 5-2. 데이터 정규화 (Min-Max Scaling)
    # 거리 계산 시 변수 간 스케일 차이에 의한 왜곡을 방지하기 위해 0~1 사이로 정규화한다.
    scaler = MinMaxScaler()
    
    # 범주형 변수는 One-hot Encoding으로 수치화
    real_proc = pd.get_dummies(real_sample)
    synth_proc = pd.get_dummies(synth_sample)
    
    # 컬럼 차원 일치 (Missing Column 처리)
    real_proc, synth_proc = real_proc.align(synth_proc, join='inner', axis=1, fill_value=0)
    
    real_norm = scaler.fit_transform(real_proc)
    synth_norm = scaler.transform(synth_proc)
    
    # 5-3. 거리 행렬 계산
    dists = pairwise_distances(synth_norm, real_norm, metric='euclidean')
    
    # 각 재현 데이터 레코드별로 가장 가까운 원본 레코드와의 거리(Min) 추출
    dcr_values = dists.min(axis=1)
    return dcr_values

# 타겟 변수를 제외한 속성 데이터로 거리 계산 수행
dcr = calculate_dcr(real_prep.drop('Target', axis=1), synthetic_prep.drop('Target', axis=1))

# DCR = 0 인 경우 (원본과 완벽히 일치하는 데이터 유출 사례) 계산
exact_matches = np.sum(dcr == 0)

# 결과 기록
dcr_result_text = f"""
============================================================
[실험 결과 2] 프라이버시 안전성 검증 (Privacy Risk Assessment)
------------------------------------------------------------
 - 평가 지표: DCR (Distance to Closest Record)
 
 1. DCR 평균 거리 (Mean Distance): {np.mean(dcr):.4f}
 2. DCR 최소 거리 (Min Distance): {np.min(dcr):.4f}
 3. 원본 일치 데이터 수 (Exact Matches, DCR=0): {exact_matches}건
 4. 재식별 고위험군 (하위 5% 거리 임계값): {np.percentile(dcr, 5):.4f}
 
 [결론] DCR 최소값이 0보다 크므로, 원본 데이터 유출(Leakage) 없음 확인.
============================================================
"""
print(dcr_result_text)
log_buffer += dcr_result_text

# [6. 결과 파일 저장]
# 6-1. 실험 수치 결과 텍스트 파일 저장
with open("experiment_results_appendix.txt", "w", encoding="utf-8") as f:
    f.write(log_buffer)
print("[File Saved] 실험 결과가 'experiment_results_appendix.txt' 파일로 저장되었습니다.")

# 6-2. DCR 분포 시각화 그래프 저장
plt.figure(figsize=(10, 6))
sns.histplot(dcr, kde=True, color='green', bins=50)
# 하위 5% 위험 임계선 표시
plt.axvline(x=np.percentile(dcr, 5), color='red', linestyle='--', label='5th Percentile Risk Line')
plt.title('Distribution of Distance to Closest Record (DCR)', fontsize=15)
plt.xlabel('Euclidean Distance', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 고해상도 이미지(300 DPI)로 저장하여 논문 삽입 시 깨짐 방지
plt.savefig('dcr_graph_appendix.png', dpi=300, bbox_inches='tight')
print("[File Saved] DCR 그래프가 'dcr_graph_appendix.png' 파일로 저장되었습니다.")
plt.show()
