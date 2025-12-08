# =============================================================================
# [최종 실험: 하이브리드 재현 및 성능 검증 (All-in-One)]
# 1. 생성: 핵심 변수(AI) + 기초금액(수식) -> 논리적 완결성 확보
# 2. 검증: 원본 vs 재현 데이터 TSTR 평가 및 시각화 즉시 출력
#
# 수정사항: 
# 1. 컬럼명 공백 제거 (strip) -> KeyError 방지
# 2. 날짜 컬럼('최종낙찰일자') 존재 여부 확인 후 처리
#
# 수정사항: 
# 1. [Critical Fix] 학습 데이터(Label) 내 NaN/Infinity 전처리 로직 강화
# 2. XGBoost 'Label contains NaN' 오류 원천 차단
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
import os

# 머신러닝 및 평가 라이브러리
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
except ImportError:
    print("[Error] SDV 라이브러리 미설치. (!pip install sdv)")

try:
    from diffprivlib.mechanisms import Laplace
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, f1_score
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# [설정]
INPUT_CSV = "data.csv"
OUTPUT_SYNTHETIC = "synthetic_data_hybrid.csv"
EPSILON_DP = 0.5

# 한글 폰트 설정
def set_korean_font():
    try:
        font_paths = ['C:/Windows/Fonts/malgun.ttf', '/System/Library/Fonts/Supplemental/AppleGothic.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
        for path in font_paths:
            if os.path.exists(path):
                font_name = fm.FontProperties(fname=path).get_name()
                plt.rc('font', family=font_name)
                plt.rc('axes', unicode_minus=False)
                return
    except:
        print("[Warning] 폰트 설정 실패")

# -----------------------------------------------------------------------------
# [1단계] 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
def load_and_preprocess():
    # 1. 데이터 로드
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"파일이 없습니다: {INPUT_CSV}")
        
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
    except:
        df = pd.read_csv(INPUT_CSV, encoding='cp949')
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    print(f"[System] 데이터 로드: {len(df)}건")

    # 2. 날짜 데이터 처리
    if '최종낙찰일자' in df.columns:
        df['최종낙찰일자'] = pd.to_datetime(df['최종낙찰일자'], errors='coerce')
        df['최종낙찰일자_year'] = df['최종낙찰일자'].dt.year
        df['최종낙찰일자_month'] = df['최종낙찰일자'].dt.month

    # 3. 수치형 변환
    def clean(x):
        try: return float(str(x).replace(',', ''))
        except: return np.nan
    
    for c in ['최종낙찰금액', '최종낙찰율', '기초금액']:
        if c in df.columns: df[c] = df[c].apply(clean)

    # 4. 기초금액 역산 (결측치 채우기)
    if '기초금액' not in df.columns or df['기초금액'].isnull().mean() > 0.5:
        if '최종낙찰금액' in df.columns and '최종낙찰율' in df.columns:
            # 0으로 나누기 방지
            rate = df['최종낙찰율'].replace(0, np.nan)
            df['기초금액'] = df['최종낙찰금액'] / (rate / 100)
    
    # 5. 기관명 범주화
    if '공고기관명' in df.columns:
        def cat_agency(x):
            s = str(x)
            if '방위사업청' in s or '국방' in s: return '중앙행정기관'
            if any(k in s for k in ['육군', '해군', '공군']): return '군'
            return '기타'
        df['기관_상위'] = df['공고기관명'].apply(cat_agency)
    else:
        df['기관_상위'] = '기타'

    # 6. 로그 변환 (학습용)
    if '최종낙찰금액' in df.columns:
        df['최종낙찰금액_log'] = np.log1p(df['최종낙찰금액'].clip(lower=0))
    
    # ★ [핵심 수정] 결측치(NaN) 및 무한대(Inf) 제거 (학습 오류 방지)
    # 주요 변수에 NaN이 있으면 머신러닝 학습 시 'Label contains NaN' 에러 발생
    important_cols = ['최종낙찰금액', '최종낙찰율', '기초금액', '기관_상위']
    important_cols = [c for c in important_cols if c in df.columns]
    
    # Inf -> NaN 변환 후 Drop
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=important_cols)
    
    print(f"[Preprocessing] 결측치 제거 후 데이터: {len(df)}건")
    return df

# -----------------------------------------------------------------------------
# [2단계] 하이브리드 생성 (AI + Rule)
# -----------------------------------------------------------------------------
def generate_hybrid(df):
    print("\n[Generation] 하이브리드 재현 데이터 생성 시작...")
    
    # 학습 컬럼 선정
    candidates = ['최종낙찰금액_log', '최종낙찰율', '기관_상위', '최종낙찰일자_year', '최종낙찰일자_month']
    train_cols = [c for c in candidates if c in df.columns]
    
    df_train = df[train_cols].dropna()
    
    # 모델 학습 (속도를 위해 Epochs=300 설정, 실제 실험용은 500 권장)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    
    model = CTGANSynthesizer(metadata, epochs=500, verbose=True)
    model.fit(df_train)
    
    # 샘플링
    synth = model.sample(len(df))
    
    # [복원 1] 로그 역변환
    if '최종낙찰금액_log' in synth.columns:
        synth['최종낙찰금액'] = np.expm1(synth['최종낙찰금액_log']).clip(lower=0)
    
    # [복원 2] 기초금액 수식 계산
    if '최종낙찰금액' in synth.columns and '최종낙찰율' in synth.columns:
        print("[Post-processing] 수식 기반 '기초금액' 복원 중...")
        # 0으로 나누기 방지
        rate = synth['최종낙찰율'].replace(0, 0.1) 
        synth['기초금액'] = synth['최종낙찰금액'] / (rate / 100)
        
        # 무한대 및 NaN 처리 (매우 중요)
        synth['기초금액'] = synth['기초금액'].replace([np.inf, -np.inf], 0).fillna(0).clip(lower=0)
    
    return synth

# -----------------------------------------------------------------------------
# [3단계] 성능 평가 (TSTR)
# -----------------------------------------------------------------------------
def evaluate_tstr(real, synth):
    print("\n" + "="*60)
    print(" [Evaluation] TSTR 성능 평가")
    print("="*60)
    
    targets = ['기초금액', '최종낙찰금액', '최종낙찰율', '기관_상위']
    results = []
    
    for target in targets:
        if target not in real.columns or target not in synth.columns: continue
        
        print(f">> 타겟 분석 중: {target}")
        
        # ★ [핵심 수정] 타겟 변수 결측치 재확인 및 제거 (2중 안전장치)
        # NaN이 하나라도 있으면 XGBoost가 멈춤
        real_clean = real.dropna(subset=[target])
        synth_clean = synth.dropna(subset=[target])
        
        if len(real_clean) == 0 or len(synth_clean) == 0:
            print(f"   [Skip] {target} 데이터가 비어있어 평가를 건너뜁니다.")
            continue

        # 데이터 준비
        drop_cols = [target] + [c for c in real.columns if 'log' in c or '공고' in c or '일자' in c]
        
        X_real = real_clean.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
        y_real = real_clean[target]
        X_synth = synth_clean.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
        y_synth = synth_clean[target]
        
        if X_real.shape[1] == 0: continue

        # Target 인코딩 (분류)
        is_classification = False
        if y_real.dtype == 'object' or y_real.nunique() < 20:
            is_classification = True
            le = LabelEncoder()
            # 학습 데이터(Synth) 기준으로 인코딩
            y_synth = le.fit_transform(y_synth.astype(str))
            
            # 테스트 데이터(Real)는 학습 데이터에 있는 클래스만 남김
            mask = y_real.astype(str).isin(le.classes_)
            if mask.sum() == 0: continue # 매칭되는 클래스 없음
            
            X_real = X_real[mask]
            y_real = le.transform(y_real[mask].astype(str))
            
            model = XGBClassifier(n_jobs=-1, random_state=42)
            metric = "F1-Score"
        else:
            # 회귀 (무한대 값 체크)
            if not np.isfinite(y_real).all() or not np.isfinite(y_synth).all():
                print("   [Skip] Target 변수에 무한대(Inf) 값이 포함되어 있습니다.")
                continue
                
            model = XGBRegressor(n_jobs=-1, random_state=42)
            metric = "R2 Score"
            
        # 데이터 분할
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
            
            # 1. TRTR
            model.fit(X_train, y_train)
            pred_trtr = model.predict(X_test)
            
            # 2. TSTR
            model.fit(X_synth, y_synth)
            pred_tstr = model.predict(X_test)
            
            # 점수
            if is_classification:
                s_trtr = f1_score(y_test, pred_trtr, average='macro')
                s_tstr = f1_score(y_test, pred_tstr, average='macro')
            else:
                s_trtr = r2_score(y_test, pred_trtr)
                s_tstr = r2_score(y_test, pred_tstr)
                
            print(f"   - TRTR: {s_trtr:.4f} / TSTR: {s_tstr:.4f}")
            results.append({'Target': target, 'TRTR': s_trtr, 'TSTR': s_tstr})
            
        except Exception as e:
            print(f"   [Error] 모델 학습 중 오류: {e}")

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# [4단계] 시각화
# -----------------------------------------------------------------------------
def plot_correlation(real, synth):
    if '기초금액' not in real.columns or '최종낙찰금액' not in real.columns: return

    # 시각화 전 결측 제거
    real = real.replace([np.inf, -np.inf], np.nan).dropna(subset=['기초금액', '최종낙찰금액'])
    synth = synth.replace([np.inf, -np.inf], np.nan).dropna(subset=['기초금액', '최종낙찰금액'])

    plt.figure(figsize=(12, 5))
    n = min(1000, len(real))
    
    plt.subplot(1, 2, 1)
    if len(real) > 0:
        samp = real.sample(n)
        sns.scatterplot(x=samp['기초금액'], y=samp['최종낙찰금액'], alpha=0.5)
        corr = samp[['기초금액','최종낙찰금액']].corr().iloc[0,1]
        plt.title(f"원본 데이터 (Corr={corr:.4f})")
    
    plt.subplot(1, 2, 2)
    if len(synth) > 0:
        samp = synth.sample(n)
        sns.scatterplot(x=samp['기초금액'], y=samp['최종낙찰금액'], color='orange', alpha=0.5)
        corr = samp[['기초금액','최종낙찰금액']].corr().iloc[0,1]
        plt.title(f"재현 데이터 (Corr={corr:.4f})")
    
    plt.tight_layout()
    plt.savefig("Hybrid_Scatter_Final.png")
    print("\n[Visualization] 산점도 그래프 저장 완료: Hybrid_Scatter_Final.png")

# -----------------------------------------------------------------------------
# [Main]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    set_korean_font()
    
    try:
        # 1. 전처리
        df_real = load_and_preprocess()
        
        # 2. 생성
        df_synth = generate_hybrid(df_real)
        df_synth.to_csv(OUTPUT_SYNTHETIC, index=False, encoding='utf-8-sig')
        print(f"[System] 파일 저장 완료: {OUTPUT_SYNTHETIC}")
        
        # 3. 평가
        res_df = evaluate_tstr(df_real, df_synth)
        
        # 4. 시각화
        plot_correlation(df_real, df_synth)
        
        print("\n[Final Summary]")
        print(res_df)
        
    except Exception as e:
        print(f"[Critical Error] {e}")