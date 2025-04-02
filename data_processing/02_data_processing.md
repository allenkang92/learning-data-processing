# 데이터 처리 심화

## 1. 데이터 전처리 파이프라인

### 1.1 데이터 전처리의 중요성
- **모델 성능 향상**: 깨끗한 데이터는 더 정확한 예측 모델을 생성합니다.
- **학습 안정성**: 이상치와 결측값이 제거된 데이터는 모델 학습 과정을 안정화합니다.
- **연산 효율성**: 불필요한 특성이 제거된 데이터는 연산 시간과 리소스를 절약합니다.

### 1.2 데이터 전처리 단계
1. **데이터 수집 및 통합**
2. **데이터 정제**
3. **특성 공학**
4. **데이터 변환**
5. **데이터 축소**

## 2. 결측치 처리 기법

### 2.1 결측치 식별
```python
import pandas as pd
import numpy as np

# 결측치 개수 확인
missing_values = df.isnull().sum()

# 결측치 비율 확인
missing_percentage = df.isnull().mean() * 100
```

### 2.2 결측치 처리 방법
1. **제거 (Deletion)**
   ```python
   # 행 제거
   df_cleaned = df.dropna()
   
   # 열 제거 (결측치가 50% 이상인 열)
   df_cleaned = df.loc[:, df.isnull().mean() < 0.5]
   ```

2. **대체 (Imputation)**
   ```python
   # 수치형 데이터: 평균, 중앙값, 최빈값으로 대체
   df['age'].fillna(df['age'].mean(), inplace=True)
   df['income'].fillna(df['income'].median(), inplace=True)
   
   # 범주형 데이터: 최빈값 또는 'Unknown' 등으로 대체
   df['category'].fillna(df['category'].mode()[0], inplace=True)
   
   # 시계열 데이터: 이전/이후 값으로 대체
   df['sales'].fillna(method='ffill', inplace=True)  # 이전 값으로 대체
   df['sales'].fillna(method='bfill', inplace=True)  # 이후 값으로 대체
   ```

3. **고급 대체 기법**
   ```python
   # KNN 기반 대체
   from sklearn.impute import KNNImputer
   imputer = KNNImputer(n_neighbors=5)
   df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
   
   # 다중 대체 (Multiple Imputation)
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer
   imputer = IterativeImputer(max_iter=10, random_state=42)
   df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
   ```

## 3. 이상치 탐지 및 처리

### 3.1 이상치 탐지 방법
1. **통계적 방법**
   ```python
   # Z-score 방법
   from scipy import stats
   z_scores = stats.zscore(df['value'])
   outliers = df[np.abs(z_scores) > 3]
   
   # IQR 방법
   Q1 = df['value'].quantile(0.25)
   Q3 = df['value'].quantile(0.75)
   IQR = Q3 - Q1
   lower_bound = Q1 - 1.5 * IQR
   upper_bound = Q3 + 1.5 * IQR
   outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
   ```

2. **기계학습 기반 방법**
   ```python
   # 고립 포레스트 (Isolation Forest)
   from sklearn.ensemble import IsolationForest
   clf = IsolationForest(contamination=0.05, random_state=42)
   outlier_labels = clf.fit_predict(df[['value1', 'value2']])
   outliers = df[outlier_labels == -1]
   
   # LOF (Local Outlier Factor)
   from sklearn.neighbors import LocalOutlierFactor
   lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
   outlier_labels = lof.fit_predict(df[['value1', 'value2']])
   outliers = df[outlier_labels == -1]
   ```

### 3.2 이상치 처리 방법
1. **제거**: 이상치를 데이터셋에서 제거합니다.
2. **변환**: 이상치를 정상 범위 내의 값으로 변환합니다 (capping, winsorizing).
3. **격리**: 이상치를 별도로 분석하거나 모델링합니다.
4. **특수 처리**: 이상치를 나타내는 새로운 특성을 생성합니다.

## 4. 특성 공학 (Feature Engineering)

### 4.1 특성 생성
```python
# 수학적 변환
df['log_income'] = np.log1p(df['income'])
df['sqrt_age'] = np.sqrt(df['age'])

# 날짜/시간 특성
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 텍스트 특성
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
```

### 4.2 특성 선택
1. **필터 방법 (Filter Methods)**
   ```python
   # 분산 기반 선택
   from sklearn.feature_selection import VarianceThreshold
   selector = VarianceThreshold(threshold=0.05)
   X_selected = selector.fit_transform(X)
   
   # 상관관계 기반 선택
   corr_matrix = df.corr().abs()
   upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
   to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
   df_selected = df.drop(to_drop, axis=1)
   ```

2. **래퍼 방법 (Wrapper Methods)**
   ```python
   # 재귀적 특성 제거 (RFE)
   from sklearn.feature_selection import RFE
   from sklearn.linear_model import LogisticRegression
   estimator = LogisticRegression()
   selector = RFE(estimator, n_features_to_select=10, step=1)
   selector = selector.fit(X, y)
   X_selected = X.iloc[:, selector.support_]
   ```

3. **임베디드 방법 (Embedded Methods)**
   ```python
   # LASSO를 이용한 특성 선택
   from sklearn.linear_model import Lasso
   from sklearn.feature_selection import SelectFromModel
   lasso = Lasso(alpha=0.1)
   selector = SelectFromModel(lasso)
   X_selected = selector.fit_transform(X, y)
   ```

## 5. 데이터 변환 및 정규화

### 5.1 인코딩 기법
1. **범주형 인코딩**
   ```python
   # 원-핫 인코딩
   df_encoded = pd.get_dummies(df, columns=['category'])
   
   # 라벨 인코딩
   from sklearn.preprocessing import LabelEncoder
   le = LabelEncoder()
   df['category_encoded'] = le.fit_transform(df['category'])
   
   # 타겟 인코딩
   target_mean = df.groupby('category')['target'].mean()
   df['category_target_encoded'] = df['category'].map(target_mean)
   ```

2. **텍스트 인코딩**
   ```python
   # Count Vectorization
   from sklearn.feature_extraction.text import CountVectorizer
   vectorizer = CountVectorizer()
   X_count = vectorizer.fit_transform(df['text'])
   
   # TF-IDF
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer()
   X_tfidf = vectorizer.fit_transform(df['text'])
   ```

### 5.2 스케일링 기법
```python
# 표준화 (Standard Scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 정규화 (Min-Max Scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 로버스트 스케일링 (이상치에 강함)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

## 6. 분산 데이터 처리 (Spark)

### 6.1 Spark 기본
```python
from pyspark.sql import SparkSession

# SparkSession 생성
spark = SparkSession.builder \
    .appName("Data Processing") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
```

### 6.2 Spark DataFrame 처리
```python
# CSV 파일 로드
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# 기본 데이터 처리
df_cleaned = df.dropna()  # 결측치 제거
df_filtered = df.filter(df["value"] > 0)  # 필터링
df_grouped = df.groupBy("category").agg({"value": "mean"})  # 그룹화 및 집계

# SQL 쿼리 사용
df.createOrReplaceTempView("data_table")
result = spark.sql("SELECT category, AVG(value) as avg_value FROM data_table GROUP BY category")
```

### 6.3 Spark ML 파이프라인
```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# 특성 컬럼 통합
assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")

# 스케일링
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# 모델 정의
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")

# 파이프라인 생성
pipeline = Pipeline(stages=[assembler, scaler, lr])

# 파이프라인 실행
model = pipeline.fit(train_data)
predictions = model.transform(test_data)
```

## 7. 데이터 처리 자동화

### 7.1 작업 스케줄링
```python
# Airflow를 이용한 데이터 처리 자동화
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email': ['email@example.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='A data processing pipeline',
    schedule_interval=timedelta(days=1),
)

def extract_data():
    # 데이터 추출 로직
    pass

def transform_data():
    # 데이터 변환 로직
    pass

def load_data():
    # 데이터 로드 로직
    pass

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

extract_task >> transform_task >> load_task
```

### 7.2 MLOps 및 자동화 도구
- **DVC (Data Version Control)**
- **MLflow**
- **Kubeflow**
- **Apache NiFi**
- **Prefect**

## 8. 실전 데이터 처리 베스트 프랙티스

### 8.1 성능 최적화
- **병렬 처리**: 대용량 데이터의 경우 다중 코어/프로세스 활용
- **메모리 관리**: 청크 단위 처리로 메모리 사용량 제한
- **이중 반복문 피하기**: 벡터화 연산 활용
- **데이터 타입 최적화**: 필요한 정밀도만큼만 메모리 사용

### 8.2 테스트 및 검증
- **데이터 품질 테스트**: 결측치, 이상치, 중복 검사
- **파이프라인 검증**: 입력/출력 형식, 값 범위 검증
- **교차 검증**: 데이터 처리 단계가 다양한 하위 집합에서 일관되게 작동하는지 확인

### 8.3 문서화 및 버전 관리
- **데이터 사전**: 각 특성에 대한 설명과 메타데이터 관리
- **코드 문서화**: 주석과 문서 문자열을 통해 코드 이해도 향상
- **버전 관리**: Git을 이용한 코드 버전 관리 및 DVC를 이용한 데이터 버전 관리

## 9. 데이터 분류와 품질 관리

### 9.1 데이터 분류

- **데이터 분류 방식의 다양성**: 데이터를 분류하는 방식은 매우 다양하며, 문맥에 따라 적절한 분류 방식을 적용해야 합니다.
- **정형, 반정형, 비정형 데이터**:
    - **정형 데이터**: 독립성 가정을 전제로 합니다. (예: seaborn의 tips 데이터셋)
        - 데이터셋은 순서가 중요하지 않지만, 시계열 데이터는 순서(시간)에 종속적입니다.
        - 회귀 분석 시 독립성이 깨지는 경우 다중공선성 개념을 사용합니다.
        - 정형 데이터는 기본적으로 독립적이어야 하지만, 예외적인 상황(영상 데이터, 나이브 베이즈 등)에서는 독립성이 깨질 수 있습니다.
    - **비정형 데이터**: 정형 데이터로 가정하고 분석하면 성능이 좋지 않습니다. 딥러닝은 비정형 데이터에 강점을 가집니다.
- **데이터 타입 분류**:
    - **값의 특성**: 순서 유무, 절대값/상대값 여부, 연속/불연속 여부
    - **통계적 분류**: 명목(nominal), 순서(ordinal), 정수(불연속), 실수(연속) 데이터 등
    - **데이터 구조**: 롱/와이드 포맷 등

### 9.2 데이터 처리 및 전처리의 중요성

- **현실 세계의 데이터**: 지저분하며(messy), 사이킷런은 결측치(missing data)에 대해 경고합니다. 결측치 처리는 분석의 전제조건입니다.
    - **Dirty data 예시**:
        - **잘못된 데이터(bad/wrong/invalid/Irrelevant)**: 철자 오류, 잘못된 값, 무응답 등
        - **결측치(missing)**: 결혼 여부, 월급 등 무응답
        - **오래된 정보(obsolete)**: 업데이트되지 않은 정보 (나이, 주소, 전화번호)
        - **비표준(non-standard)**: 동일한 정보를 다른 형식으로 표현 ("홍길동", "홍**", "홍*동")
        - **불완전(incomplete)**: 불완전한 정보 (주소 일부 누락)
        - **중복(redundant/duplicated)**: 중복 데이터
        - **비일관적(Inconsistent)**: 일관성 없는 데이터
        - **노이즈/이상치(noisy/outlier)**: 이상치
    - **GIGO (Garbage in, Garbage out)**: 잘못된 데이터 입력은 잘못된 결과로 이어집니다.
- **데이터셋 vs. 데이터 시퀀스**: 데이터셋은 중복이 없지만, 데이터 시퀀스는 중복이 있을 수 있습니다.
- **노이즈(noise)**: 엉터리 값 추가는 데이터 분석에서 가장 골치 아픈 문제 중 하나입니다.
- **데이터 전처리**:
    - **필요성**: 의미 있는 데이터로 변환하는 과정은 분석의 첫 단계입니다.
    - **데이터 클리닝(Data Cleaning)**: 데이터 과학자는 대부분의 시간을 데이터 클리닝 및 포맷팅에 할애합니다. (Data wrangling/munging)
        - 80%의 노력이 80%의 가치를 결정합니다. (T. Dasu and T. Johnson)
    - **Data cleansing, data cleaning, data scrubbing**: 대체로 같은 의미로 사용되지만, data scrubbing은 중복, 불필요, 오래된 데이터를 제거하는 데 초점을 맞추기도 합니다.
    - **Data Preparation = Data Cleansing + Feature Engineering**
    - **Data Preprocessing vs Data Wrangling**
        - **Data Preprocessing**: 데이터 소스 접근 직후 데이터 준비, 개발자/데이터 과학자가 초기 변환, 집계, 데이터 정제 수행 (한 번 실행)
        - **Data Wrangling**: 대화형 데이터 분석 및 모델 구축 중 데이터 준비, 데이터 과학자/분석가가 데이터 뷰 변경 및 특성 엔지니어링 수행 (반복적)
    - **Data Preprocessing 단계**
        - Data Cleaning
        - Data Integration
        - Data Transformation
        - Data Reduction
            - sampling, transformation, denoising, imputation, normalization, feature extraction

### 9.3 현실 데이터 오류의 원인

- **인적 오류(Human Errors)**: 사람에 의한 오류 (예: 기부 폼 작성 오류)
- **기계 오류(Machine Errors)**: 기계에 의한 오류
    - BMI 계산 코드 버그, 센서/기계 결함, 시계 정확도, 렌즈 먼지, 네트워크 전송 오류, 우주선(cosmic rays) 등
    - 직렬화 오류(serialization error): big-endian/little-endian 포맷 불일치 등
- **온도 측정 오류**: 제조 AI에서 측정 단계부터 오류 가능성을 고려해야 합니다.
- **도메인 지식**: 통계적으로는 엉터리 데이터지만 도메인 지식으로는 맞는 데이터일 수 있으므로, 도메인 지식도 중요합니다.

### 9.4 오류 탐지 (Detecting Errors)

- **도메인 지식(Domain knowledge)**: 도메인 지식 활용
- **하위 그룹(Subgroups)**: 하위 그룹 분석
- **구조(Scheme)**: 데이터 구조 활용
- **검증(Validation)**: 검증
- **편향(Bias)**: 편향으로 인한 실수 발생 가능성
- **인지 편향**: 빅데이터 분석 시 모든 데이터를 확인하기 어려우므로, 민감한 데이터는 수동 검토(check & double-check)가 필요합니다.
- **범위 파악**: max-min 값으로 데이터 범위를 파악합니다.
- **시스템적 오류 탐지**:
    - 데이터 구조를 잘 설계하면 오류를 쉽게 파악할 수 있습니다.
    - JSON 스키마, protocol buffer(Kaggle), 데이터베이스 설계, ORM 설계 등 활용
    - cue lang(텍스트 처리), pydantic(웹 프로그래밍 구조화), dataclass(pydantic의 쉬운 버전)
    - pandera(pandas 확장 밸리데이션 스키마), marshmallow(웹 ORM, SQL Alchemy와 유사)
- **Data Profiling 주요 단계**
    - 여러 소스 시스템에서 데이터 및 메타데이터 수집
    - 데이터 통합 및 정제
    - 통계 수집, 특성 파악 및 문제 식별
- **일반적인 Data Quality Metrics**
    - 월별/분기별 데이터 오류 수 및 수정 수
    - 데이터 정확도 및 오류율 (허용 수준 초과 시 경고)
    - 데이터 완전성, 일관성, 무결성, 적시성에 대한 정량적 측정
    - 데이터 정의, 메타데이터, 데이터 카탈로그의 품질 수준 평가
    - 사용자 피드백

### 9.5 문제 정의와 해결

- **문제 정의**:
    - 분석 경험이 많으면 문제 정의와 함께 추가적인 인사이트를 요구받는 경우가 있습니다.
    - 문제 정의가 주어지면 기존에 존재하는 해결책(경험 있는 사람들이 쓰는 틀)을 먼저 활용합니다.
    - 해결책을 모른다면 배우는 것이 핵심입니다.
- **해결책 탐색**:
    - 문제에 맞는 기법들을 정리하고, 논문을 통해 해결책을 찾습니다.
    - 도메인에 따라 해결책은 다르며, 통계 기법을 사용하는 다양한 방식이 존재합니다.
- **데이터 구조 설계**: 기법에 맞는 데이터 구조 설계부터 시작합니다. (초보자는 시행착오 필요)
- **데이터 랭글링(Data Wrangling)**: 범용적으로 잘 설계된 모델을 사용하기 위해 전처리 및 변환하는 과정입니다.
- **문제 해결 방식**:
    - 전통적인 방법: 단일 문제 해결에 적합
    - 인공지능: 복합적인 문제 해결, 병렬 처리 가능 (시퀀스 모델은 순차적 처리)

### 9.6 오류 방지 (Preventing Errors)

- **직렬화 형식(Serialization formats)**: CSV, JSON, Apache Parquet 등
- **디지털 서명(Digital signature)**: 데이터 무결성 검증
- **ETL/ELT, 데이터 파이프라인(Data pipelines)**: Apache Airflow 등
- **자동화(Automation)**: Invoke 등 도구 활용
- **트랜잭션(Transactions)**: 데이터 일관성 유지
- **데이터 구성과 정돈된 데이터(Data organization and tidy data)**: long form 등
- **프로세스 및 데이터 품질 지표(Process and data quality metrics)**: Prometheus, InfluxDB 등

### 9.7 오류 수정 (Fixing Errors)

- NumPy 관점에서 오류 해결
- 인덱싱과 슬라이싱을 활용하여 문제 발견 및 데이터 처리
- **구체적인 방법**:
    - 필드 이름 변경(Renaming fields)
    - 타입 수정(Fixing types)
    - 데이터 결합 및 분할(Joining and splitting data)
    - 잘못된 데이터 삭제(Deleting bad data)
    - 결측값 채우기(Filling missing values): Listwise Deletion, Imputation 등
    - 데이터 재구성(Reshaping data)
    - 이상치 탐지(Outlier detection): Statistics-based, Distance-based, Model-based
    - 일관성 없는 범주형 데이터 처리(Incoherent categorical data): Cap/Floor, Database functions, Substitution, Programming
    - 데이터 중복 제거(Data Deduplication)
    - 행 압축(Row Compression)
    - 열 압축(Column Compression)

### 9.8 데이터 누출 (Data Leakage)

- **정의**: 훈련 데이터셋에 테스트 데이터셋 또는 미래의 정보가 포함되어 모델 성능이 과대평가되는 현상
- **원인**:
    - 전처리: 전체 데이터셋에 전처리 적용
    - 테스트/검증 데이터 중복
    - 시간 관련 데이터에서 시간 순서 고려 X
- **방지 기법**:
    - Cross-validation 전 정규화
    - 데이터셋 분할 (훈련, 검증, 테스트)
    - 중복 제거
- **Dataset shift**
    - Covariate shift: 입력 분포가 변경되지만 출력 분포는 유지됨
    - Prior probability shift: 출력 분포가 변경되지만 입력 분포는 유지됨
    - Concept shift: 입력과 출력 간의 관계가 변경됨
    - Internal covariate shift: 신경망 내부 레이어의 입력 분포 변화

### 9.9 머신러닝 파이프라인

머신러닝 프로젝트의 일반적인 파이프라인 단계:

1. **Raw Data Collection**: 원시 데이터 수집
2. **Pre-Processing**:
   - Missing Data 처리
   - Feature Extraction
   - Sampling
   - Feature Selection
   - Feature Scaling
3. **Split**: 훈련 데이터셋, 테스트 데이터셋 분리
4. **Supervised Learning**:
   - Learning Algorithm 적용
   - Cross Validation
   - Hyperparameter Optimization
   - Model Selection
5. **Final Model Evaluation**: 성능 지표 평가
6. **Prediction**:
   - Post-Processing
   - Final Classification/Regression Model
7. **Refinement**: 모델 개선 및 튜닝
8. **New Data**: 새로운 데이터에 적용

### 9.10 데이터 처리 핵심 원칙

- **"You don't have to reinvent the wheel"**: 이미 존재하는 해결책이나 도구를 활용하라는 의미입니다. 데이터 처리에서도 검증된 방법과 도구를 활용하는 것이 효율적입니다.
- **데이터 품질이 분석 품질을 결정합니다**: 고품질 데이터가 고품질 분석을 가능하게 합니다.
- **도메인 지식의 중요성**: 기술적 지식과 함께 도메인 지식을 활용하면 더 정확한 데이터 처리가 가능합니다.
- **반복적 과정**: 데이터 전처리는 일회성이 아닌 반복적인 과정으로, 지속적인 개선이 필요합니다.

## 10. 데이터 처리의 어려움과 접근 방식

### 10.1 현실 데이터의 특성
- **데이터 "Messiness"**: 현실 데이터는 대부분 지저분하며(messy), 다양한 오류를 포함합니다.
- **데이터 오류 유형**:
  - 결측값, 중복값, 이상치
  - 포맷 불일치, 타입 불일치
  - 시스템적 오류, 인적 오류
- **도전과제**: 오류가 포함된 데이터로부터 의미 있는 정보를 추출해야 합니다.

### 10.2 데이터 오류 감지 방법
- **소규모 데이터**: 
  - 수작업(manpower)으로 데이터 검토 가능
  - 일반적으로 10,000행 이하의 데이터에 적합
- **빅데이터**:
  - 수작업 불가능, 자동화된 방법 필요
  - 데이터 프로파일링 도구 활용
  - 통계적 분석 방법 적용

### 10.3 데이터 오류 방지 전략
- **유효성 검사(Validation)**:
  ```python
  # 파이썬 기본 검증
  assert 0 <= age <= 120, "나이가 유효 범위를 벗어났습니다"
  
  # Pandas를 이용한 검증
  import pandas as pd
  assert df.duplicated().sum() == 0, "중복 데이터가 존재합니다"
  assert df['age'].between(0, 120).all(), "나이 데이터에 오류가 있습니다"
  
  # pandera를 이용한 스키마 검증
  import pandera as pa
  schema = pa.DataFrameSchema({
      "age": pa.Column(int, pa.Check.in_range(0, 120)),
      "email": pa.Column(str, pa.Check.str_matches(r"^[\w\.-]+@[\w\.-]+\.\w+$"))
  })
  validated_df = schema.validate(df)
  ```

- **자동화된 시스템 구축**:
  - 데이터 입력 단계에서 규칙(rule) 적용
  - 시스템적 오류를 줄이는 파이프라인 설계

### 10.4 데이터 오류 수정 접근법
- **데이터 파이프라인 구축**:
  - ETL(Extract, Transform, Load) 프로세스 설계
  - 자동화된 데이터 정제 단계 포함
- **지속적인 모니터링과 개선**:
  - 오류 패턴 파악 및 추적
  - 피드백 루프를 통한 지속적 개선

## 11. NumPy와 Pandas 개요

### 11.1 NumPy 기본 개념
- **특징**:
  - 다차원 배열(ndarray)을 효율적으로 처리
  - C로 구현되어 빠른 연산 속도 제공
  - 과학 계산, 데이터 분석, 머신러닝의 기반 라이브러리
- **구조화된 배열(Structured Array)**:
  ```python
  import numpy as np
  
  # 구조화된 배열 생성
  dtype = [('name', 'U10'), ('age', 'i4'), ('salary', 'f4')]
  employees = np.array([
      ('Alice', 30, 75000.0),
      ('Bob', 35, 85000.0),
      ('Charlie', 40, 95000.0)
  ], dtype=dtype)
  
  # 접근 방법
  print(employees[0])               # 첫 번째 행 접근
  print(employees['name'])          # 'name' 열 접근
  
  # 레코드 배열로 변환하여 점 표기법 사용
  rec_array = employees.view(np.recarray)
  print(rec_array.name)             # 'name' 열에 점 표기법으로 접근
  ```

### 11.2 Pandas 소개
- **배경**:
  - Wes McKinney가 개발한 데이터 분석 라이브러리
  - NumPy를 기반으로 구축
  - '프로그래밍 가능한 엑셀'로 비유됨
- **주요 구조**:
  - **Series**: 1차원 레이블이 있는 배열
  - **DataFrame**: 2차원 테이블 형태의 데이터 구조
    - 각 열(column)은 동일한 데이터 타입(homogeneous)
    - 행(row)은 서로 다른 데이터 타입(heterogeneous) 가능
  - **Panel**: 3차원 데이터(여러 개의 DataFrame)

### 11.3 NumPy와 Pandas 비교
- **공통점**:
  - 벡터화된 연산 지원
  - C 기반 고성능 구현
- **차이점**:
  - NumPy: 수치 연산에 최적화, 동일 데이터 타입 요구
  - Pandas: 데이터 분석에 최적화, 열별로 다른 데이터 타입 허용

## 12. Pandas 데이터 탐색 및 분석

### 12.1 데이터 세트 로드 및 기본 정보
```python
import pandas as pd
import seaborn as sns

# 예제 데이터셋 로드
tips = sns.load_dataset('tips')

# 데이터 구조 확인
print(type(tips))                # pandas.core.frame.DataFrame
print(dir(tips))                 # 사용 가능한 메서드와 속성 확인

# 데이터 기본 정보
tips.info()                      # 컬럼, 데이터 타입, 결측치 등 기본 정보
print(tips.shape)                # 행과 열 수 (튜플)
```

### 12.2 데이터 개요 확인
```python
# 데이터 미리보기
print(tips.head())               # 처음 5개 행
print(tips.tail(3))              # 마지막 3개 행
print(tips.sample(5))            # 무작위 5개 행

# 기술 통계량 (descriptive statistics)
print(tips.describe())           # 수치형 컬럼의 통계량
print(tips.describe(include='category'))  # 범주형 컬럼의 통계량
print(tips.describe(include='all'))       # 모든 컬럼의 통계량
```

### 12.3 범주형 데이터 분석
```python
# 고유값 확인
print(tips['sex'].unique())      # 중복 제거된 고유값 목록
print(tips['sex'].nunique())     # 고유값 개수

# 빈도 분석
print(tips['sex'].value_counts())                # 절대 빈도
print(tips['sex'].value_counts(normalize=True))  # 상대 빈도(비율)
print(tips['sex'].value_counts(dropna=False))    # 결측치 포함

# 그룹별 통계
print(tips.groupby('sex')['total_bill'].mean())  # 성별에 따른 총액 평균
print(tips.groupby(['sex', 'smoker'])['tip'].agg(['mean', 'count']))  # 다중 통계
```

## 13. Pandas 인덱싱과 슬라이싱

### 13.1 기본 인덱싱 방법
```python
# 열 선택
bill_column = tips['total_bill']             # Series 반환
bill_df = tips[['total_bill']]               # DataFrame 반환 (차원 유지)
subset = tips[['total_bill', 'tip', 'sex']]  # 여러 열 선택 (팬시 인덱싱)

# 행 선택 (위치 기반)
first_row = tips.iloc[0]         # 첫 번째 행
first_three = tips.iloc[0:3]     # 처음 3개 행 (0, 1, 2)
specific_rows = tips.iloc[[0, 2, 4]]  # 특정 행 선택 (팬시 인덱싱)

# 행 선택 (레이블 기반)
first_row_label = tips.loc[0]    # 인덱스가 0인 행
range_rows = tips.loc[0:3]       # 인덱스가 0부터 3까지의 행 (3 포함)
```

### 13.2 고급 인덱싱 기법
```python
# 불린 인덱싱
female_customers = tips[tips['sex'] == 'Female']
high_tippers = tips[tips['tip'] > 5]
complex_filter = tips[(tips['sex'] == 'Female') & (tips['smoker'] == 'Yes')]

# 행과 열 동시 선택
specific_cell = tips.iloc[0, 1]  # 0행 1열의 값
specific_cell_label = tips.loc[0, 'tip']  # 인덱스가 0인 행의 'tip' 열 값
subset_loc = tips.loc[0:2, ['total_bill', 'tip']]  # 0~2행의 지정 열만 선택

# 빠른 접근 (단일 값)
fast_access_1 = tips.iat[0, 1]   # iloc 보다 빠름
fast_access_2 = tips.at[0, 'tip']  # loc 보다 빠름
```

### 13.3 성능 및 주의사항
```python
# 성능 측정 (Jupyter Notebook)
%timeit tips.loc[0, 'tip']
%timeit tips.at[0, 'tip']        # at이 더 빠름

# 데이터 수정 시 주의
# 괜찮은 방법
tips.loc[0, 'smoker'] = 'Yes'

# 경고 발생 가능 (SettingWithCopyWarning)
female_customers = tips[tips['sex'] == 'Female']
female_customers['smoker'] = 'Yes'  # 복사본 수정 문제 발생 가능

# 대신 이렇게 사용
tips.loc[tips['sex'] == 'Female', 'smoker'] = 'Yes'  # 권장 방법
```

## 14. 파일 입출력과 데이터 저장

### 14.1 다양한 파일 형식
```python
# CSV 파일
tips.to_csv('tips.csv', index=False)
tips_loaded = pd.read_csv('tips.csv')

# Excel 파일
tips.to_excel('tips.xlsx', sheet_name='tip_data', index=False)
tips_excel = pd.read_excel('tips.xlsx', sheet_name='tip_data')

# Pickle 파일 (Python 객체 직렬화)
tips.to_pickle('tips.pkl')
tips_pickle = pd.read_pickle('tips.pkl')

# Parquet 파일 (컬럼 기반 저장)
tips.to_parquet('tips.parquet')
tips_parquet = pd.read_parquet('tips.parquet')

# JSON 파일
tips.to_json('tips.json', orient='records')
tips_json = pd.read_json('tips.json')
```

### 14.2 데이터 저장 시 고려사항
- **텍스트 파일 (CSV, JSON 등)**:
  - 장점: 가독성, 호환성 높음
  - 단점: 데이터 타입 정보 손실, 용량 큼
- **이진 파일 (Pickle, Parquet 등)**:
  - 장점: 데이터 타입 보존, 압축 지원, 속도 빠름
  - 단점: 가독성 없음, 버전 호환성 문제 가능
- **압축 옵션**:
  ```python
  # 압축 적용
  tips.to_csv('tips.csv.gz', compression='gzip')
  tips.to_parquet('tips.parquet', compression='snappy')
  
  # NumPy 압축 저장
  import numpy as np
  arr = np.arange(100).reshape(10, 10)
  np.savez_compressed('data.npz', array=arr)
  ```

## 15. 데이터 변환과 통계 분석

### 15.1 데이터 이름 변경
```python
# 컬럼 이름 변경
tips_renamed = tips.rename(columns={
    'total_bill': 'total_amount',
    'size': 'party_size'
})

# 인덱스 이름 변경
tips_index_renamed = tips.rename(index={0: 'first', 1: 'second'})

# 원본 변경
tips.rename(columns={'tip': 'tip_amount'}, inplace=True)

# 일괄 변경 (모든 컬럼명 소문자로)
tips.columns = [col.lower() for col in tips.columns]
```

### 15.2 부분집합 분석과 이상치 탐지
```python
# 부분집합 분석
by_sex = tips.groupby('sex')[['total_bill', 'tip']].describe()
by_day = tips.groupby('day')['tip'].agg(['min', 'max', 'mean', 'std'])

# 이상치 탐지
q1 = tips['total_bill'].quantile(0.25)
q3 = tips['total_bill'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = tips[(tips['total_bill'] < lower_bound) | (tips['total_bill'] > upper_bound)]
print(f"이상치 개수: {len(outliers)}")

# 박스플롯을 통한 이상치 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('요일별 계산서 금액의 분포')
plt.show()
```

## 16. NumPy의 축(Axis), 스트라이드(Stride), 리듀스(Reduce) 개념

### 16.1 축(Axis)과 배열 구조
```python
import numpy as np

# 2차원 배열 생성
arr_2d = np.arange(12).reshape(3, 4)
print(arr_2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# 배열 정보 확인
print(f"Shape: {arr_2d.shape}")      # (3, 4)
print(f"Dimensions: {arr_2d.ndim}")  # 2
print(f"Data type: {arr_2d.dtype}")  # int64
print(f"Item size: {arr_2d.itemsize}")  # 8 (bytes)
print(f"Strides: {arr_2d.strides}")  # (32, 8) - 행과 열 이동 시 바이트 수
```

### 16.2 축(Axis)의 개념과 이해

- **정의:** Axis는 다차원 배열에서 연산이 적용되는 방향을 지정하는 개념입니다.
- **NumPy에서의 Axis:**
  - NumPy 배열은 여러 개의 축(axis)을 가질 수 있습니다:
    - 1차원 배열: 축 0 (하나의 축)
    - 2차원 배열: 축 0 (행), 축 1 (열)
    - 3차원 배열: 축 0, 축 1, 축 2
  - `axis` 파라미터를 사용하여 연산을 적용할 축을 지정합니다.
  - `axis=None` (기본값): 배열의 모든 요소에 대해 연산을 수행합니다.

```python
import numpy as np

# 2차원 배열
a = np.arange(12).reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

print(a.sum(axis=0))  # [12 15 18 21] (열별 합계, shape: (4,))
print(a.sum(axis=1))  # [ 6 22 38] (행별 합계, shape: (3,))
print(a.sum(axis=None)) # 66 (전체 합계)
print(a.sum()) # 66 (axis=None과 동일)

# 3차원 배열
b = np.arange(24).reshape(2, 3, 4)
# [[[ 0  1  2  3]
#   [ 4  5  6  7]
#   [ 8  9 10 11]]
#
#  [[12 13 14 15]
#   [16 17 18 19]
#   [20 21 22 23]]]

print(b.sum(axis=0))
# [[12 14 16 18]
#  [20 22 24 26]
#  [28 30 32 34]] (shape: (3, 4))

print(b.sum(axis=1))
# [[12 15 18 21]
#  [48 51 54 57]] (shape: (2, 4))

print(b.sum(axis=2))
# [[ 6 22 38]
#  [54 70 86]] (shape: (2, 3))
```

- **axis와 차원 축소:** `axis`를 지정하면 해당 축이 사라지고, 나머지 축으로 구성된 배열이 반환됩니다. 즉, 차원 축소가 발생합니다.

- **Pandas에서의 Axis:**
  - Pandas DataFrame은 2차원 배열(테이블) 형태이므로, 일반적으로 `axis=0` (행)과 `axis=1` (열)을 사용합니다.
  - `axis=0` (또는 `axis='index'`): 행 방향 (↓)
  - `axis=1` (또는 `axis='columns'`): 열 방향 (→)

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')

print(tips.sum(axis=0)) # 각 열의 합계 (결측치 제외)
print(tips.mean(axis=1)) # 각 행의 숫자형 열의 평균 (결측치 제외)
```

- **주의:** Pandas의 `axis`는 NumPy의 `axis`와 동작 방식이 약간 다릅니다. Pandas에서는 `axis`가 연산이 적용되는 방향을 나타내는 동시에, 연산 후 남는 축을 의미하기도 합니다.

### 16.3 리듀스(Reduce) 연산의 심화 개념

- **정의:** Reduce는 여러 개의 값을 하나의 값으로 줄이는 연산을 의미합니다. 차원을 축소하는 연산입니다.
- **대표적인 Reduce 연산:**
  - **`sum()`:** 합계
  - **`mean()`:** 평균
  - **`min()`:** 최솟값
  - **`max()`:** 최댓값
  - **`std()`:** 표준편차
  - **`var()`:** 분산
  - **`median()`:** 중앙값
  - **`prod()`:** 모든 요소의 곱
  - **`any()`:** 하나 이상의 요소가 True인지 확인
  - **`all()`:** 모든 요소가 True인지 확인

- **NumPy에서의 Reduce:**

```python
import numpy as np

a = np.arange(1, 6)  # [1 2 3 4 5]
print(np.sum(a))  # 15 (배열의 모든 요소 합계)
print(np.mean(a)) # 3.0 (배열의 모든 요소 평균)
```

- **Pandas에서의 Reduce:**

```python
import pandas as pd
import seaborn as sns

tips = sns.load_dataset('tips')
print(tips['tip'].mean())  # 팁의 평균
print(tips.total_bill.max()) # 최대 total_bill 값
```

- **Reduce의 중요성:**
  - **데이터 요약:** Reduce 연산을 통해 데이터의 전체적인 특성을 파악할 수 있습니다. (예: 평균, 표준편차를 통해 데이터의 중심 경향과 분산 정도 파악)
  - **통계적 예측:** Reduce된 값(예: 평균)을 사용하여 미래의 값을 예측하는 데 활용 가능합니다.
  - **이상치 탐지:** 평균, 표준편차 등 Reduce된 값을 기준으로 데이터에서 크게 벗어나는 값(이상치)을 탐지할 수 있습니다.
  - **차원 축소:** Reduce 연산은 데이터의 차원을 줄이는 효과도 있습니다. (예: 2차원 배열에서 행별 합계를 구하면 1차원 배열이 됨)

### 16.4 Reduce와 Axis의 관계

1. **Axis 지정:** 연산을 적용할 축(axis)을 지정합니다.
2. **Reduce 연산:** 지정된 축(axis)을 따라 Reduce 연산(예: `sum()`, `mean()`)을 수행합니다.
3. **결과:** 지정된 축(axis)이 사라지고(차원 축소), 나머지 축으로 구성된 결과가 반환됩니다.

**예시:**

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

# axis=0 (열)을 따라 sum() 연산 수행
result = a.sum(axis=0)  # [5 7 9]

# 1. axis=0 (열) 지정
# 2. 각 열([1, 4], [2, 5], [3, 6])에 대해 sum() 연산 수행
# 3. axis=0 (열)이 사라지고, axis=1 (행)만 남은 결과 [5 7 9] 반환
```

### 16.5 스트라이드(Stride)와 메모리 레이아웃
- **스트라이드**: 배열에서 다음 요소로 이동하기 위해 건너뛰어야 하는 바이트 수
- **메모리 효율성**: 스트라이드를 통해 메모리 접근 효율성 최적화 가능
```python
# 스트라이드 응용: 전치 연산
transposed = arr_2d.T
print("원본 배열 스트라이드:", arr_2d.strides)  # (32, 8)
print("전치 배열 스트라이드:", transposed.strides)  # (8, 32)

# 뷰(view)와 복사(copy)
view_arr = arr_2d.view()  # 데이터 공유 (스트라이드만 변경)
copy_arr = arr_2d.copy()  # 데이터 복제

# 스트라이드를 이용한 효율적인 연산
# 예: 큰 행렬에서 하위 행렬 추출
big_matrix = np.arange(10000).reshape(100, 100)
sub_matrix = big_matrix[10:20, 30:40]  # 뷰(view) - 데이터 복사 없음
```

### 16.6 실전 응용: 박스플롯을 통한 데이터 분석
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 요일별 tip 비율 (tip/total_bill) 비교
tips['tip_pct'] = tips['tip'] / tips['total_bill']

plt.figure(figsize=(12, 6))
sns.boxplot(x='day', y='tip_pct', hue='time', data=tips)
plt.title('요일 및 시간대별 팁 비율 분포')
plt.ylabel('팁 비율 (tip/total_bill)')
plt.ylim(0, 0.4)  # y축 범위 제한
plt.show()

# 결과 해석:
# - 박스의 중앙선: 중앙값 (50% 분위수)
# - 박스 범위: 25% ~ 75% 분위수 (IQR)
# - 수염(Whisker): IQR의 1.5배 이내의 데이터
# - 점: 이상치 (outlier)
```

## 17. 데이터 처리의 어려움과 실전 접근 방식

### 17.1 실제 데이터 문제 해결

- **현실 데이터는 "messy"**: 지저분하며 오류를 포함할 가능성이 높습니다.
- **데이터 오류 감지 전략:**
  - **소규모 데이터**: 수작업(manpower)으로 오류 검토가 가능합니다.
  - **빅데이터**: 수작업은 불가능하며, 자동화된 방법이 필요합니다. 수작업하면 미친사람이 아닌가 싶을 정도로 비효율적입니다.
- **데이터 오류 방지 방법:**
  - **Validation**: 데이터 입력 단계에서 규칙(rule)을 적용하여 잘못된 데이터 유입을 차단합니다.
  - **다양한 기술 활용**: 데이터 처리 파이프라인 전반에 걸쳐 오류를 줄이는 기술을 적용합니다.
- **데이터 오류 수정 과정:**
  - 시스템 운영 및 유지보수 과정에서 지속적인 오류 수정이 필요합니다.

### 17.2 NumPy와 Pandas의 강점 활용

- **NumPy:**
  - **기능**: 다차원 배열(ndarray)을 효율적으로 처리하는 Python 라이브러리입니다.
  - **구현**: C로 작성되어 빠른 속도를 제공합니다.
  - **Structured Array**: 서로 다른 데이터 타입을 갖는 열(column)로 구성된 테이블 형태의 데이터를 표현하는 NumPy의 기능입니다.

- **Pandas:**
  - **기능**: 데이터 분석 및 조작을 위한 고수준의 자료구조(DataFrame 등)와 함수를 제공하는 Python 라이브러리입니다.
  - **기반**: NumPy를 기반으로 구축되어 NumPy의 기능을 활용할 수 있습니다.
  - **DataFrame**:
    - 2차원 테이블 형태의 데이터 구조입니다.
    - 각 열(column)은 동일한 데이터 타입(homogeneous)을 가집니다.
    - 행(row)은 서로 다른 데이터 타입(heterogeneous)을 가질 수 있습니다 (튜플 형태로 묶으면 homogeneous하게 처리 가능).

### 17.3 데이터 조망 및 분석 접근법

- **데이터 조망(Overview):**
  - **목적**: 데이터의 전체적인 구조, 패턴, 이상치 등을 빠르게 파악합니다.
  - **방법**: `tips.head()`, `tips.tail()`, `tips.sample()` 등을 사용합니다.
  - **통계 도구 및 그래프 활용**: 데이터 조망을 위한 보조 도구로 활용합니다.

- **pandas 유용한 메서드:**
  - **`.info()`**: DataFrame의 전체적인 정보 요약(행/열 개수, 결측치 여부, 데이터 타입, 메모리 사용량 등)
  - **`.describe()`**: DataFrame의 기술 통계량(descriptive statistics) 요약
    - `include` 옵션: 특정 데이터 타입의 통계량만 확인 가능
    - `include='category'`: 범주형 데이터의 빈도(frequency) 관련 통계량
    - `include=['category', 'float64']`: 여러 데이터 타입 지정 가능

### 17.4 데이터 입출력 및 저장 고려사항

- **텍스트 파일:**
  - 모든 데이터를 문자열로 처리합니다.
  - 데이터 타입 정보 유지에 어려움이 있습니다.
- **Pickle:**
  - Python 객체를 직렬화(serialization)하여 파일로 저장합니다.
  - `pickle.dump()`: 객체 저장
  - `pickle.load()`: 객체 불러오기
  - Python에서만 사용 가능하여 호환성에 제한이 있습니다.
- **범용적인 데이터 포맷:**
  - **Parquet**: 컬럼 기반(columnar) 저장 방식으로, 효율적인 압축 및 데이터 타입을 지원합니다.
  - **JSON**: 텍스트 기반이며, key-value 쌍으로 구성된 데이터 형식입니다.
  - 데이터 타입, 유효성 검사 정보 등을 포함하여 데이터 손실을 최소화합니다.
- **데이터베이스(DB)**: 데이터 저장 및 관리의 안정적인 방법을 제공합니다.

### 17.5 Box Plot(상자 그림)을 통한 데이터 이해

Box Plot은 데이터 분포와 이상치를 시각적으로 파악하는 강력한 도구입니다:

- **중앙선**: 중앙값(median)을 나타냅니다.
- **박스**: Q1(25% 분위수)에서 Q3(75% 분위수)까지의 범위(IQR)를 표시합니다.
- **수염(whisker)**: IQR의 1.5배 범위 내에 있는 가장 작은/큰 값까지 확장됩니다.
- **점**: 수염 밖에 있는 데이터로, 이상치(outlier)로 간주됩니다.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
tips = sns.load_dataset('tips')

# 박스플롯 생성
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('요일별 계산서 금액 분포')
plt.show()
```