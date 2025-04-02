## 데이터 분석 및 시각화 (Pandas, Matplotlib, Seaborn)

### 1. 환경 설정 및 기본 라이브러리

- **IPython 매직 명령어 활용:**
    - `%whos`: 현재 네임스페이스의 변수 확인 (초기에는 비어있음)
    - `%matplotlib`: Matplotlib 백엔드 설정 (팝업 창 등)
    - `%lsmagic`: 사용 가능한 매직 명령어 목록 확인
    - `%load_ext version_information`: `version_information` 확장 기능 로드
    - `%version_information numpy, pandas, matplotlib`: 주요 라이브러리 버전 확인 (데이터 분석 시 버전 문제 확인에 유용)
- **라이브러리 임포트:**
    - `import pandas as pd`
    - `import matplotlib.pyplot as plt`
    - `import matplotlib as mp`
    - `import seaborn as sns`
    - `import missingno as mino`: 결측치 시각화 라이브러리
    - `import json`: JSON 데이터 처리
    - `import inspect`: 객체 정보 확인 (e.g., 함수 소스 코드)

### 2. Matplotlib 기초 및 한글 폰트

- Matplotlib은 기본적으로 한글 폰트를 제대로 지원하지 않아 별도 설정 필요 (`plt.rcParams`).
- `mp.font_manager.FontManager().ttflist`: 시스템에 설치된 폰트 목록 확인 가능.

### 3. 데이터 로딩 및 기본 정보 확인

- **데이터 로딩:**
    - `sns.load_dataset('dataset_name')`: Seaborn 제공 예제 데이터셋 로드 (e.g., 'mpg', 'tips', 'iris')
    - `pd.read_json()`: JSON 파일 로드 (구조화, 타입 설정 가능하나 숫자 타입은 하나)
- **기본 정보 확인:**
    - `df.info()`: 데이터프레임의 컬럼별 정보 (Non-Null Count, Dtype) 확인 -> 결측치 유무 파악 용이
    - `df.dtypes`: 컬럼별 데이터 타입 확인
    - `df.describe()`: 숫자형 데이터의 기술 통계량 확인 (문자열/객체 타입은 제외됨)

### 4. 결측치(Missing Data) 처리

- **중요성:** Pandas는 연산 시 결측치를 스킵하는 경우가 많지만, Scikit-learn 등 다른 라이브러리는 결측치 처리가 필수적.
- **결측치 확인:**
    - `df.isna()` / `df.isnull()`: 데이터프레임 전체에 대해 결측치 여부를 Boolean 값으로 반환 (Element-wise 연산)
    - `df.isnull().any()`: 컬럼별로 결측치가 하나라도 있는지 확인 (Boolean Series 반환)
    - `df.isnull().sum()`: 컬럼별 결측치 개수 합산 (True=1, False=0으로 계산)
    - `df.columns[df.isnull().any()]`: 결측치가 있는 컬럼 이름 확인
    - `df[df['column_name'].isna()]`: 특정 컬럼에 결측치가 있는 행 필터링 (Boolean Indexing)
- **결측치 시각화 (`missingno` 라이브러리):**
    - `mino.matrix(df, color=(r, g, b))`: 매트릭스 형태로 결측치 분포 시각화 (흰색 부분이 결측치)
    - `mino.heatmap(df)`: 컬럼 간 결측치 상관관계 시각화
- **처리 전략:**
    - **삭제 (Deletion):**
        - 로우 삭제: `df.dropna()` (결측치가 있는 행 제거) -> 데이터 손실 발생, 특히 로우 수가 적을 때 신중해야 함.
        - 컬럼 삭제: `df.drop()` (결측치 비율이 매우 높은 컬럼 제거) -> 정보 손실 발생, 컬럼은 분석의 중요한 '뉘앙스'를 담고 있음.
    - **대치 (Imputation):**
        - 가짜 데이터(평균, 중앙값, 최빈값 등)로 결측치를 채움: `df.fillna()`
        - 다른 컬럼 정보를 활용 가능하여 무작정 삭제하는 것보다 성능 향상에 도움이 되는 경우가 많음.
- **고려사항:**
    - 결측치 처리에는 절대적인 기준이 없으며, 데이터와 분석 목적에 따라 상황에 맞게 판단해야 함.
    - 결측치 비율(e.g., 10% 이상)에 따른 처리 방침, 재처리 여부, 경고 발생 시점 등을 정의할 필요가 있음.

### 5. 데이터 탐색 및 시각화 (EDA - Exploratory Data Analysis)

- **목표:** 데이터를 더 잘 이해하기 위해 통계와 그래프 활용.
- **집계:**
    - `df.groupby()`: 특정 컬럼 기준으로 그룹화하여 집계 연산 수행 (e.g., `.mean()`, `.sum()`)
    - `pd.pivot_table()`: 피벗 테이블 생성 (언급됨)
- **기본 시각화 (Pandas Plotting):**
    - `df.plot(kind='bar')` 또는 `df.plot.bar()`: 막대 그래프 (비교에 용이)
    - `stacked=True`: 누적 막대 그래프 (내부 구성 요소 비교)
    - `df.plot.barh()`: 수평 막대 그래프
    - `df.plot(kind='pie')`: 파이 차트 (비율 표시에 적합) -> 상황에 맞는 그래프 선택 중요
- **상관관계 분석:**
    - `df.corr(numeric_only=True)`: 숫자형 컬럼 간의 상관계수 계산 (Pandas 2.0 이후 `numeric_only=True` 필요)
    - `sns.heatmap(df.corr(), annot=True, cbar=False)`: 상관계수 행렬을 히트맵으로 시각화
        - `annot=True`: 상관계수 값 표시
        - `cbar=False`: 컬러바 숨김
- **Seaborn 활용:** (Pandas 기반, 더 미려하고 복잡한 그래프 용이)
    - `sns.set_theme()` / `sns.set_style()`: 그래프 테마/스타일 설정
    - `sns.violinplot()`: 바이올린 플롯 (분포 + 중앙값/사분위수), `hue` 옵션으로 그룹별 비교
    - `sns.boxenplot()`: Box plot보다 더 많은 정보 제공 (언급됨)
    - `sns.lmplot()`: 회귀선 포함 산점도, `hue`, `col`, `row` 옵션으로 다차원 분석 (FacetGrid 생성)
    - `sns.relplot()`: 관계형 플롯 (산점도, 선 그래프), `hue`, `col`, `row` 옵션으로 다차원 분석 (FacetGrid 생성)
    - `sns.pairplot(df, hue='category_column')`: 데이터프레임의 숫자형 컬럼 간 모든 쌍의 관계 시각화 (대각선: 분포, 나머지: 산점도), `hue`로 그룹 구분
    - `sns.jointplot()`: 두 변수 간의 관계(산점도)와 각 변수의 분포(히스토그램/커널밀도)를 함께 표시, `hue` 가능
    - `sns.swarmplot()` / `sns.catplot(kind='swarm')`: 카테고리별 분포 시각화 (점이 겹치지 않게), `hue` 가능
    - `sns.catplot()`: 카테고리 데이터 시각화 통합 인터페이스 (kind 옵션으로 다양한 플롯 지정: 'bar', 'box', 'violin', 'swarm', 'strip', 'point' 등)
        - `catplot` 사용 시 명시적으로 카테고리 데이터용 플롯임을 알 수 있음.
        - 개별 함수(`swarmplot`, `boxplot` 등)는 해당 함수가 카테고리용임을 미리 알고 있어야 함.
- **Matplotlib 객체지향 방식 (다중 그래프):**
    - `plt.subplot(rows, cols, index)`: Matplotlib의 상태 기반 방식 (MATLAB 스타일, 인덱스 1부터 시작, `221`처럼 붙여 쓸 수 있음)
    - `fig, ax = plt.subplots(rows, cols)`: 객체지향 방식 (Figure와 Axes 객체 반환, 권장)
        - `ax[row, col].plot()`: 특정 Axes에 그래프 그리기
        - `ax[row, col].set_title()`: 특정 Axes에 제목 설정 (순서 무관)
    - Seaborn의 `FacetGrid` (lmplot, relplot, catplot)도 내부적으로 Matplotlib의 다중 그래프 기능을 활용.

### 6. Pandas 데이터 타입: `category`

- **주요 데이터 타입:** `int64`, `float64`, `object` (주로 문자열), `category`
- **`category` 타입 특징:**
    - **메모리 효율성:** 내부적으로는 정수(codes)로 관리하고, 사용자에게는 문자열 레이블로 보여줌.
    - **데이터 검증:** 지정된 카테고리 외의 값은 기본적으로 입력 불가 (오류 발생).
    - **편의성:** `.cat` 접근자(accessor)를 통해 카테고리 관련 속성/메서드 사용 가능.
        - `series.cat.codes`: 내부 정수 코드 확인
        - `series.cat.categories`: 카테고리 목록 확인
        - `series.cat.add_categories()`: 새로운 카테고리 추가
    - **용도:** 명목형(Nominal) 데이터 관리에 유용. 주로 Pandas, Seaborn 등 Pandas 기반 라이브러리에서 활용됨. 정형 데이터 분석 외에는 잘 쓰이지 않음.
- **타입 변환:**
    - `series.astype('category')`: 다른 타입을 카테고리로 변환
    - `series.astype('object')`: 카테고리를 일반 객체(문자열) 타입으로 변환
- **Seaborn과의 연동:** `sns.load_dataset()`으로 불러온 데이터 중 일부(e.g., 'tips'의 'day', 'sex')는 자동으로 `category` 타입으로 변환되어 있음 (`inspect.getsource(sns.load_dataset)` 확인). `catplot` 등은 이 타입을 효과적으로 활용.



**핵심 요약:**

- 데이터 분석 전 환경 설정(라이브러리 임포트, 버전 확인)이 중요.
- 결측치는 반드시 확인하고, 데이터 특성과 분석 목적에 맞게 삭제 또는 대치 전략을 선택해야.. (`missingno` 시각화 유용).
- EDA는 데이터 이해의 핵심이며, 목적(비교, 비율, 분포, 관계, 상관관계)에 맞는 시각화 방법을 선택. (Pandas Plotting, Matplotlib, Seaborn 활용).
- Seaborn은 Pandas 데이터프레임을 기반으로 더 쉽고 미려한 시각화를 제공하며, 특히 `hue`, `col`, `row` 옵션을 통한 다차원 분석과 `catplot`을 통한 카테고리 데이터 처리가 강점.
- Matplotlib의 객체지향 방식(`plt.subplots`)은 여러 그래프를 체계적으로 관리하고 커스터마이징하는 데 유용.
- Pandas의 `category` 타입은 메모리 효율성과 데이터 검증 측면에서 이점이 있으며, Seaborn과 잘 연동.