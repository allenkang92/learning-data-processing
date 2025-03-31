## Pandas 및 시각화 

### 1. Pandas의 역할 및 특징 (데이터 분석 관점)

- **데이터 저장 이상:** Pandas는 단순히 데이터를 저장하는 도구가 아님. 다양한 포맷으로 데이터를 변환하고 처리하는 데 강점.
- **데이터프레임 활용:** 데이터를 DataFrame 구조로 변환하여 Pandas의 강력한 기능(필터링, 그룹화, 조작 등)을 활용.
- **부분집합(Subset) 처리:** 전체 데이터가 아닌 특정 부분집합을 효율적으로 다룰 수 있음.
- **시각화 기능 내장:** Pandas 자체적으로 기본적인 그래프(Matplotlib 기반)를 그릴 수 있음. EDA(탐색적 데이터 분석)에 중요.
- **데이터 조작 용이성:** 행(row)과 열(column)을 자유롭게 추가/삭제 가능.
- **파생 변수 생성:** 기존 열(column)을 기반으로 새로운 의미를 갖는 열(파생 변수)을 만드는 작업 용이 (`how to create new columns derived from existing columns`).
- **내재적/외재적 분석:** 데이터 자체 분석(내재적)뿐 아니라, 외부 데이터를 결합하여 분석(외재적)하는 데 유용.
- **시계열(Time Series) 분석:** 시계열 데이터 처리에 특화된 기능 제공 (개발자 전문 분야).
- **NumPy 연관 개념:**
    - Masked Array: 결측치(Missing Data) 처리와 관련.
    - Record Array: 동종(Homogeneous) 데이터 구조. Pandas DataFrame은 이종(Heterogeneous) 데이터 처리 가능.
- **One-liner 지향:** 데이터 분석가들은 간결한 한 줄 코드를 선호하는 경향이 있음. Pandas는 이를 지원. (단, 목적에 맞게 명확히 작성하는 것이 중요).

### 2. 데이터 시각화 (Data Visualization)

- **필요성:**
    - 정보를 **명확하고 효율적으로 전달**하는 핵심 수단.
    - 단일 통계 척도(평균, 분산 등)가 보여주지 못하는 데이터의 **전체적인 특성 및 세부 패턴**을 파악 가능.
    - 복잡한 데이터를 쉽게 이해하고 접근 가능하게 함.
    - 비교 분석, 인과 관계 이해 등 특정 분석 작업 촉진.
    - **그림 우월성 효과 (Picture Superiority Effect):** 인간은 텍스트보다 이미지를 더 잘 기억하고 처리함. 시각 정보는 뇌의 정보 처리 대역폭이 가장 높음.
- **목적:**
    - **설명/제시 (Presentation/Explanation):** 이미 알려진 사실을 효과적으로 전달 (결과 소통).
    - **탐색 (Exploration):** 가설을 가지고 데이터를 탐색하며 새로운 가설 생성.
    - **확인 (Confirmation):** 주어진 가설을 데이터를 통해 검증/반증.
- **시각화 절차:**
    1. **정보 조직화 (Data Organization):** 데이터를 분류, 배열하여 질서 부여 (주로 **Pandas** 활용). 사용자의 정보 '인지'에 관여.
    2. **정보 시각화 (Data Visualization):** 조직화된 데이터를 시각적 요소(그래프 형태, 색상, 크기 등)로 변환하여 효율적 전달 (주로 **Matplotlib, Seaborn** 활용). 사용자의 정보 '지각'에 관여.
    3. **상호작용 (Interaction):** 사용자가 시각화 결과와 상호작용하며 정보를 탐색할 수 있도록 설계 (주로 **Plotly, Bokeh** 등 웹 기반 라이브러리 활용). 사용자 경험(UX) 디자인 측면.

### 3. Python 시각화 라이브러리 개요

- **렌더링/문법 기반 분류:**
    - **Python 네이티브:** Matplotlib (기반), Pandas, Seaborn, ggplot (R 스타일 문법).
    - **JavaScript 기반 (웹, 인터랙티브):** Bokeh, Plotly.
    - **Vega 기반 (선언적, 인터랙티브):** Altair, Vincent, Pdvega.
    - **SVG 기반 (웹 벡터 그래픽):** Pygal.
    - **OpenGL 기반 (고성능 2D/3D):** Vispy.
    - **QT 기반 (데스크톱 GUI):** PyQwt, PyQtGraph.
- **주요 Python 라이브러리 특징:**
    - **Matplotlib:** Python 시각화의 **사실상 표준(De Facto)**. 대부분 라이브러리의 기반. MATLAB 스타일 문법 계승. 유연하지만 기본 스타일은 다소 오래됨.
    - **Pandas:** 데이터 조작이 주 목적이나, Matplotlib 기반의 편리한 시각화 기능 내장. DataFrame/Series 객체에서 직접 `.plot()` 호출 가능.
    - **Seaborn:** Matplotlib 기반. **통계적 시각화**에 특화. 더 미려한 기본 스타일과 색상 팔레트 제공. Pandas DataFrame과 잘 통합됨.
    - **ggplot:** R의 유명한 `ggplot2` 패키지 문법(Grammar of Graphics)을 Python으로 구현. Matplotlib 기반. 계층적으로 그래프 요소(축, 점, 선 등)를 쌓아나가는 방식.

### 4. Matplotlib 기초 및 사용법

- **핵심:** Python 시각화 생태계의 기반. Pandas, Seaborn 등은 Matplotlib의 Wrapper(래퍼).
- **구성 요소 (계층 구조):**
    - **Canvas:** 실제 그림이 그려지는 객체 (사용자에게는 거의 보이지 않음).
    - **Figure:** 전체 그림 영역(캔버스 위의 도화지). 0개 이상의 Axes를 포함. 크기(`figsize`), 해상도(`dpi`), 배경색(`facecolor`) 등 설정 가능.
    - **Axes:** Figure 내의 개별 그래프(subplot) 영역. 데이터 공간(x축, y축 범위 등) 정의. 2~3개의 Axis 객체를 가짐.
    - **Axis:** 축 자체. 눈금(Tick) 위치(Locator) 및 레이블(Formatter) 관리.
    - **Artist:** Figure 위에 보이는 모든 요소 (Figure, Axes, Axis 객체 포함. Text, Line2D, Patch 등).
- **사용 방식:**
    1. **State Machine (MATLAB 방식, `pyplot` 인터페이스):** `plt.plot()`, `plt.figure()`, `plt.axes()`, `plt.grid()` 등 함수 호출 시, 암묵적으로 현재 활성화된 Figure/Axes에 명령 적용. 코드가 간결하나 복잡한 그래프 제어 어려움.
    2. **Object-Oriented (객체 지향 방식):** Figure, Axes 객체를 명시적으로 생성하고 해당 객체의 메서드를 호출하여 그래프 구성. 더 명확하고 제어 용이. (이번 강의 노트에는 명시적 예시 부족)
    3. **혼용 방식:** 두 방식을 섞어 사용.
- **주요 함수/개념:**
    - `plt.figure()`: 새 Figure 생성 또는 기존 Figure 선택. `figsize`, `facecolor` 등 설정. `;` 사용 시 Jupyter에서 불필요한 출력 숨김.
    - `plt.axes()`: Figure 내에 Axes 생성 또는 선택. 위치/크기 지정 가능 (`plt.axes((left, bottom, width, height))`). `polar=True`로 극좌표계 지정 가능. `label` 지정 가능.
    - `plt.plot()`: 선 그래프 등 기본적인 플롯 생성. 내부적으로 Axes를 찾아 그림.
    - `plt.show()`: 생성된 그래프를 화면에 표시 (스크립트 환경 등에서 필요).
    - `plt.grid(True)`: 그리드(격자) 표시.
    - `plt.style.available`: 사용 가능한 스타일 목록 확인.
    - `plt.style.use('style_name')`: 전역적으로 그래프 스타일 변경 (톤앤매너 유지 중요).
    - `with plt.xkcd()`: 특정 코드 블록 내에서만 일시적으로 스타일 변경 (컨텍스트 매니저).
    - `plt.ylim([min, max])`: Y축 범위 수동 조절 (차이 강조 테크닉).
- **커스터마이징:** Pandas, Seaborn 그래프도 Matplotlib 기반이므로, Matplotlib의 Figure, Axes 객체에 접근하여 세부적인 커스터마이징 가능.

### 5. Seaborn 기초 및 사용법

- **특징:** Matplotlib 기반, 통계 시각화 특화, 미려한 기본 스타일, Pandas DataFrame과 높은 호환성.
- **사용법:**
    - `import seaborn as sns`
    - `sns.load_dataset('dataset_name')`: 예제 데이터셋 로드 (e.g., 'tips').
    - **그래프 함수:** `sns.boxplot()`, `sns.barplot()`, `sns.histplot()` 등.
        - 주요 인자: `x` (x축 변수명), `y` (y축 변수명), `data` (사용할 DataFrame), `hue` (색상으로 구분할 추가 범주형 변수명).
- **Pandas Plotting과의 차이 (Boxplot 예시):**
    - Pandas: `tips.boxplot()` (수치형 데이터가 컬럼으로). 단축 표현.
    - Seaborn: `sns.boxplot(x='col_x', y='col_y', data=tips)` (x, y, data 명시). 더 명확하고 유연함.
- **`hue` 파라미터:** 하나의 그래프 내에서 추가적인 범주형 변수에 따라 데이터를 나누어 시각화 (데이터 압축 해제/분할 효과).
- **Matplotlib 연동:** Seaborn 그래프도 Matplotlib Figure/Axes 위에 그려지므로, `plt.figure(figsize=...)` 등으로 Figure 속성 조절 가능, `plt.style` 적용 가능.

### 6. 데이터 분석과 시각화의 자세

- **One-liner 함정:** 코드를 한 줄로 짜는 것보다 **어떤 그래프를 왜 그리는지** 고민하는 것이 더 중요.
- **목표:** 데이터를 명확하게 이해하고, 전달하고자 하는 메시지를 효과적으로 표현하는 것.
- **해석과 설득:** 데이터 분석 결과와 시각화는 보는 사람을 **납득**시키고 **설득**하는 과정. 정답은 없으며, 상대방이 이해할 수 있도록 만드는 것이 중요.
- **창의성:** 기존 그래프 형식에 얽매이지 않고, 데이터를 가장 잘 나타낼 수 있는 역발상이나 새로운 시각화 방법 모색 필요.
- **데이터 구조화:** 시각화하려는 내용에 맞게 Pandas `groupby()`, `pivot_table()`, `set_index()` 등으로 데이터를 미리 적절하게 구조화하는 것이 중요.
