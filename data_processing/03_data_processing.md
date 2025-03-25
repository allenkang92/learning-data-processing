### 1. 정형 데이터 분석과 Tidy Data.

- **정형 데이터 (Structured Data):**
    - 행(row)과 열(column)로 구성된 테이블 형태의 데이터.
    - 관계형 데이터베이스 (Relational Database), CSV, Excel 파일 등이 대표적인 정형 데이터 형식.
    - 데이터 분석에서 가장 많이 다루는 데이터 유형.
- **정형 데이터 분석 도구:**
    - **R:** 통계 분석 및 데이터 시각화에 특화된 프로그래밍 언어. 여전히 정형 데이터 분석 분야에서 많이 사용됨.
    - **Python (Pandas):** 데이터 분석 및 조작을 위한 라이브러리. R에서 유용한 기능을 Python으로 포팅(porting)하는 경우가 많음.
- **Messy Data vs. Tidy Data:**
    - **Messy Data (지저분한 데이터):** 분석하기 어렵거나 비효율적인 형태로 구성된 데이터.
        - 열 이름이 값이거나, 여러 변수가 하나의 열에 저장된 경우 등.
    - **Tidy Data (깔끔한 데이터):** 분석하기 좋은 형태로 구성된 데이터.
        - **3가지 조건:**
            1. 각 변수(variable)는 열(column)을 구성.
            2. 각 관측값(observation)은 행(row)을 구성.
            3. 각 관측 단위(observational unit) 유형은 테이블(table)을 구성.
        - **장점:**
            - 데이터 조작, 모델링, 시각화가 용이.
            - Pandas, dplyr(R) 등 데이터 분석 도구와 호환성이 좋음.
            - 일관된 데이터 구조로 다양한 분석 기법 적용 가능.
        - **Tidy Data = Narrow Data = Long Form:** 같은 의미로 사용되는 용어.
        - **Wide Form:** Tidy Data와 반대되는 개념 (분석에 적합하지 않음).

### 2. 관계형 데이터베이스 (RDB)와 Tidy Data.

- **관계형 데이터베이스 (Relational Database, RDB):**
    - 데이터를 테이블(table) 형태로 저장하고, 테이블 간의 관계(relationship)를 정의하여 데이터를 효율적으로 관리하는 시스템.
    - **장점:**
        - 데이터 중복 최소화.
        - 데이터 무결성(integrity) 보장.
        - 데이터 일관성 유지.
        - 데이터 검색, 수정, 삭제 등 데이터 조작 용이.
        - **CRUD:** Create(생성), Read(읽기), Update(수정), Delete(삭제) 연산 지원.
        - **Aggregation (집계):** 데이터 요약 (평균, 합계, 개수 등) 용이.
        - **Validation (유효성 검사):** 데이터 입력 시 규칙(rule)을 적용하여 잘못된 데이터 유입 방지.
- **Tidy Data와 RDB:**
    - Tidy Data는 관계형 데이터베이스의 테이블 구조와 유사.
    - Tidy Data는 RDB에서 데이터를 가져와 분석할 때 적합한 형태.
    - 데이터 구조 자체도 "더티 데이터(dirty data)"가 될 수 있음 (Tidy Data가 아닌 경우).

### 3. Pandas를 이용한 데이터 구조 변환

Pandas는 Wide Form 데이터를 Long Form (Tidy Data)으로, 또는 그 반대로 변환하는 기능을 제공합니다.

- **`pd.read_csv()`:** CSV 파일 읽기 (탭으로 구분된 파일 예제).
- **Wide Form → Long Form:**
    - **`data.melt()`:**
        - Wide Form 데이터를 Long Form (Tidy Data)으로 변환하는 함수 (Unpivot).
        - **`id_vars`:** 식별 변수(identifier variable)로 유지할 열 (변환하지 않을 열)을 지정 (e.g., `'religion'`).
        - **`value_vars`**: 값으로 사용할 열들을 지정. 생략시 `id_vars`로 지정되지 않은 모든 열을 값으로 변환
        - **`var_name`:** 값으로 변환된 열의 이름을 지정 (기본값: 'variable').
        - **`value_name`:** 값 열의 이름을 지정 (기본값: 'value').
        - **`inplace=True`:** 원본 DataFrame을 직접 변경 (mutable). `inplace=True`를 사용하지 않으면 변환된 결과를 새로운 변수에 할당해야 함 (재할당).
    - **예시:**
        
        ```python
        import pandas as pd
        
        # Pew Research Center 종교별 소득 데이터 (Wide Form)
        data = pd.read_csv('pew.txt', sep='\\t')
        print(data)
        
        # Long Form으로 변환
        data_molten = data.melt(id_vars='religion', var_name='income', value_name='count')
        print(data_molten)
        
        ```
        
- **Long Form → Wide Form:**
    - **`data_molten.pivot_table()`:**
        - Long Form 데이터를 Wide Form으로 변환 (Pivot).
        - **`index`:** 행 인덱스로 사용할 열 지정.
        - **`columns`:** 열 이름으로 사용할 열 지정.
        - **`values`:** 값으로 사용할 열 지정.
        - **`aggfunc`:** 집계 함수 지정 (기본값: 평균).
    - **`data_molten.pivot()`**: 인덱스, 컬럼, 값만 사용하여 형태를 바꿈.
    - **예시:**
        
        ```python
        # Long Form 데이터를 Wide Form으로 변환 (pivot_table)
        data_wide = data_molten.pivot_table(index='religion', columns='income', values='count')
        print(data_wide)
        
        ```
        
- **`melt()` 활용:**
    - `id_vars`에 여러 개의 열을 리스트 형태로 지정 가능.
    - `melt()`는 데이터 구조를 Tidy Data로 변환하여 데이터 분석 및 관리를 용이하게 함.
    - Long Form 데이터는 한눈에 보기 어려울 수 있지만, 데이터 분석 도구 (e.g., Pandas, Seaborn)를 사용하면 효율적으로 처리 가능.

### 4. 데이터 정제 (Data Cleaning)

- **`data.melt(data.columns[:7])`:**
    - `data.columns[:7]`: DataFrame의 처음 7개 열을 선택 (식별 변수로 유지).
    - 나머지 열들 (주차별 순위)을 값으로 변환.
- **`data_molten.info()`:**
    - 변환된 DataFrame의 정보 확인 (결측치, 데이터 타입 등).
    - `Non-Null Count`: 결측치가 아닌 값의 개수.
- **`data_molten.dropna()`:**
    - 결측치 (missing value)가 있는 행 제거.
- **문자열 처리:**
    - **`data_molten.variable.str.extract('(\\d)')`:**
        - `str`: Pandas Series의 문자열 메서드 접근자.
        - `extract()`: 정규 표현식 (regular expression)을 사용하여 문자열 추출.
        - `(\\d)`: 정규 표현식에서 숫자를 의미하는 패턴 (괄호로 묶어 캡처).
    - **`data_molten.variable.str[:-9]`:**
        - `str` 접근자를 사용한 문자열 슬라이싱.
        - `[:-9]`: 문자열의 마지막 9개 문자를 제외한 나머지 부분 선택.
    - **`map()`:** 함수를 적용하여 값을 변환.
- **데이터 타입 변환:**
    - **`data_molten.value.astype('int64')`:**
        - `astype()`: 데이터 타입을 변환.
        - `'int64'`: 64비트 정수 타입.
- **데이터 정제의 중요성:**
    - 잘못된 데이터 (e.g., 결측치, 이상치, 부정확한 값)를 수정하고, 데이터 타입을 분석에 적합하게 변환하는 과정.
    - 데이터 분석 결과의 신뢰성을 높이는 데 필수적.

### 5. 데이터 구조와 차원의 저주

- **컬럼(Column)의 개수:**
    - 컬럼이 너무 많으면:
        - **차원의 저주 (Curse of Dimensionality):** 차원(컬럼)이 증가할수록 데이터 분석 및 머신 러닝 모델의 성능이 저하되는 현상.
        - 모델 복잡도 증가, 과적합(overfitting) 위험 증가.
        - 계산 비용 증가 (CPU, 메모리 사용량 증가).
    - 컬럼이 너무 적으면:
        - 데이터 분석에 필요한 정보 부족.
        - 새로운 특성(feature)을 생성해야 할 수 있음 (특성 공학, feature engineering).
- **데이터 구조와 유지보수:**
    - 데이터 구조 (e.g., Wide Form, Long Form)는 데이터 유지보수 및 분석 효율성에 영향.
    - Tidy Data (Long Form)는 관계형 데이터베이스와 유사한 구조로, 데이터 관리 및 분석에 유리.

### 6. 분할 정복 (Divide and Conquer)

- **분할 정복 (Divide and Conquer):**
    - 복잡한 문제를 해결하기 위해 문제를 작은 부분 문제(subproblem)로 나누고, 각 부분 문제를 해결한 후, 결과를 결합하여 전체 문제의 해답을 얻는 전략.
    - **장점:**
        - 복잡한 문제를 단순화하여 해결 가능성 높임.
        - 병렬 처리(parallel processing)를 통해 계산 속도 향상 가능.
    - **단점:**
        - **분할의 오류:** 문제를 잘못 분할하면 올바른 해답을 얻을 수 없음.
        - **결합의 오류:** 부분 문제의 해답을 잘못 결합하면 올바른 해답을 얻을 수 없음.
- **함수형 프로그래밍 (Functional Programming):**
    - 분할 정복 전략을 구현하는 데 유용한 프로그래밍 패러다임.
    - **특징:**
        - 순수 함수 (pure function): 동일한 입력에 대해 항상 동일한 출력을 반환하고, 부작용(side effect)이 없는 함수.
        - 불변성 (immutability): 데이터가 변경되지 않음.
        - 고차 함수 (higher-order function): 함수를 인자로 받거나 함수를 반환하는 함수.
    - **장점:**
        - **형식적 증명 가능성 (Formal Provability):** 함수의 동작을 수학적으로 증명 가능 (이론적으로).
        - 코드의 간결성, 가독성, 재사용성 향상.
        - 병렬 처리 용이.
    - **현실적인 한계:** 형식적 증명 가능성은 이론적인 개념이며, 실제로는 모든 경우에 대해 증명이 완벽하게 이루어지기 어려울 수 있음.
- **데이터 분석에서의 분할 정복:**
    - **Split-Apply-Combine:**
        - 데이터를 그룹(group)으로 분할 (split).
        - 각 그룹에 함수(function)를 적용 (apply).
        - 결과를 다시 결합 (combine).
        - Pandas의 `groupby()`가 대표적인 Split-Apply-Combine 예시.

### 7. GroupBy를 활용한 데이터 분석

- **GroupBy:**
    - Pandas에서 데이터를 그룹별로 나누고, 각 그룹에 대해 독립적으로 연산을 수행하는 기능.
    - Split-Apply-Combine 전략 구현.
- **`tips.groupby('sex')`:**
    - 'sex' 열을 기준으로 데이터를 그룹화 (남성 그룹, 여성 그룹).
    - `groupby()`는 `DataFrameGroupBy` 객체를 반환.
- **`tips.groupby('sex')['tip'].mean()`:**
    - 'sex' 그룹별로 'tip' 열의 평균 계산 (Reduce 연산).
    - 1차원 Series 반환.
- **`tips.groupby('sex')[['tip']].mean()`:**
    - 'sex' 그룹별로 'tip' 열의 평균 계산.
    - `[['tip']]`: 팬시 인덱싱을 사용하여 2차원 DataFrame 반환.
- **시각화:**
    - `tips.groupby('sex')['tip'].mean().plot.bar()`: 그룹별 평균을 막대 그래프로 시각화.
- **`groupby()`의 다양한 활용:**
    - 여러 개의 열을 기준으로 그룹화: `tips.groupby(['sex', 'smoker'])`
    - 여러 개의 집계 함수 적용: `tips.groupby('sex')['tip'].agg(['mean', 'std', 'min', 'max'])`
    - `transform()`: 그룹별 연산을 수행하지만, 원래 DataFrame의 크기를 유지 (map과 유사).
        - `tips.groupby('sex')[['tip']].transform(lambda x: x + 1)`: 각 그룹 내에서 'tip' 열에 1을 더함.
    - `as_index=False`: 그룹화 기준 열을 인덱스로 사용하지 않음.
- **Tidy Data와 GroupBy:**
    - `groupby()`는 Tidy Data에서 특히 유용.
    - Tidy Data는 각 변수가 열을 구성하고, 각 관측값이 행을 구성하므로, `groupby()`를 사용하여 특정 변수(열)를 기준으로 데이터를 쉽게 그룹화하고 분석할 수 있음.

### 8. Pandas 확장 기능 (Side Table, Pandas Profiling)

- **Side Table (`sidetable`):**
    - Pandas DataFrame에 대한 빈도표 (frequency table), 누락값 요약 등 유용한 기능을 추가하는 라이브러리.
    - `!pip install -U sidetable`: sidetable 설치 (최신 버전으로 업그레이드).
    - `import sidetable`: sidetable import.
    - **몽키 패치 (Monkey Patch):** 런타임(runtime)에 동적으로 클래스나 모듈의 기능을 변경하는 기법.
        - `sidetable`은 Pandas DataFrame에 `stb` 접근자(accessor)를 추가하여 기능을 확장 (몽키 패치).
        - `tips.stb.freq(['day'])`: 'day' 열의 빈도표 계산.
        - `tips.stb.subtotal()`: 그룹별 부분합(subtotal) 계산.
        - `tips.stb.missing()`: 결측치 요약
    - **참고:** 몽키 패치는 코드의 예측 불가능성을 높일 수 있으므로 주의해서 사용해야 합니다.
- **Pandas Profiling (`ydata-profiling`):**
    - Pandas DataFrame에 대한 상세한 프로파일링 보고서 (profiling report)를 생성하는 라이브러리.
    - `!pip install -U ydata-profiling`: ydata-profiling 설치.
    - `import ydata_profiling`: ydata-profiling import.
    - `ydata_profiling.ProfileReport(tips)`: `tips` DataFrame에 대한 프로파일링 보고서 생성.
        - 변수별 요약 통계, 히스토그램, 상관 관계, 결측치 정보 등 제공.
        - HTML 형식으로 저장 가능.
    - `tsmode=True`: 시계열 데이터 분석 모드 활성화.
- **Pandas 확장 기능 활용:**
    - `sidetable`, `ydata-profiling`과 같은 라이브러리(스몰 프로젝트)를 활용하면 Pandas의 기능을 확장하여 데이터 분석을 더 효율적으로 수행할 수 있습니다.
    - 자신만의 데이터 분석 방법을 정립하는 데 도움이 됩니다.

- **Tidy Data:** 분석하기 좋은 데이터 형태 (각 변수는 열, 각 관측값은 행, 각 관측 단위는 테이블).
- **데이터 구조 변환:** Pandas의 `melt()`, `pivot_table()` 등을 사용하여 Wide Form과 Long Form 간 변환.
- **GroupBy:** 데이터를 그룹별로 나누어 분석 (Split-Apply-Combine).
- **데이터 정제:** 결측치 처리, 데이터 타입 변환, 문자열 처리 등.
- **Pandas 확장 기능:** `sidetable`, `ydata-profiling` 등을 활용하여 Pandas 기능 확장.