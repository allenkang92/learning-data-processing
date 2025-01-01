# PySpark 기본 사용법

## 1. SparkSession 생성
```python
# SparkSession 생성
# - Spark 애플리케이션의 시작점
# - 모든 Spark 기능의 진입점
spark = SparkSession.builder.getOrCreate()
```

## 2. 데이터 읽기

### 2.1 기본 CSV 읽기
```python
# 기본적인 CSV 파일 읽기
# - 헤더를 첫 번째 행 데이터로 취급
# - 모든 컬럼을 문자열(string) 타입으로 읽음
df = spark.read.csv("/opt/spark/work-dir/data/book_data.csv")
df.printSchema()
```

### 2.2 헤더 포함 CSV 읽기
```python
# 헤더 옵션을 추가한 CSV 파일 읽기
# - header=true: 첫 번째 행을 컬럼명으로 사용
df = spark.read.option("header", "true").csv("/opt/spark/work-dir/data/book_data.csv")
df.printSchema()
```

## 3. 데이터 확인
```python
# DataFrame 내용 출력
# - show(n, truncate): 상위 n개 행 출력
# - truncate=False: 긴 문자열을 자르지 않고 전체 출력
df.show(3, False)
```

## 4. 데이터 필터링
```python
# SQL의 WHERE 절과 동일
# - F.col()로 컬럼 참조
# - 문자열 비교 시 따옴표 필수
import pyspark.sql.functions as F
df = df.where(F.col('AREA_NM') == '강원도')
```

## 5. 데이터 저장
```python
# DataFrame을 CSV 파일로 저장
# - coalesce(1): 파티션을 1개로 합쳐서 단일 파일로 저장
# - mode('overwrite'): 기존 파일 덮어쓰기
# - 다른 mode 옵션: 'append', 'error', 'ignore'
df.coalesce(1).write.mode('overwrite').csv("/opt/spark/work-dir/data/gangwon_data.csv")
```
