# RDD 사용법

## 1. RDD 생성

### 1.1 `parallelize()` 메서드
```python
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

data = [1, 2, 3, 4, 5]
rdd = sc.parallelize(data)
```
Use code with caution.

### 1.2 `range()` 메서드
```python
rdd_range = sc.range(0, 10, 2)  # 0부터 10 미만까지 2씩 증가
```
Use code with caution.

## 2. RDD 액션

### 2.1 collect()
```python
# RDD의 모든 요소를 드라이버 프로그램으로 가져오기 (작은 데이터셋에 적합)
collected_data = rdd.collect()
print(collected_data)
```

### 2.2 take(n)
```python
# RDD의 처음 n개 요소 가져오기
first_three = rdd.take(3)
print(first_three)
```

### 2.3 first()
```python
# RDD의 첫 번째 요소 가져오기
first_element = rdd.first()
print(first_element)
```

### 2.4 count()
```python
# RDD 내의 요소 개수 세기
count = rdd.count()
print(f"RDD 요소 개수: {count}")
```

### 2.5 distinct()
```python
# RDD에서 중복된 요소 제거
distinct_rdd = rdd.distinct()
print(distinct_rdd.collect())
```

### 2.6 countApproxDistinct(relativeSD=0.05)
```python
# RDD 내 고유 요소의 대략적인 개수 세기
approx_distinct_count = rdd.countApproxDistinct()
print(f"RDD 대략적인 고유 요소 개수: {approx_distinct_count}")
```

### 2.7 takeOrdered(n, key=None, reverse=False)
```python
# RDD에서 정렬된 처음 n개 요소 가져오기
ordered_rdd = rdd.takeOrdered(3)
print(ordered_rdd)

ordered_rdd_desc = rdd.takeOrdered(3, reverse=True)
print(ordered_rdd_desc)
```

### 2.8 top(n, key=None)
```python
# RDD에서 가장 큰 n개 요소 가져오기 (내림차순)
top_elements = rdd.top(2)
print(top_elements)
```

### 2.9 reduce(func)
```python
# RDD 요소들을 결합하여 단일 값 생성
sum_of_elements = rdd.reduce(lambda a, b: a + b)
print(f"RDD 요소의 합: {sum_of_elements}")
```

### 2.10 foreach(func)
```python
# RDD의 각 요소에 대해 함수 실행 (결과 드라이버에 반환 X)
def print_element(x):
    print(f"Processing element: {x}")

rdd.foreach(print_element)
```

### 2.11 saveAsTextFile(path)
```python
# RDD 내용을 텍스트 파일로 저장
rdd.saveAsTextFile("/opt/spark/work-dir/rdd_output")
```

### 2.12 toLocalIterator()
```python
# RDD 요소를 순차적으로 처리하기 위한 iterator 생성
for element in rdd.toLocalIterator():
    print(f"Iterating over element: {element}")
```

## 3. RDD 정보 확인

### 3.1 getNumPartitions()
```python
# RDD의 파티션 수 확인
num_partitions = rdd.getNumPartitions()
print(f"RDD 파티션 수: {num_partitions}")
```

### 3.2 glom()
```python
# 각 파티션 내의 요소들을 리스트로 묶기
grouped_partitions = rdd.glom().collect()
print(grouped_partitions)
```