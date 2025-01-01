# DataFrame 출력 방법

## 1. show() 메서드
```python
# show() 메서드 - 가장 기본적인 데이터프레임 출력 방법
# 테이블 형태로 예쁘게 출력해줌
df.show()                    # 기본값으로 20행까지 출력
df.show(5)                   # 상위 5행만 출력
df.show(n=5, truncate=True)  # 긴 문자열을 잘라서 출력
df.show(5, truncate=False)   # 5행을 전체 길이로 출력
```

## 2. head() 메서드
```python
# head() 메서드 - 파이썬 리스트로 변환해서 보기
# Row 객체의 리스트 형태로 반환
df.head()    # 첫 번째 행만 반환
df.head(3)   # 처음 3개 행을 리스트로 반환
```

## 3. take() 메서드
```python
# take() 메서드 - 지정된 수만큼 행 가져오기
# head()와 비슷하지만 내부 동작 방식이 다름
df.take(5)   # 처음 5개 행을 Row 객체 리스트로 반환
```

## 4. toPandas() 메서드
```python
# toPandas() 메서드 - Pandas DataFrame으로 변환
# 주의: 대용량 데이터의 경우 메모리 부족 발생 가능
pandas_df = df.toPandas()    # 전체 데이터를 Pandas DataFrame으로 변환
print(pandas_df)             # Pandas 스타일로 출력
```

## 5. printSchema() 메서드
```python
# printSchema() 메서드 - 데이터프레임의 구조 확인
# 컬럼명, 데이터 타입, null 허용 여부 등을 트리 구조로 표시
df.printSchema()
```

## 6. describe() 메서드
```python
# describe() 메서드 - 기본 통계 정보
# 숫자형 컬럼의 count, mean, stddev, min, max 값을 보여줌
df.describe().show()
```

## 7. count() 메서드
```python
# count() 메서드 - 전체 행 수 반환
total_rows = df.count()
print(f"총 행 수: {total_rows}")
```
