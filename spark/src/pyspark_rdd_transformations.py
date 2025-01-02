from pyspark import SparkContext

# SparkContext 생성
sc = SparkContext.getOrCreate()

# 1. map, flatMap 예제
data = ["col_1,col_2,col_3", "col_4, col_5, col_6"]
rdd1 = sc.parallelize(data)

# take 액션: RDD의 처음 n개 요소 반환
rdd1.take(10)

# map 변환: RDD의 각 요소에 함수 적용 (일대일 매핑)
rdd2 = rdd1.map(lambda v: v.upper())
rdd2.take(10)

# map 변환: 각 요소를 콤마로 분리 (결과는 리스트의 RDD)
rdd3 = rdd1.map(lambda v: v.split(','))
rdd3.take(10)

# flatMap 변환: 각 요소에 함수 적용 후 평탄화 (일대다 매핑 후 통합)
rdd4 = rdd1.flatMap(lambda v: v.split(','))
rdd4.take(10)

# 2. subtract 예제
# subtract 변환: 첫 번째 RDD에서 두 번째 RDD의 요소를 제외한 요소 반환
r1 = sc.parallelize([1, 2, 3, 4, 5])
r2 = sc.parallelize([4, 5, 6, 7, 8])
rdd5 = r1.subtract(r2)
rdd5.take(10)

# 3. 파티션 관련 예제
# getNumPartitions 액션: RDD의 파티션 수 반환
rdd = sc.range(0, 100, 1, 5)
rdd.getNumPartitions()

# glom 변환: 각 파티션 내의 요소들을 리스트로 묶음
rdd = sc.range(0, 100, 1, 5)
rdd1 = rdd.glom().map(lambda arr: len(arr))
rdd1.take(10)

# glom 변환 결과 확인
rdd.glom().take(10)

# 4. Key-Value RDD 변환 예제
data = ["a", "b", "c", "b", "b", "d"]
rdd1 = sc.parallelize(data)

# map 변환: 각 요소를 (key, value) 형태의 튜플로 변환
rdd2 = rdd1.map(lambda v: (v, 1))
rdd2.take(10)

# mapValues 변환: Key-Value RDD의 Value에 함수 적용
rdd1.mapValues(lambda v: v + 1).take(10) # 이 코드는 에러 발생, rdd1은 key-value 형태가 아님

# mapValues 변환: Key-Value RDD의 Value에 함수 적용
rdd3 = rdd2.mapValues(lambda v: v + 1)
rdd3.take(10)

# flatMapValues 변환: Key-Value RDD의 각 Value에 함수 적용 후 평탄화
data = [("a", "1, 2, 3"), ("b", "4, 5, 6")]
rdd = sc.parallelize(data)
rdd4 = rdd.flatMapValues(lambda v: v.split(","))
rdd4.take(10)

# reduceByKey 변환: 동일한 키를 가진 값들을 주어진 함수로 합침
rdd6 = rdd2.reduceByKey(lambda a, b: a + b)
rdd6.take(10)

# groupByKey 변환: 동일한 키를 가진 값들을 그룹화
rdd7 = rdd6.groupByKey()
rdd7.map(lambda x : (x[0], list(x[1]))).collect()

# cogroup 변환: 두 개의 Key-Value RDD에 대해 동일한 키를 가진 값들을 함께 그룹화
kv1 = [("k1", "v1"), ("k2", "v2"), ("k3", "v3")]
rdd8 = sc.parallelize(kv1)

kv2 = [("k1", "v4"), ("k2", "v5"), ("k3", "v6")]
rdd9 = sc.parallelize(kv2)

rdd10 = rdd8.cogroup(rdd9)
[(x, tuple(map(list, y))) for x, y in sorted(list(rdd10.collect()))]

rdd10.take(10)