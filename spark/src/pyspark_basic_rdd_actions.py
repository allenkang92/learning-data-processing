from pyspark import SparkContext

# SparkContext 생성 (SparkSession이 이미 생성되어 있다면 sc를 가져올 수 있습니다.)
sc = SparkContext.getOrCreate()

# 1. range 액션
# - 지정된 범위의 숫자로 RDD 생성
# - (start, end, step, numPartitions)
rdd = sc.range(0, 1000, 1, 10)
# RDD 내의 요소 개수 반환
rdd.count()

# RDD의 처음 n개 요소 반환 (순서 보장 X)
rdd.take(10)

# RDD의 모든 요소를 드라이버 프로그램으로 가져옴 (주의: 대용량 데이터셋에서는 사용하지 않도록 권장)
rdd.collect()

# 2. parallelize 액션
# - 로컬 컬렉션으로부터 RDD 생성
data = ["uno", "ni", "than"]
rdd1 = sc.parallelize(data)

# RDD의 처음 n개 요소 반환 (순서 보장 X)
rdd1.take(3)

# RDD의 처음 n개 요소를 정렬하여 반환 (오름차순)
rdd1.takeOrdered(3)

# RDD에서 가장 큰 요소 n개 반환 (내림차순)
rdd1.top(1)

# RDD의 첫 번째 요소 반환
rdd1.first()

# 3. distinct 액션
data = ["hana", "dule", "set", "net", "hana"]
rdd1 = sc.parallelize(data)

# RDD에서 중복된 요소를 제거하고 반환 (순서 보장 X)
rdd1.distinct().take(10)

# RDD 내의 고유 요소의 대략적인 개수 반환
rdd1.countApproxDistinct()

# 4. toLocalIterator 액션
# - RDD의 모든 요소를 순차적으로 처리하기 위한 iterator 반환
for v in rdd1.toLocalIterator():
    print(v)