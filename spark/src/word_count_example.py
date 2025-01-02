from pyspark import SparkContext

def create_spark_context():
    """SparkContext 생성"""
    return SparkContext("local", "Word Count Example")

def get_sample_text():
    """샘플 텍스트 데이터 반환"""
    return [
        "Apache Spark is a unified analytics engine for large-scale data processing",
        "Spark provides high-level APIs in Java Scala Python and R",
        "Spark offers much faster performance than traditional Hadoop MapReduce",
        "PySpark is the Python API for Apache Spark",
        "Spark includes libraries for SQL streaming machine learning and graph processing",
        "Data engineers use Spark for ETL processing and data transformation",
        "Spark can process data in memory which makes it faster than disk-based processing",
        "RDD is the fundamental data structure of Apache Spark",
        "Spark SQL provides a programming abstraction called DataFrames",
        "Machine learning library in Spark is called MLlib"
    ]

def word_count(sc, text):
    """단어 수를 세는 메인 로직"""
    # 텍스트를 RDD로 변환
    text_rdd = sc.parallelize(text)

    # 단어 단위로 분리
    words = text_rdd.flatMap(lambda line: line.split())

    # 각 단어별 카운트
    word_counts = words.map(lambda word: (word, 1))
    word_counts = word_counts.reduceByKey(lambda a, b: a + b)

    return word_counts

def main():
    # SparkContext 초기화
    sc = create_spark_context()

    try:
        # 샘플 텍스트 가져오기
        sample_text = get_sample_text()

        # 단어 수 계산
        result = word_count(sc, sample_text)

        # 결과 출력
        print("\n=== 단어 빈도수 상위 20개 ===")
        for word, count in result.takeOrdered(20, key=lambda x: -x[1]):
            print(f"{word}: {count}")

    finally:
        sc.stop()

if __name__ == "__main__":
    main()