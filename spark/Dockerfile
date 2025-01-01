FROM apache/spark-py:latest

# 작업 디렉토리 설정
WORKDIR /opt/spark/work-dir

# 볼륨 설정
VOLUME ["/opt/spark/work-dir/data"]  # 데이터 저장용 볼륨
VOLUME ["/opt/spark/work-dir/logs"]  # 로그 저장용 볼륨

# 컨테이너 시작 시 pyspark 실행
CMD ["/opt/spark/bin/pyspark"]
