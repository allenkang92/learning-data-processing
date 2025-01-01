FROM apache/flink:latest

# Python 3 및 pip 설치
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /opt/flink/work-dir

# Python 패키지 설치
RUN pip3 install apache-flink

# 환경 변수 설정
ENV PYTHONPATH="${PYTHONPATH}:/opt/flink/work-dir"

# 볼륨 설정
VOLUME ["/opt/flink/work-dir/data"]  # 데이터 저장용 볼륨
VOLUME ["/opt/flink/work-dir/logs"]  # 로그 저장용 볼륨

# 기본 명령어 설정 (Flink Python shell)
CMD ["python3"]
