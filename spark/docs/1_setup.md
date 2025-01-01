# PySpark 환경 설정

## 1. 도커 환경 설정

### 1.1 도커 이미지 가져오기
```bash
docker pull apache/spark-py
```

### 1.2 도커 실행
```bash
docker run -it --rm -p 4040:4040 \
    -v $(pwd)/spark/src:/opt/spark/work-dir/src \
    -v $(pwd)/spark/data:/opt/spark/work-dir/data \
    -v $(pwd)/spark/logs:/opt/spark/work-dir/logs \
    my-pyspark
```

- `-it`: 대화형 터미널
- `--rm`: 컨테이너 종료 시 자동 삭제
- `-p 4040:4040`: Spark UI 포트
- `-v`: 볼륨 마운트 (로컬 디렉토리와 컨테이너 연결)

## 2. 프로젝트 구조
```
spark/
├── data/         # 데이터 파일 저장
├── logs/         # 로그 파일 저장
└── src/          # 소스 코드 저장
```
