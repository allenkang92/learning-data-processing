# 1. 도커 이미지 빌드
cd /Users/ddang/learning-data-processing/learning-data-processing/spark
docker build -t my-pyspark .

# 2. 로컬 볼륨 디렉토리 생성
mkdir -p ./data ./logs

# 3. 도커 컨테이너 실행
docker run -it \
  --name pyspark-container \
  -v "$(pwd)/data:/opt/spark/work-dir/data" \
  -v "$(pwd)/logs:/opt/spark/work-dir/logs" \
  my-pyspark



주요 설명:

-t my-pyspark: 이미지 이름 지정
-it: 대화형 터미널로 실행
-v: 로컬 디렉토리와 컨테이너 볼륨 연결
$(pwd): 현재 작업 디렉토리 경로
컨테이너가 실행되면 PySpark 셸에 접속됩니다.