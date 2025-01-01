# PySpark 학습 프로젝트

## 문서 구조
1. [환경 설정](docs/1_setup.md)
   - 도커 환경 설정
   - 프로젝트 구조

2. [기본 사용법](docs/2_basic_usage.md)
   - SparkSession 생성
   - 데이터 읽기/쓰기
   - 데이터 필터링

3. [DataFrame 출력 방법](docs/3_dataframe_output.md)
   - 다양한 출력 메서드
   - 데이터 확인 방법

## 주요 특징
- SparkContext (sc): 저수준 API, 클러스터 연결 및 기본 작업
- SparkSession (spark): 고수준 API, DataFrame/SQL 작업
- 볼륨 마운트로 로컬 개발 환경과 연동
- `localhost:4040`에서 Spark UI 접근 가능
