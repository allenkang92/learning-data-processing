# 91. How to create a record array from a regular array?

import numpy as np

# 문제: 일반 배열에서 레코드 배열을 만드는 방법

# 레코드 배열은 구조화된 배열의 특별한 형태로, 필드에 이름으로 접근할 수 있음
# dtype을 사용하여 필드 이름과 데이터 타입 지정

# 방법 1: np.rec.array 사용
print("방법 1: np.rec.array 사용")
data = np.array([(1, 1.5, 'Hello'), (2, 2.0, 'World')], 
                dtype=[('id', int), ('value', float), ('name', 'U10')])
rec_array = np.rec.array(data)

print("레코드 배열 타입:", type(rec_array))
print("첫 번째 레코드:", rec_array[0])
print("모든 id:", rec_array.id)
print("모든 value:", rec_array.value)
print("모든 name:", rec_array.name)

# 방법 2: np.recarray로 변환
print("\n방법 2: np.recarray로 변환")
regular_array = np.array([(10, 3.14, 'NumPy'), (20, 2.718, 'SciPy')],
                          dtype=[('id', int), ('value', float), ('name', 'U10')])
rec_array2 = regular_array.view(np.recarray)

print("변환된 레코드 배열 타입:", type(rec_array2))
print("첫 번째 레코드:", rec_array2[0])
print("name 필드만 접근:", rec_array2.name)

# 방법 3: np.core.records 모듈 사용
print("\n방법 3: np.core.records 모듈 사용")
rec_array3 = np.core.records.fromarrays(
    [[1, 2, 3], [10.1, 20.2, 30.3], ['X', 'Y', 'Z']],
    names='id, value, name',
    formats='i4, f8, U1'
)
print("레코드 배열 형태:", rec_array3.shape)
print("레코드 배열 필드:", rec_array3.dtype.names)
print("value 필드만 접근:", rec_array3.value)

# 방법 4: 튜플 리스트에서 직접 생성
print("\n방법 4: 튜플 리스트에서 직접 생성")
data_list = [(100, 'Alpha', 1.1), (200, 'Beta', 2.2), (300, 'Gamma', 3.3)]
rec_array4 = np.core.records.fromrecords(
    data_list,
    names='id, name, value',
    formats='i4, U5, f8'
)
print("생성된 레코드 배열:", rec_array4)
print("name 필드만 접근:", rec_array4.name)

# 레코드 배열과 구조화된 배열의 차이점
print("\n레코드 배열과 구조화된 배열의 차이점:")
print("- 레코드 배열은 구조화된 배열의 특별한 형태로, '.' 표기법으로 필드에 접근 가능")
print("- 일반 구조화된 배열은 필드에 대괄호로 접근: array['field_name']")

# 구조화된 배열 예시
structured_array = np.array([(1, 'A'), (2, 'B')], dtype=[('num', int), ('char', 'U1')])
print("\n구조화된 배열에서 필드 접근:")
print("대괄호 표기법:", structured_array['num'])
# print("점 표기법 (오류 발생):", structured_array.num)  # 이 방식은 작동하지 않음

# 레코드 배열로 변환 후
record_view = structured_array.view(np.recarray)
print("\n레코드 배열로 변환 후 필드 접근:")
print("점 표기법:", record_view.num)
print("대괄호 표기법도 여전히 사용 가능:", record_view['char'])

# 응용: 표 형태의 데이터 처리
print("\n응용: 표 형태의 데이터 처리")
# 학생 정보를 저장하는 레코드 배열
students = np.recarray(3, dtype=[('name', 'U10'), ('age', int), ('grade', float)])
students.name = ['Alice', 'Bob', 'Charlie']
students.age = [20, 22, 21]
students.grade = [85.5, 92.0, 78.5]

print("학생 레코드 배열:")
for student in students:
    print(f"이름: {student.name}, 나이: {student.age}, 성적: {student.grade}")

# 조건에 따른 필터링 (예: 평균 성적 이상인 학생)
avg_grade = np.mean(students.grade)
print(f"\n평균 성적: {avg_grade:.2f}")
good_students = students[students.grade >= avg_grade]
print("평균 이상 성적의 학생:")
for student in good_students:
    print(f"이름: {student.name}, 성적: {student.grade}")



# 1. 레코드 배열(recarray)은 일반 구조화된 배열보다 접근 속도가 느릴 수 있음
# 2. 필드가 많고 큰 배열에서는 성능 차이가 중요할 수 있음
# 3. 최신 NumPy 버전에서는 dtype의 fields 속성 사용을 권장함
# 4. 데이터 분석 작업에는 pandas DataFrame이 더 적합할 수 있음