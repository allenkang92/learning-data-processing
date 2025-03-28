# 100 numpy exercises

This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow
and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
and new users but also to provide a set of exercises for those who teach.

> 이것은 numpy 메일링 리스트, 스택 오버플로우 및 numpy 문서에서 수집된 연습 문제들의 모음입니다. 이 모음의 목표는 기존 사용자와 새로운 사용자 모두에게 빠른 참조를 제공할 뿐만 아니라 가르치는 사람들을 위한 연습 문제 세트를 제공하는 것입니다.

If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>.
File automatically generated. See the documentation to update questions/answers/hints programmatically.

> 오류를 발견하거나 일부 문제를 더 잘 해결할 방법이 있다고 생각되면 <https://github.com/rougier/numpy-100>에서 이슈를 자유롭게 열어주세요.
> 파일은 자동으로 생성되었습니다. 질문/답변/힌트를 프로그래밍 방식으로 업데이트하려면 문서를 참조하세요.

#### 1. Import the numpy package under the name `np` (★☆☆)
#### 1. numpy 패키지를 `np`라는 이름으로 가져오기 (★☆☆)

#### 2. Print the numpy version and the configuration (★☆☆)
#### 2. numpy 버전과 구성 출력하기 (★☆☆)

#### 3. Create a null vector of size 10 (★☆☆)
#### 3. 크기가 10인 영벡터 생성하기 (★☆☆)

#### 4. How to find the memory size of any array (★☆☆)
#### 4. 배열의 메모리 크기를 찾는 방법 (★☆☆)

#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
#### 5. 명령줄에서 numpy add 함수의 문서를 얻는 방법은? (★☆☆)

#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
#### 6. 크기가 10인 영벡터를 만들되 다섯 번째 값만 1로 설정하기 (★☆☆)

#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)
#### 7. 10부터 49까지의 값을 가지는 벡터 생성하기 (★☆☆)

#### 8. Reverse a vector (first element becomes last) (★☆☆)
#### 8. 벡터 뒤집기 (첫 번째 요소가 마지막이 됨) (★☆☆)

#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
#### 9. 0부터 8까지의 값을 가지는 3x3 행렬 생성하기 (★☆☆)

#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
#### 10. [1,2,0,0,4,0]에서 0이 아닌 요소들의 인덱스 찾기 (★☆☆)

#### 11. Create a 3x3 identity matrix (★☆☆)
#### 11. 3x3 항등 행렬 생성하기 (★☆☆)

#### 12. Create a 3x3x3 array with random values (★☆☆)
#### 12. 3x3x3 크기의 랜덤 값을 가진 배열 생성하기 (★☆☆)

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
#### 13. 10x10 크기의 랜덤 값을 가진 배열을 생성하고 최소값과 최대값 찾기 (★☆☆)

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
#### 14. 크기가 30인 랜덤 벡터를 생성하고 평균값 찾기 (★☆☆)

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
#### 15. 테두리는 1이고 내부는 0인 2차원 배열 생성하기 (★☆☆)

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
#### 16. 기존 배열 주위에 테두리(0으로 채워진)를 추가하는 방법은? (★☆☆)

#### 17. What is the result of the following expression? (★☆☆)
#### 17. 다음 표현식의 결과는 무엇인가? (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
#### 18. 대각선 바로 아래에 1,2,3,4 값을 가지는 5x5 행렬 생성하기 (★☆☆)

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
#### 19. 8x8 행렬을 생성하고 체커보드 패턴으로 채우기 (★☆☆)

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
#### 20. (6,7,8) 형태의 배열에서 100번째 요소의 인덱스 (x,y,z)는 무엇인가? (★☆☆)

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
#### 21. tile 함수를 사용하여 체커보드 8x8 행렬 생성하기 (★☆☆)

#### 22. Normalize a 5x5 random matrix (★☆☆)
#### 22. 5x5 랜덤 행렬 정규화하기 (★☆☆)

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
#### 23. 색상을 네 개의 부호 없는 바이트(RGBA)로 설명하는 사용자 정의 dtype 생성하기 (★☆☆)

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
#### 24. 5x3 행렬과 3x2 행렬 곱하기 (실제 행렬 곱) (★☆☆)

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
#### 25. 1D 배열에서 3과 8 사이의 모든 요소를 제자리에서 부정하기 (★☆☆)

#### 26. What is the output of the following script? (★☆☆)
#### 26. 다음 스크립트의 출력은 무엇인가? (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
#### 27. 정수 벡터 Z가 있을 때, 다음 표현식 중 어떤 것이 합법적인가? (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. What are the result of the following expressions? (★☆☆)
#### 28. 다음 표현식의 결과는 무엇인가? (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

#### 29. How to round away from zero a float array ? (★☆☆)
#### 29. 부동 소수점 배열을 0에서 멀어지도록 반올림하는 방법은? (★☆☆)

#### 30. How to find common values between two arrays? (★☆☆)
#### 30. 두 배열 간의 공통 값을 찾는 방법은? (★☆☆)

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
#### 31. 모든 numpy 경고를 무시하는 방법은 (권장하지 않음)? (★☆☆)

#### 32. Is the following expressions true? (★☆☆)
#### 32. 다음 표현식이 참인가? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
#### 33. 어제, 오늘, 내일의 날짜를 얻는 방법은? (★☆☆)

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
#### 34. 2016년 7월에 해당하는 모든 날짜를 얻는 방법은? (★★☆)

#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
#### 35. ((A+B)*(-A/2))를 제자리에서 계산하는 방법은 (복사 없이)? (★★☆)

#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
#### 36. 4가지 다른 방법을 사용하여 양수 랜덤 배열의 정수 부분 추출하기 (★★☆)

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
#### 37. 행 값이 0부터 4까지인 5x5 행렬 생성하기 (★★☆)

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
#### 38. 10개의 정수를 생성하는 제너레이터 함수를 고려하여 배열을 구축하기 (★☆☆)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
#### 39. 0과 1 사이의 값(둘 다 제외)을 가진 크기가 10인 벡터 생성하기 (★★☆)

#### 40. Create a random vector of size 10 and sort it (★★☆)
#### 40. 크기가 10인 랜덤 벡터를 생성하고 정렬하기 (★★☆)

#### 41. How to sum a small array faster than np.sum? (★★☆)
#### 41. np.sum보다 작은 배열을 더 빠르게 합산하는 방법은? (★★☆)

#### 42. Consider two random array A and B, check if they are equal (★★☆)
#### 42. 두 개의 랜덤 배열 A와 B를 고려하여, 그들이 동일한지 확인하기 (★★☆)

#### 43. Make an array immutable (read-only) (★★☆)
#### 43. 배열을 불변(읽기 전용)으로 만들기 (★★☆)

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
#### 44. 직교 좌표를 나타내는 랜덤 10x2 행렬을 고려하여, 극좌표로 변환하기 (★★☆)

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
#### 45. 크기가 10인 랜덤 벡터를 생성하고 최대값을 0으로 대체하기 (★★☆)

#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)
#### 46. [0,1]x[0,1] 영역을 커버하는 `x`와 `y` 좌표를 가진 구조화된 배열 생성하기 (★★☆)

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)
#### 47. 두 배열 X와 Y가 주어졌을 때, 코시 행렬 C (Cij =1/(xi - yj)) 구성하기 (★★☆)

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
#### 48. 각 numpy 스칼라 타입에 대해 표현 가능한 최소값과 최대값 출력하기 (★★☆)

#### 49. How to print all the values of an array? (★★☆)
#### 49. 배열의 모든 값을 출력하는 방법은? (★★☆)

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
#### 50. 벡터에서 (주어진 스칼라에) 가장 가까운 값을 찾는 방법은? (★★☆)

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
#### 51. 위치 (x,y)와 색상 (r,g,b)을 나타내는 구조화된 배열 생성하기 (★★☆)

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
#### 52. 좌표를 나타내는 형태가 (100,2)인 랜덤 벡터를 고려하여, 점과 점 사이의 거리 찾기 (★★☆)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
#### 53. 부동 소수점(32비트) 배열을 제자리에서 정수(32비트)로 변환하는 방법은?

#### 54. How to read the following file? (★★☆)
#### 54. 다음 파일을 읽는 방법은? (★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
#### 55. numpy 배열에서 enumerate와 동등한 것은 무엇인가? (★★☆)

#### 56. Generate a generic 2D Gaussian-like array (★★☆)
#### 56. 일반적인 2D 가우시안 형태의 배열 생성하기 (★★☆)

#### 57. How to randomly place p elements in a 2D array? (★★☆)
#### 57. 2D 배열에 p개의 요소를 무작위로 배치하는 방법은? (★★☆)

#### 58. Subtract the mean of each row of a matrix (★★☆)
#### 58. 행렬의 각 행에서 평균 빼기 (★★☆)

#### 59. How to sort an array by the nth column? (★★☆)
#### 59. n번째 열을 기준으로 배열을 정렬하는 방법은? (★★☆)

#### 60. How to tell if a given 2D array has null columns? (★★☆)
#### 60. 주어진 2D 배열에 null 열이 있는지 확인하는 방법은? (★★☆)

#### 61. Find the nearest value from a given value in an array (★★☆)
#### 61. 배열에서 주어진 값에 가장 가까운 값 찾기 (★★☆)

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
#### 62. 형태가 (1,3)과 (3,1)인 두 배열을 고려할 때, 반복자를 사용하여 그들의 합을 계산하는 방법은? (★★☆)

#### 63. Create an array class that has a name attribute (★★☆)
#### 63. 이름 속성을 가진 배열 클래스 만들기 (★★☆)

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
#### 64. 주어진 벡터를 고려할 때, 두 번째 벡터에 의해 인덱싱된 각 요소에 1을 더하는 방법은 (반복된 인덱스에 주의)? (★★★)

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
#### 65. 인덱스 리스트(I)를 기반으로 벡터(X)의 요소를 배열(F)에 누적하는 방법은? (★★★)

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
#### 66. (dtype=ubyte)의 (w,h,3) 이미지를 고려할 때, 고유한 색상의 수를 계산하기 (★★☆)

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
#### 67. 4차원 배열을 고려할 때, 마지막 두 축에 대한 합을 한 번에 얻는 방법은? (★★★)

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)
#### 68. 1차원 벡터 D를 고려할 때, 부분집합 인덱스를 설명하는 동일한 크기의 벡터 S를 사용하여 D의 부분집합 평균을 계산하는 방법은? (★★★)

#### 69. How to get the diagonal of a dot product? (★★★)
#### 69. 점곱의 대각선을 얻는 방법은? (★★★)

#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
#### 70. 벡터 [1, 2, 3, 4, 5]를 고려할 때, 각 값 사이에 3개의 연속된 0이 끼워진 새 벡터를 만드는 방법은? (★★★)

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
#### 71. 차원이 (5,5,3)인 배열을 차원이 (5,5)인 배열로 곱하는 방법은? (★★★)

#### 72. How to swap two rows of an array? (★★★)
#### 72. 배열의 두 행을 교환하는 방법은? (★★★)

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)
#### 73. 10개의 삼각형(공유 정점 포함)을 설명하는 10개의 세쌍을 고려할 때, 모든 삼각형을 구성하는 고유한 선분 집합 찾기 (★★★)

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
#### 74. bincount에 해당하는 정렬된 배열 C가 주어졌을 때, np.bincount(A) == C가 되는 배열 A를 생성하는 방법은? (★★★)

#### 75. How to compute averages using a sliding window over an array? (★★★)
#### 75. 배열 위에서 슬라이딩 윈도우를 사용하여 평균을 계산하는 방법은? (★★★)

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
#### 76. 1차원 배열 Z를 고려할 때, 첫 번째 행이 (Z[0],Z[1],Z[2])이고 각 후속 행이 1씩 시프트된 2차원 배열 구축하기 (마지막 행은 (Z[-3],Z[-2],Z[-1])이어야 함) (★★★)

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
#### 77. 불리언을 부정하거나 부동 소수점의 부호를 제자리에서 변경하는 방법은? (★★★)

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
#### 78. 선(2d)을 설명하는 2개의 점 집합 P0,P1과 한 점 p를 고려할 때, p에서 각 선 i (P0[i],P1[i])까지의 거리를 계산하는 방법은? (★★★)

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
#### 79. 선(2d)을 설명하는 2개의 점 집합 P0,P1과 점 집합 P를 고려할 때, 각 점 j (P[j])에서 각 선 i (P0[i],P1[i])까지의 거리를 계산하는 방법은? (★★★)

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)
#### 80. 임의의 배열을 고려할 때, 고정된 모양을 가지고 주어진 요소를 중심으로 하는 부분을 추출하는 함수 작성하기 (필요할 때 `fill` 값으로 패딩) (★★★)

#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
#### 81. 배열 Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]를 고려할 때, 배열 R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]을 생성하는 방법은? (★★★)

#### 82. Compute a matrix rank (★★★)
#### 82. 행렬 랭크 계산하기 (★★★)

#### 83. How to find the most frequent value in an array?
#### 83. 배열에서 가장 빈번한 값을 찾는 방법은?

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
#### 84. 랜덤 10x10 행렬에서 모든 연속된 3x3 블록 추출하기 (★★★)

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
#### 85. Z[i,j] == Z[j,i]가 되는 2D 배열 서브클래스 생성하기 (★★★)

#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
#### 86. 형태가 (n,n)인 p개의 행렬 집합과 형태가 (n,1)인 p개의 벡터 집합을 고려할 때, p개의 행렬곱의 합을 한 번에 계산하는 방법은? (결과는 형태가 (n,1)) (★★★)

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
#### 87. 16x16 배열을 고려할 때, 블록 합(블록 크기는 4x4)을 얻는 방법은? (★★★)

#### 88. How to implement the Game of Life using numpy arrays? (★★★)
#### 88. numpy 배열을 사용하여 생명 게임을 구현하는 방법은? (★★★)

#### 89. How to get the n largest values of an array (★★★)
#### 89. 배열에서 n개의 가장 큰 값을 얻는 방법은? (★★★)

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
#### 90. 임의의 수의 벡터가 주어졌을 때, 데카르트 곱(모든 항목의 모든 조합) 구축하기 (★★★)

#### 91. How to create a record array from a regular array? (★★★)
#### 91. 일반 배열에서 레코드 배열을 생성하는 방법은? (★★★)

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
#### 92. 큰 벡터 Z를 고려할 때, 3가지 다른 방법을 사용하여 Z의 3승 계산하기 (★★★)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
#### 93. 형태가 (8,3)과 (2,2)인 두 배열 A와 B를 고려할 때, B의 요소 순서에 상관없이 B의 각 행의 요소를 포함하는 A의 행을 찾는 방법은? (★★★)

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
#### 94. 10x3 행렬을 고려할 때, 불균등한 값을 가진 행 추출하기(예: [2,2,3]) (★★★)

#### 95. Convert a vector of ints into a matrix binary representation (★★★)
#### 95. 정수 벡터를 행렬 이진 표현으로 변환하기 (★★★)

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)
#### 96. 2차원 배열이 주어졌을 때, 고유한 행을 추출하는 방법은? (★★★)

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
#### 97. 두 벡터 A와 B를 고려할 때, inner, outer, sum, 그리고 mul 함수의 einsum 등가물 작성하기 (★★★)

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
#### 98. 두 벡터(X,Y)로 설명된 경로를 고려할 때, 등간격 샘플을 사용하여 샘플링하는 방법은 (★★★)?

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
#### 99. 정수 n과 2D 배열 X가 주어졌을 때, n 차수를 가진 다항 분포에서 추출된 것으로 해석될 수 있는 행을 X에서 선택하기, 즉, 정수만 포함하고 합이 n인 행. (★★★)

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
#### 100. 1D 배열 X의 평균에 대한 부트스트랩된 95% 신뢰 구간 계산하기(즉, 배열의 요소를 N번 대체하여 리샘플링하고, 각 샘플의 평균을 계산한 다음, 평균에 대한 백분위수 계산하기). (★★★)
