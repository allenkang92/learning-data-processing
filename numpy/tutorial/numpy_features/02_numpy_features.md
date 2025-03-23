# Saving and Sharing Your NumPy Arrays
# NumPy 배열 저장 및 공유하기

## What you'll learn
## 배울 내용

You'll save your NumPy arrays as zipped files and human-readable comma-delimited files i.e. *.csv. You will also learn to load both of these file types back into NumPy workspaces.
NumPy 배열을 압축 파일과 사람이 읽을 수 있는 쉼표로 구분된 파일(*.csv) 형태로 저장하는 방법을 배웁니다. 또한 이러한 파일 유형을 NumPy 작업 공간으로 다시 불러오는 방법도 배웁니다.

## What you'll do
## 할 일

- You'll learn two ways of saving and reading files–as compressed and as text files–that will serve most of your storage needs in NumPy.
- NumPy에서 대부분의 저장 요구를 충족시키는 압축 파일과 텍스트 파일로 저장하고 읽는 두 가지 방법을 배웁니다.
- You'll create two 1D arrays and one 2D array
- 1차원 배열 두 개와 2차원 배열 하나를 생성합니다.
- You'll save these arrays to files
- 이러한 배열을 파일로 저장합니다.
- You'll remove variables from your workspace
- 작업 공간에서 변수를 제거합니다.
- You'll load the variables from your saved file
- 저장된 파일에서 변수를 불러옵니다.
- You'll compare zipped binary files to human-readable delimited files
- 압축된 이진 파일과 사람이 읽을 수 있는 구분된 파일을 비교합니다.
- You'll finish with the skills of saving, loading, and sharing NumPy arrays
- NumPy 배열을 저장, 불러오기 및 공유하는 기술을 습득하게 됩니다.

## What you'll need
## 필요한 것

- NumPy
- NumPy
- read-write access to your working directory
- 작업 디렉토리에 대한 읽기-쓰기 권한

Load the necessary functions using the following command.
다음 명령을 사용하여 필요한 함수를 불러옵니다.

```python
import numpy as np
```

In this tutorial, you will use the following Python, IPython magic, and NumPy functions:
이 튜토리얼에서는 다음과 같은 Python, IPython 매직 및 NumPy 함수를 사용합니다:

- `np.arange`
- `np.savez`
- `del`
- `whos`
- `np.load`
- `np.block`
- `np.newaxis`
- `np.savetxt`
- `np.loadtxt`

## Create your arrays
## 배열 생성하기

Now that you have imported the NumPy library, you can make a couple of arrays; let's start with two 1D arrays, x and y, where y = x**2. You will assign x to the integers from 0 to 9 using np.arange.
이제 NumPy 라이브러리를 가져왔으니 몇 가지 배열을 만들 수 있습니다. y = x**2인 두 개의 1차원 배열 x와 y부터 시작해 보겠습니다. np.arange를 사용하여 x에 0부터 9까지의 정수를 할당할 것입니다.

```python
x = np.arange(10)
y = x ** 2
print(x)
print(y)
```

