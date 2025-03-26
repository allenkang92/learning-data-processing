# NumPy Quickstart
# NumPy 빠른 시작 가이드

## Prerequisites
## 사전 요구사항

You'll need to know a bit of Python. For a refresher, see the Python tutorial.
To work the examples, you'll need matplotlib installed in addition to NumPy.

> 파이썬에 대한 기본 지식이 필요합니다. 복습이 필요하면 파이썬 튜토리얼을 참조하세요.
> 예제를 실행하려면 NumPy 외에도 matplotlib이 설치되어 있어야 합니다.

## Learner profile
## 학습자 프로필

This is a quick overview of arrays in NumPy. It demonstrates how n-dimensional (n-d) arrays are represented and can be manipulated. In particular, if you don't know how to apply common functions to n-dimensional arrays (without using for-loops), or if you want to understand axis and shape properties for n-dimensional arrays, this article might be of help.

> 이 문서는 NumPy 배열에 대한 간략한 개요입니다. n차원(n-d) 배열이 어떻게 표현되고 조작될 수 있는지 보여줍니다. 특히 for 루프를 사용하지 않고 n차원 배열에 일반적인 함수를 적용하는 방법을 모르거나, n차원 배열의 축(axis)과 형태(shape) 속성을 이해하고 싶다면 이 글이 도움이 될 것입니다.

## Learning Objectives
## 학습 목표

After reading, you should be able to:

* Understand the difference between one-, two- and n-dimensional arrays in NumPy;
* Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;
* Understand axis and shape properties for n-dimensional arrays.

> 이 글을 읽은 후에는 다음을 할 수 있어야 합니다:
> * NumPy의 1차원, 2차원 및 n차원 배열 간의 차이점 이해
> * for 루프를 사용하지 않고 n차원 배열에 선형 대수 연산을 적용하는 방법 이해
> * n차원 배열의 축(axis)과 형태(shape) 속성 이해

## The basics
## 기초

NumPy's main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers. In NumPy dimensions are called axes.

For example, the array for the coordinates of a point in 3D space, [1, 2, 1], has one axis. That axis has 3 elements in it, so we say it has a length of 3. In the example pictured below, the array has 2 axes. The first axis has a length of 2, the second axis has a length of 3.

> 예를 들어, 3D 공간의 한 점의 좌표를 나타내는 배열 [1, 2, 1]은 하나의 축을 가집니다. 이 축은 3개의 요소를 가지고 있으므로 길이가 3이라고 말합니다. 아래 그림의 예제에서, 배열은 2개의 축을 가집니다. 첫 번째 축의 길이는 2이고, 두 번째 축의 길이는 3입니다.

```python
[[1., 0., 0.],
 [0., 1., 2.]]
```

NumPy's array class is called ndarray. It is also known by the alias array. Note that numpy.array is not the same as the Standard Python Library class array.array, which only handles one-dimensional arrays and offers less functionality. The more important attributes of an ndarray object are:

> NumPy의 배열 클래스는 ndarray라고 불립니다. 또한 array라는 별칭으로도 알려져 있습니다. numpy.array는 표준 파이썬 라이브러리의 array.array 클래스와 다르다는 점에 유의하세요. array.array는 1차원 배열만 다루며 기능이 더 제한적입니다. ndarray 객체의 중요한 속성들은 다음과 같습니다:

ndarray.ndim
the number of axes (dimensions) of the array.

> ndarray.ndim
> 배열의 축(차원) 개수.

ndarray.shape
the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension. For a matrix with n rows and m columns, shape will be (n,m). The length of the shape tuple is therefore the number of axes, ndim.

> ndarray.shape
> 배열의 차원. 각 차원에서의 배열 크기를 나타내는 정수형 튜플입니다. n행 m열을 가진 행렬의 경우, shape는 (n,m)이 됩니다. 따라서 shape 튜플의 길이는 축의 개수인 ndim과 같습니다.

ndarray.size
the total number of elements of the array. This is equal to the product of the elements of shape.

> ndarray.size
> 배열의 전체 요소 개수. 이는 shape의 모든 요소들을 곱한 값과 같습니다.

ndarray.dtype
an object describing the type of the elements in the array. One can create or specify dtype's using standard Python types. Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.

> ndarray.dtype
> 배열 내 요소들의 타입을 설명하는 객체. 표준 파이썬 타입을 사용하여 dtype을 생성하거나 지정할 수 있습니다. 또한 NumPy는 자체적인 타입도 제공합니다. numpy.int32, numpy.int16, numpy.float64 등이 그 예입니다.

ndarray.itemsize
the size in bytes of each element of the array. For example, an array of elements of type float64 has itemsize 8 (=64/8), while one of type complex32 has itemsize 4 (=32/8). It is equivalent to ndarray.dtype.itemsize.

> ndarray.itemsize
> 배열 각 요소의 바이트 단위 크기. 예를 들어, float64 타입 요소를 가진 배열은 itemsize가 8(=64/8)이고, complex32 타입은 itemsize가 4(=32/8)입니다. 이는 ndarray.dtype.itemsize와 동일합니다.

ndarray.data
the buffer containing the actual elements of the array. Normally, we won't need to use this attribute because we will access the elements in an array using indexing facilities.

> ndarray.data
> 배열의 실제 요소들이 저장된 버퍼. 일반적으로 배열의 요소에 접근할 때는 인덱싱 기능을 사용하므로 이 속성을 직접 사용할 필요는 없습니다.

## An example
## 예시

```python
import numpy as np
a = np.arange(15).reshape(3, 5)
a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
a.shape
(3, 5)
a.ndim
2
a.dtype.name
'int64'
a.itemsize
8
a.size
15
type(a)
<class 'numpy.ndarray'>
b = np.array([6, 7, 8])
b
array([6, 7, 8])
type(b)
<class 'numpy.ndarray'>
```

## Array creation
## 배열 생성

There are several ways to create arrays.

> 배열을 생성하는 여러 가지 방법이 있습니다.

For example, you can create an array from a regular Python list or tuple using the array function. The type of the resulting array is deduced from the type of the elements in the sequences.

> 예를 들어, array 함수를 사용하여 일반 파이썬 리스트나 튜플로부터 배열을 생성할 수 있습니다. 결과 배열의 타입은 시퀀스 내 요소들의 타입으로부터 추론됩니다.

```python
import numpy as np
a = np.array([2, 3, 4])
a
array([2, 3, 4])
a.dtype
dtype('int64')
b = np.array([1.2, 3.5, 5.1])
b.dtype
dtype('float64')
```

A frequent error consists in calling array with multiple arguments, rather than providing a single sequence as an argument.

> 흔한 실수 중 하나는 단일 시퀀스를 인자로 제공하지 않고 array를 여러 인자로 호출하는 것입니다.

```python
a = np.array(1, 2, 3, 4)    # WRONG
Traceback (most recent call last):
  ...
TypeError: array() takes from 1 to 2 positional arguments but 4 were given
a = np.array([1, 2, 3, 4])  # RIGHT
```

array transforms sequences of sequences into two-dimensional arrays, sequences of sequences of sequences into three-dimensional arrays, and so on.

> array는 시퀀스의 시퀀스를 2차원 배열로, 시퀀스의 시퀀스의 시퀀스를 3차원 배열로 변환하는 식으로 작동합니다.

```python
b = np.array([(1.5, 2, 3), (4, 5, 6)])
b
array([[1.5, 2. , 3. ],
       [4. , 5. , 6. ]])
```

The type of the array can also be explicitly specified at creation time:

> 배열의 타입은 생성 시 명시적으로 지정할 수도 있습니다:

```python
c = np.array([[1, 2], [3, 4]], dtype=complex)
c
array([[1.+0.j, 2.+0.j],
       [3.+0.j, 4.+0.j]])
```

Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers several functions to create arrays with initial placeholder content. These minimize the necessity of growing arrays, an expensive operation.

> 종종 배열의 요소는 처음에 알 수 없지만 그 크기는 알고 있는 경우가 있습니다. 따라서 NumPy는 초기 자리 표시자 내용으로 배열을 생성하는 여러 함수를 제공합니다. 이는 배열 확장이라는 비용이 큰 작업의 필요성을 최소화합니다.

The function zeros creates an array full of zeros, the function ones creates an array full of ones, and the function empty creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is float64, but it can be specified via the key word argument dtype.

> zeros 함수는 0으로 가득 찬 배열을 생성하고, ones 함수는 1로 가득 찬 배열을 생성하며, empty 함수는 초기 내용이 랜덤하고 메모리 상태에 따라 달라지는 배열을 생성합니다. 기본적으로 생성된 배열의 dtype은 float64이지만, dtype 키워드 인자를 통해 지정할 수 있습니다.

```python
np.zeros((3, 4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
np.ones((2, 3, 4), dtype=np.int16)
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)
np.empty((2, 3)) 
array([[3.73603959e-262, 6.02658058e-154, 6.55490914e-260],  # may vary
       [5.30498948e-313, 3.14673309e-307, 1.00000000e+000]])
```

To create sequences of numbers, NumPy provides the arange function which is analogous to the Python built-in range, but returns an array.

> 숫자 시퀀스를 생성하기 위해 NumPy는 파이썬 내장 함수인 range와 유사한 arange 함수를 제공하지만, 이는 배열을 반환합니다.

```python
np.arange(10, 30, 5)
array([10, 15, 20, 25])
np.arange(0, 2, 0.3)  # it accepts float arguments
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
```

When arange is used with floating point arguments, it is generally not possible to predict the number of elements obtained, due to the finite floating point precision. For this reason, it is usually better to use the function linspace that receives as an argument the number of elements that we want, instead of the step:

> arange를 부동 소수점 인자와 함께 사용할 때, 유한한 부동 소수점 정밀도로 인해 얻게 될 요소의 개수를 예측하는 것이 일반적으로 불가능합니다. 이러한 이유로, 보통은 단계 대신 원하는 요소의 개수를 인자로 받는 linspace 함수를 사용하는 것이 좋습니다:

```python
from numpy import pi
np.linspace(0, 2, 9)                   # 9 numbers from 0 to 2
array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
x = np.linspace(0, 2 * pi, 100)        # useful to evaluate function at lots of points
f = np.sin(x)
```