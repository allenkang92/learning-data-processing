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

## Printing arrays
## 배열 출력하기

When you print an array, NumPy displays it in a similar way to nested lists, but with the following layout:

* the last axis is printed from left to right,
* the second-to-last is printed from top to bottom,
* the rest are also printed from top to bottom, with each slice separated from the next by an empty line.

> 배열을 출력할 때, NumPy는 중첩된 리스트와 비슷한 방식으로 표시하지만 다음과 같은 레이아웃을 따릅니다:
> 
> * 마지막 축은 왼쪽에서 오른쪽으로 출력됩니다.
> * 마지막에서 두 번째 축은 위에서 아래로 출력됩니다.
> * 나머지 축들도 위에서 아래로 출력되며, 각 슬라이스는 빈 줄로 구분됩니다.

One-dimensional arrays are then printed as rows, bidimensionals as matrices and tridimensionals as lists of matrices.

> 따라서 1차원 배열은 행으로, 2차원 배열은 행렬로, 3차원 배열은 행렬 목록으로 출력됩니다.

```python
a = np.arange(6)                    # 1d array
print(a)
[0 1 2 3 4 5]

b = np.arange(12).reshape(4, 3)     # 2d array
print(b)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]

c = np.arange(24).reshape(2, 3, 4)  # 3d array
print(c)
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
```

See below to get more details on reshape.

> reshape에 관한 자세한 내용은 아래에서 확인할 수 있습니다.

If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:

> 배열이 출력하기에 너무 큰 경우, NumPy는 자동으로 배열의 중앙 부분을 생략하고 모서리 부분만 출력합니다:

```python
print(np.arange(10000))
[   0    1    2 ... 9997 9998 9999]

print(np.arange(10000).reshape(100, 100))
[[   0    1    2 ...   97   98   99]
 [ 100  101  102 ...  197  198  199]
 [ 200  201  202 ...  297  298  299]
 ...
 [9700 9701 9702 ... 9797 9798 9799]
 [9800 9801 9802 ... 9897 9898 9899]
 [9900 9901 9902 ... 9997 9998 9999]]
```

To disable this behaviour and force NumPy to print the entire array, you can change the printing options using set_printoptions.

> 이 동작을 비활성화하고 NumPy가 전체 배열을 출력하도록 하려면 set_printoptions를 사용하여 출력 옵션을 변경할 수 있습니다.

```python
np.set_printoptions(threshold=sys.maxsize)  # sys module should be imported
```

## Basic operations
## 기본 연산

Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

> 배열에 대한 산술 연산자는 요소별로 적용됩니다. 결과로 새 배열이 생성되고 채워집니다.

```python
a = np.array([20, 30, 40, 50])
b = np.arange(4)
b
array([0, 1, 2, 3])
c = a - b
c
array([20, 29, 38, 47])
b**2
array([0, 1, 4, 9])
10 * np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
a < 35
array([ True,  True, False, False])
```

Unlike in many matrix languages, the product operator * operates elementwise in NumPy arrays. The matrix product can be performed using the @ operator (in python >=3.5) or the dot function or method:

> 다른 많은 행렬 언어와 달리, NumPy 배열에서 곱셈 연산자 *는 요소별로 작동합니다. 행렬 곱은 @ 연산자(파이썬 >=3.5)나 dot 함수 또는 메서드를 사용하여 수행할 수 있습니다:

```python
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
A * B     # elementwise product
array([[2, 0],
       [0, 4]])
A @ B     # matrix product
array([[5, 4],
       [3, 4]])
A.dot(B)  # another matrix product
array([[5, 4],
       [3, 4]])
```

Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.

> +=, *= 같은 일부 연산은 새 배열을 생성하지 않고 기존 배열을 직접 수정합니다.

```python
rg = np.random.default_rng(1)  # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
a *= 3
a
array([[3, 3, 3],
       [3, 3, 3]])
b += a
b
array([[3.51182162, 3.9504637 , 3.14415961],
       [3.94864945, 3.31183145, 3.42332645]])
a += b  # b is not automatically converted to integer type
Traceback (most recent call last):
    ...
numpy._core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
```

When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).

> 서로 다른 타입의 배열로 연산할 때, 결과 배열의 타입은 더 일반적이거나 정밀한 타입으로 결정됩니다(이를 업캐스팅이라고 합니다).

```python
a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
b.dtype.name
'float64'
c = a + b
c
array([1.        , 2.57079633, 4.14159265])
c.dtype.name
'float64'
d = np.exp(c * 1j)
d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
d.dtype.name
'complex128'
```

Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the ndarray class.

> 배열의 모든 요소 합계를 계산하는 것과 같은 많은 단항 연산은 ndarray 클래스의 메서드로 구현되어 있습니다.

```python
a = rg.random((2, 3))
a
array([[0.82770259, 0.40919914, 0.54959369],
       [0.02755911, 0.75351311, 0.53814331]])
a.sum()
3.1057109529998157
a.min()
0.027559113243068367
a.max()
0.8277025938204418
```

By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the axis parameter you can apply an operation along the specified axis of an array:

> 기본적으로 이러한 연산은 배열의 형태에 상관없이 숫자 목록처럼 적용됩니다. 그러나 axis 매개변수를 지정하면 배열의 특정 축을 따라 연산을 적용할 수 있습니다:

```python
b = np.arange(12).reshape(3, 4)
b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

b.sum(axis=0)     # sum of each column
array([12, 15, 18, 21])

b.min(axis=1)     # min of each row
array([0, 4, 8])

b.cumsum(axis=1)  # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
```

## Universal functions
## 유니버설 함수

NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called "universal functions" (ufunc). Within NumPy, these functions operate elementwise on an array, producing an array as output.

> NumPy는 sin, cos, exp와 같은 익숙한 수학 함수들을 제공합니다. NumPy에서 이것들은 '유니버설 함수'(ufunc)라고 불립니다. NumPy 내에서 이러한 함수들은 배열에 대해 요소별로 작동하며, 출력으로 배열을 생성합니다.

```python
B = np.arange(3)
B
array([0, 1, 2])
np.exp(B)
array([1.        , 2.71828183, 7.3890561 ])
np.sqrt(B)
array([0.        , 1.        , 1.41421356])
C = np.array([2., -1., 4.])
np.add(B, C)
array([2., 0., 6.])
```

## Indexing, slicing and iterating
## 인덱싱, 슬라이싱 및 반복

One-dimensional arrays can be indexed, sliced and iterated over, much like lists and other Python sequences.

> 1차원 배열은 리스트 및 다른 파이썬 시퀀스처럼 인덱싱, 슬라이싱 및 반복이 가능합니다.

```python
a = np.arange(10)**3
a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
a[2]
8
a[2:5]
array([ 8, 27, 64])
# equivalent to a[0:6:2] = 1000;
# from start to position 6, exclusive, set every 2nd element to 1000
a[:6:2] = 1000
a
array([1000,    1, 1000,   27, 1000,  125,  216,  343,  512,  729])
a[::-1]  # reversed a
array([ 729,  512,  343,  216,  125, 1000,   27, 1000,    1, 1000])
for i in a:
    print(i**(1 / 3.))

9.999999999999998  # may vary
1.0
9.999999999999998
3.0
9.999999999999998
4.999999999999999
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```

Multidimensional arrays can have one index per axis. These indices are given in a tuple separated by commas:

> 다차원 배열은 각 축마다 하나의 인덱스를 가질 수 있습니다. 이러한 인덱스들은 쉼표로 구분된 튜플로 제공됩니다:

```python
def f(x, y):
    return 10 * x + y

b = np.fromfunction(f, (5, 4), dtype=int)
b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
b[2, 3]
23
b[0:5, 1]  # each row in the second column of b
array([ 1, 11, 21, 31, 41])
b[:, 1]    # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
b[1:3, :]  # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
```

When fewer indices are provided than the number of axes, the missing indices are considered complete slices:

> 축의 개수보다 더 적은 수의 인덱스가 제공되면, 누락된 인덱스는 완전한 슬라이스로 간주됩니다:

```python
b[-1]   # the last row. Equivalent to b[-1, :]
array([40, 41, 42, 43])
```

The expression within brackets in b[i] is treated as an i followed by as many instances of : as needed to represent the remaining axes. NumPy also allows you to write this using dots as b[i, ...].

> b[i]의 대괄호 안에 있는 표현식은 i와 함께 나머지 축을 표현하기 위해 필요한 만큼의 : 인스턴스가 따라오는 것으로 처리됩니다. NumPy는 b[i, ...]와 같이 점을 사용하여 이를 작성할 수도 있게 합니다.

The dots (...) represent as many colons as needed to produce a complete indexing tuple. For example, if x is an array with 5 axes, then

> 점(...)은 완전한 인덱싱 튜플을 만드는 데 필요한 만큼의 콜론을 나타냅니다. 예를 들어, x가 5개의 축을 가진 배열이라면:

x[1, 2, ...] is equivalent to x[1, 2, :, :, :],

x[..., 3] to x[:, :, :, :, 3] and

x[4, ..., 5, :] to x[4, :, :, 5, :].

> x[1, 2, ...]는 x[1, 2, :, :, :]와 동일하고,
> 
> x[..., 3]은 x[:, :, :, :, 3]과 동일하며,
> 
> x[4, ..., 5, :]는 x[4, :, :, 5, :]와 동일합니다.

```python
c = np.array([[[  0,  1,  2],  # a 3D array (two stacked 2D arrays)
               [ 10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])
c.shape
(2, 2, 3)
c[1, ...]  # same as c[1, :, :] or c[1]
array([[100, 101, 102],
       [110, 112, 113]])
c[..., 2]  # same as c[:, :, 2]
array([[  2,  13],
       [102, 113]])
```

Iterating over multidimensional arrays is done with respect to the first axis:

> 다차원 배열에 대한 반복은 첫 번째 축을 기준으로 수행됩니다:

```python
for row in b:
    print(row)

[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
```

However, if one wants to perform an operation on each element in the array, one can use the flat attribute which is an iterator over all the elements of the array:

> 그러나 배열의 각 요소에 대해 작업을 수행하려면 배열의 모든 요소를 반복하는 이터레이터인 flat 속성을 사용할 수 있습니다:

```python
for element in b.flat:
    print(element)

0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43
```

## Shape manipulation
## 배열 모양 조작

### Changing the shape of an array
### 배열의 모양 변경하기

An array has a shape given by the number of elements along each axis:

> 배열의 모양은 각 축을 따라 있는 요소들의 개수로 정의됩니다:

```python
a = np.floor(10 * rg.random((3, 4)))
a
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
a.shape
(3, 4)
```

The shape of an array can be changed with various commands. Note that the following three commands all return a modified array, but do not change the original array:

> 배열의 모양은 다양한 명령어로 변경할 수 있습니다. 다음 세 명령어는 모두 수정된 배열을 반환하지만, 원래 배열은 변경하지 않는다는 점에 유의하세요:

```python
a.ravel()  # returns the array, flattened
array([3., 7., 3., 4., 1., 4., 2., 2., 7., 2., 4., 9.])
a.reshape(6, 2)  # returns the array with a modified shape
array([[3., 7.],
       [3., 4.],
       [1., 4.],
       [2., 2.],
       [7., 2.],
       [4., 9.]])
a.T  # returns the array, transposed
array([[3., 1., 7.],
       [7., 4., 2.],
       [3., 2., 4.],
       [4., 2., 9.]])
a.T.shape
(4, 3)
a.shape
(3, 4)
```

The order of the elements in the array resulting from ravel is normally "C-style", that is, the rightmost index "changes the fastest", so the element after a[0, 0] is a[0, 1]. If the array is reshaped to some other shape, again the array is treated as "C-style". NumPy normally creates arrays stored in this order, so ravel will usually not need to copy its argument, but if the array was made by taking slices of another array or created with unusual options, it may need to be copied. The functions ravel and reshape can also be instructed, using an optional argument, to use FORTRAN-style arrays, in which the leftmost index changes the fastest.

> ravel로 인한 결과 배열에서 요소의 순서는 일반적으로 "C-스타일"입니다. 즉, 가장 오른쪽 인덱스가 "가장 빠르게 변경"됩니다. 따라서 a[0, 0] 다음 요소는 a[0, 1]입니다. 배열이 다른 모양으로 재구성되는 경우에도 역시 "C-스타일"로 처리됩니다. NumPy는 보통 이 순서로 저장된 배열을 생성하므로 ravel은 일반적으로 인수를 복사할 필요가 없지만, 배열이 다른 배열의 슬라이스를 취하거나 특이한 옵션으로 생성된 경우에는 복사가 필요할 수 있습니다. ravel과 reshape 함수는 선택적 인수를 사용하여 가장 왼쪽 인덱스가 가장 빠르게 변경되는 FORTRAN-스타일 배열을 사용하도록 지시할 수도 있습니다.

The reshape function returns its argument with a modified shape, whereas the ndarray.resize method modifies the array itself:

> reshape 함수는 수정된 모양으로 인수를 반환하는 반면, ndarray.resize 메서드는 배열 자체를 수정합니다:

```python
a
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
a.resize((2, 6))
a
array([[3., 7., 3., 4., 1., 4.],
       [2., 2., 7., 2., 4., 9.]])
```

If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated:

> 재구성 작업에서 차원을 -1로 지정하면 다른 차원은 자동으로 계산됩니다:

```python
a.reshape(3, -1)
array([[3., 7., 3., 4.],
       [1., 4., 2., 2.],
       [7., 2., 4., 9.]])
```

### "Automatic" reshaping
### "자동" 리셰이핑

To change the dimensions of an array, you can omit one of the sizes which will then be deduced automatically:

> 배열의 차원을 변경할 때, 크기 중 하나를 생략하면 자동으로 추론됩니다:

```python
a = np.arange(30)
b = a.reshape((2, -1, 3))  # -1 means "whatever is needed"
b.shape
(2, 5, 3)
b
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14]],

       [[15, 16, 17],
        [18, 19, 20],
        [21, 22, 23],
        [24, 25, 26],
        [27, 28, 29]]])
```

### Stacking together different arrays
### 서로 다른 배열 쌓기

Several arrays can be stacked together along different axes:

> 여러 배열은 서로 다른 축을 따라 함께 쌓을 수 있습니다:

```python
a = np.floor(10 * rg.random((2, 2)))
a
array([[9., 7.],
       [5., 2.]])
b = np.floor(10 * rg.random((2, 2)))
b
array([[1., 9.],
       [5., 1.]])
np.vstack((a, b))
array([[9., 7.],
       [5., 2.],
       [1., 9.],
       [5., 1.]])
np.hstack((a, b))
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
```

### Vector stacking
### 벡터 스택킹

How do we construct a 2D array from a list of equally-sized row vectors? In MATLAB this is quite easy: if x and y are two vectors of the same length you only need do m=[x;y]. In NumPy this works via the functions column_stack, dstack, hstack and vstack, depending on the dimension in which the stacking is to be done. For example:

> 동일한 크기의 행 벡터 목록에서 2D 배열을 어떻게 구성할까요? MATLAB에서는 꽤 쉽습니다: x와 y가 같은 길이의 두 벡터라면 m=[x;y]만 하면 됩니다. NumPy에서는 스택킹을 수행할 차원에 따라 column_stack, dstack, hstack 및 vstack 함수를 통해 작동합니다. 예를 들어:

```python
x = np.arange(0, 10, 2)
y = np.arange(5)
m = np.vstack([x, y])
m
array([[0, 2, 4, 6, 8],
       [0, 1, 2, 3, 4]])
xy = np.hstack([x, y])
xy
array([0, 2, 4, 6, 8, 0, 1, 2, 3, 4])
```

The logic behind those functions in more than two dimensions can be strange.

> 두 차원 이상에서 이러한 함수들의 논리는 이상하게 보일 수 있습니다.

See also

NumPy for MATLAB users

> 참조
> 
> MATLAB 사용자를 위한 NumPy

The function column_stack stacks 1D arrays as columns into a 2D array. It is equivalent to hstack only for 2D arrays:

> column_stack 함수는 1D 배열들을 2D 배열의 열로 쌓습니다. 이는 2D 배열에 대해서만 hstack과 동일합니다:

```python
from numpy import newaxis
np.column_stack((a, b))  # with 2D arrays
array([[9., 7., 1., 9.],
       [5., 2., 5., 1.]])
a = np.array([4., 2.])
b = np.array([3., 8.])
np.column_stack((a, b))  # returns a 2D array
array([[4., 3.],
       [2., 8.]])
np.hstack((a, b))        # the result is different
array([4., 2., 3., 8.])
a[:, newaxis]  # view `a` as a 2D column vector
array([[4.],
       [2.]])
np.column_stack((a[:, newaxis], b[:, newaxis]))
array([[4., 3.],
       [2., 8.]])
np.hstack((a[:, newaxis], b[:, newaxis]))  # the result is the same
array([[4., 3.],
       [2., 8.]])
```

In general, for arrays with more than two dimensions, hstack stacks along their second axes, vstack stacks along their first axes, and concatenate allows for an optional arguments giving the number of the axis along which the concatenation should happen.

> 일반적으로, 두 개 이상의 차원을 가진 배열의 경우, hstack은 두 번째 축을 따라 쌓고, vstack은 첫 번째 축을 따라 쌓으며, concatenate는 연결이 일어나야 할 축의 번호를 지정하는 선택적 인자를 허용합니다.

Note

In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. They allow the use of range literals :.

> 참고
> 
> 복잡한 경우, r_와 c_는 한 축을 따라 숫자를 쌓아 배열을 생성하는 데 유용합니다. 이들은 범위 리터럴 :의 사용을 허용합니다.

```python
np.r_[1:4, 0, 4]
array([1, 2, 3, 0, 4])
```

When used with arrays as arguments, r_ and c_ are similar to vstack and hstack in their default behavior, but allow for an optional argument giving the number of the axis along which to concatenate.

> 배열을 인자로 사용할 때, r_와 c_는 기본 동작에서 vstack과 hstack과 유사하지만, 연결할 축의 번호를 지정하는 선택적 인자를 허용합니다.

See also

hstack, vstack, column_stack, concatenate, c_, r_

> 참조
> 
> hstack, vstack, column_stack, concatenate, c_, r_

### Splitting one array into several smaller ones
### 하나의 배열을 여러 개의 작은 배열로 분할하기

Using hsplit, you can split an array along its horizontal axis, either by specifying the number of equally shaped arrays to return, or by specifying the columns after which the division should occur:

> hsplit을 사용하여 배열을 수평 축을 따라 분할할 수 있습니다. 동일한 형태의 배열 수를 지정하거나, 분할이 발생해야 할 열을 지정하는 방식으로 나눌 수 있습니다:

```python
a = np.floor(10 * rg.random((2, 12)))
a
array([[6., 7., 6., 9., 0., 5., 4., 0., 6., 8., 5., 2.],
       [8., 5., 5., 7., 1., 8., 6., 7., 1., 8., 1., 0.]])
# Split `a` into 3
np.hsplit(a, 3)
array([[6., 7., 6., 9.],
       [8., 5., 5., 7.]]), array([[0., 5., 4., 0.],
       [1., 8., 6., 7.]]), array([[6., 8., 5., 2.],
       [1., 8., 1., 0.]])]
# Split `a` after the third and the fourth column
np.hsplit(a, (3, 4))
array([[6., 7., 6.],
       [8., 5., 5.]]), array([[9.],
       [7.]]), array([[0., 5., 4., 0., 6., 8., 5., 2.],
       [1., 8., 6., 7., 1., 8., 1., 0.]])]
```

vsplit splits along the vertical axis, and array_split allows one to specify along which axis to split.

> vsplit은 수직 축을 따라 분할하고, array_split은 어떤 축을 따라 분할할지 지정할 수 있게 합니다.

## Copies and views
## 복사와 뷰

When operating and manipulating arrays, their data is sometimes copied into a new array and sometimes not. This is often a source of confusion for beginners. There are three cases:

> 배열을 연산하고 조작할 때, 데이터가 때때로 새 배열에 복사되고 때로는 그렇지 않습니다. 이는 종종 초보자들이 혼란을 겪는 원인이 됩니다. 세 가지 경우가 있습니다:

### No copy at all
### 전혀 복사하지 않는 경우

Simple assignments make no copy of objects or their data.

> 단순 할당은 객체나 해당 데이터의 복사본을 만들지 않습니다.

```python
a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
b = a            # no new object is created
b is a           # a and b are two names for the same ndarray object
True
```

Python passes mutable objects as references, so function calls make no copy.

> 파이썬은 변경 가능한 객체를 참조로 전달하므로, 함수 호출은 복사본을 만들지 않습니다.

```python
def f(x):
    print(id(x))

id(a)  # id is a unique identifier of an object 
148293216  # may vary
f(a)   
148293216  # may vary
```

### View or shallow copy
### 뷰 또는 얕은 복사

Different array objects can share the same data. The view method creates a new array object that looks at the same data.

> 서로 다른 배열 객체가 동일한 데이터를 공유할 수 있습니다. view 메서드는 동일한 데이터를 보는 새 배열 객체를 생성합니다.

```python
c = a.view()
c is a
False
c.base is a            # c is a view of the data owned by a
True
c.flags.owndata
False

c = c.reshape((2, 6))  # a's shape doesn't change, reassigned c is still a view of a
a.shape
(3, 4)
c[0, 4] = 1234         # a's data changes
a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

Slicing an array returns a view of it:

> 배열을 슬라이싱하면 해당 배열의 뷰가 반환됩니다:

```python
s = a[:, 1:3]
s[:] = 10  # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

### Deep copy
### 깊은 복사

The copy method makes a complete copy of the array and its data.

> copy 메서드는 배열과 그 데이터의 완전한 복사본을 만듭니다.

```python
d = a.copy()  # a new array object with new data is created
d is a
False
d.base is a  # d doesn't share anything with a
False
d[0, 0] = 9999
a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

Sometimes copy should be called after slicing if the original array is not required anymore. For example, suppose a is a huge intermediate result and the final result b only contains a small fraction of a, a deep copy should be made when constructing b with slicing:

> 원본 배열이 더 이상 필요하지 않은 경우 슬라이싱 후 copy를 호출해야 할 때가 있습니다. 예를 들어, a가 거대한 중간 결과이고 최종 결과 b가 a의 작은 부분만 포함한다고 가정하면, b를 슬라이싱으로 구성할 때 깊은 복사를 수행해야 합니다:

```python
a = np.arange(int(1e8))
b = a[:100].copy()
del a  # the memory of ``a`` can be released.
```

If b = a[:100] is used instead, a is referenced by b and will persist in memory even if del a is executed.

> 대신 b = a[:100]이 사용된다면, a는 b에 의해 참조되어 del a가 실행되더라도 메모리에 계속 남아있게 됩니다.

## Functions and methods overview
## 함수와 메서드 개요

Here is a list of some useful NumPy functions and methods names ordered in categories. See Routines and objects by topic for the full list.

> 다음은 범주별로 정리된 유용한 NumPy 함수 및 메서드 이름 목록입니다. 전체 목록은 주제별 루틴 및 객체를 참조하세요.

**Array Creation**
**배열 생성**
arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r_, zeros, zeros_like

**Conversions**
**변환**
ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

**Manipulations**
**조작**
array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

**Questions**
**질문**
all, any, nonzero, where

**Ordering**
**정렬**
argmax, argmin, argsort, max, min, ptp, searchsorted, sort

**Operations**
**연산**
choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum

**Basic Statistics**
**기본 통계**
cov, mean, std, var

**Basic Linear Algebra**
**기본 선형대수**
cross, dot, outer, linalg.svd, vdot

## Broadcasting rules
## 브로드캐스팅 규칙

Broadcasting allows universal functions to deal in a meaningful way with inputs that do not have exactly the same shape.

> 브로드캐스팅은 유니버설 함수가 정확히 같은 형태를 갖지 않는 입력들을 의미 있는 방식으로 처리할 수 있게 합니다.

The first rule of broadcasting is that if all input arrays do not have the same number of dimensions, a "1" will be repeatedly prepended to the shapes of the smaller arrays until all the arrays have the same number of dimensions.

> 브로드캐스팅의 첫 번째 규칙은 모든 입력 배열이 같은 차원 수를 가지고 있지 않을 경우, 모든 배열이 같은 차원 수를 가질 때까지 작은 배열의 형태에 "1"이 반복적으로 앞에 추가된다는 것입니다.

The second rule of broadcasting ensures that arrays with a size of 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is assumed to be the same along that dimension for the "broadcast" array.

> 브로드캐스팅의 두 번째 규칙은 특정 차원을 따라 크기가 1인 배열이 해당 차원을 따라 가장 큰 형태의 배열 크기를 가진 것처럼 작동하도록 보장한다는 것입니다. "브로드캐스트" 배열의 경우 해당 차원을 따라 배열 요소의 값은 동일하다고 가정합니다.

After application of the broadcasting rules, the sizes of all arrays must match. More details can be found in Broadcasting.

> 브로드캐스팅 규칙을 적용한 후에는 모든 배열의 크기가 일치해야 합니다. 자세한 내용은 브로드캐스팅에서 확인할 수 있습니다.

## Advanced indexing and index tricks
## 고급 인덱싱 및 인덱스 기법

NumPy offers more indexing facilities than regular Python sequences. In addition to indexing by integers and slices, as we saw before, arrays can be indexed by arrays of integers and arrays of booleans.

> NumPy는 일반 파이썬 시퀀스보다 더 많은 인덱싱 기능을 제공합니다. 앞서 살펴본 정수와 슬라이스에 의한 인덱싱 외에도, 배열은 정수 배열 및 불리언 배열로 인덱싱할 수 있습니다.

### Indexing with arrays of indices
### 인덱스 배열을 사용한 인덱싱

```python
a = np.arange(12)**2  # the first 12 square numbers
i = np.array([1, 1, 3, 8, 5])  # an array of indices
a[i]  # the elements of `a` at the positions `i`
array([ 1,  1,  9, 64, 25])

j = np.array([[3, 4], [9, 7]])  # a bidimensional array of indices
a[j]  # the same shape as `j`
array([[ 9, 16],
       [81, 49]])
```

When the indexed array a is multidimensional, a single array of indices refers to the first dimension of a. The following example shows this behavior by converting an image of labels into a color image using a palette.

> 인덱싱된 배열 a가 다차원인 경우, 단일 인덱스 배열은 a의 첫 번째 차원을 참조합니다. 다음 예제는 팔레트를 사용하여 레이블 이미지를 컬러 이미지로 변환하는 이 동작을 보여줍니다.

```python
palette = np.array([[0, 0, 0],         # black
                    [255, 0, 0],       # red
                    [0, 255, 0],       # green
                    [0, 0, 255],       # blue
                    [255, 255, 255]])  # white
image = np.array([[0, 1, 2, 0],  # each value corresponds to a color in the palette
                  [0, 3, 4, 0]])
palette[image]  # the (2, 4, 3) color image
array([[[  0,   0,   0],
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0,   0]],

       [[  0,   0,   0],
        [  0,   0, 255],
        [255, 255, 255],
        [  0,   0,   0]]])
```

We can also give indexes for more than one dimension. The arrays of indices for each dimension must have the same shape.

> 하나 이상의 차원에 대한 인덱스를 제공할 수도 있습니다. 각 차원에 대한 인덱스 배열은 같은 형태를 가져야 합니다.

```python
a = np.arange(12).reshape(3, 4)
a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
i = np.array([[0, 1],  # indices for the first dim of `a`
              [1, 2]])
j = np.array([[2, 1],  # indices for the second dim
              [3, 3]])

a[i, j]  # i and j must have equal shape
array([[ 2,  5],
       [ 7, 11]])

a[i, 2]
array([[ 2,  6],
       [ 6, 10]])

a[:, j]
array([[[ 2,  1],
        [ 3,  3]],

       [[ 6,  5],
        [ 7,  7]],

       [[10,  9],
        [11, 11]]])
```

In Python, arr[i, j] is exactly the same as arr[(i, j)]—so we can put i and j in a tuple and then do the indexing with that.

> 파이썬에서 arr[i, j]는 arr[(i, j)]와 정확히 동일합니다. 따라서 i와 j를 튜플에 넣고 그것으로 인덱싱을 할 수 있습니다.

```python
l = (i, j)
# equivalent to a[i, j]
a[l]
array([[ 2,  5],
       [ 7, 11]])
```

However, we can not do this by putting i and j into an array, because this array will be interpreted as indexing the first dimension of a.

> 그러나 i와 j를 배열에 넣어서는 이것을 할 수 없습니다. 이 배열은 a의 첫 번째 차원을 인덱싱하는 것으로 해석되기 때문입니다.

```python
s = np.array([i, j])
# not what we want
a[s]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3
# same as `a[i, j]`
a[tuple(s)]
array([[ 2,  5],
       [ 7, 11]])
```

Another common use of indexing with arrays is the search of the maximum value of time-dependent series:

> 배열로 인덱싱하는 또 다른 일반적인 용도는 시간 의존적 시리즈의 최대값을 검색하는 것입니다:

```python
time = np.linspace(20, 145, 5)  # time scale
data = np.sin(np.arange(20)).reshape(5, 4)  # 4 time-dependent series
time
array([ 20.  ,  51.25,  82.5 , 113.75, 145.  ])
data
array([[ 0.        ,  0.84147098,  0.90929743,  0.14112001],
       [-0.7568025 , -0.95892427, -0.2794155 ,  0.6569866 ],
       [ 0.98935825,  0.41211849, -0.54402111, -0.99999021],
       [-0.53657292,  0.42016704,  0.99060736,  0.65028784],
       [-0.28790332, -0.96139749, -0.75098725,  0.14987721]])
# index of the maxima for each series
ind = data.argmax(axis=0)
ind
array([2, 0, 3, 1])
# times corresponding to the maxima
time_max = time[ind]

data_max = data[ind, range(data.shape[1])]  # => data[ind[0], 0], data[ind[1], 1]...
time_max
array([ 82.5 ,  20.  , 113.75,  51.25])
data_max
array([0.98935825, 0.84147098, 0.99060736, 0.6569866 ])
np.all(data_max == data.max(axis=0))
True
```

You can also use indexing with arrays as a target to assign to:

> 배열로 인덱싱을 대상에 할당하는 데에도 사용할 수 있습니다:

```python
a = np.arange(5)
a
array([0, 1, 2, 3, 4])
a[[1, 3, 4]] = 0
a
array([0, 0, 2, 0, 0])
```

However, when the list of indices contains repetitions, the assignment is done several times, leaving behind the last value:

> 그러나 인덱스 목록에 반복이 포함된 경우 할당은 여러 번 수행되며, 마지막 값만 남게 됩니다:

```python
a = np.arange(5)
a[[0, 0, 2]] = [1, 2, 3]
a
array([2, 1, 3, 3, 4])
```

This is reasonable enough, but watch out if you want to use Python's += construct, as it may not do what you expect:

> 이것은 충분히 합리적이지만, 파이썬의 += 구문을 사용하려는 경우 예상대로 작동하지 않을 수 있으므로 주의하세요:

```python
a = np.arange(5)
a[[0, 0, 2]] += 1
a
array([1, 1, 3, 3, 4])
```

Even though 0 occurs twice in the list of indices, the 0th element is only incremented once. This is because Python requires a += 1 to be equivalent to a = a + 1.

> 0이 인덱스 목록에 두 번 나타나더라도 0번째 요소는 한 번만 증가합니다. 이는 파이썬이 a += 1을 a = a + 1과 동등하게 요구하기 때문입니다.

### Indexing with boolean arrays
### 불리언 배열을 사용한 인덱싱

When we index arrays with arrays of (integer) indices we are providing the list of indices to pick. With boolean indices the approach is different; we explicitly choose which items in the array we want and which ones we don't.

> (정수) 인덱스의 배열로 배열을 인덱싱할 때, 우리는 선택할 인덱스의 목록을 제공합니다. 불리언 인덱스를 사용할 때는 접근 방식이 다릅니다. 배열에서 원하는 항목과 원하지 않는 항목을 명시적으로 선택합니다.

The most natural way one can think of for boolean indexing is to use boolean arrays that have the same shape as the original array:

> 불리언 인덱싱을 위한 가장 자연스러운 방법은 원본 배열과 동일한 형태를 가진 불리언 배열을 사용하는 것입니다:

```python
a = np.arange(12).reshape(3, 4)
b = a > 4
b  # `b` is a boolean with `a`'s shape
array([[False, False, False, False],
       [False,  True,  True,  True],
       [ True,  True,  True,  True]])
a[b]  # 1d array with the selected elements
array([ 5,  6,  7,  8,  9, 10, 11])
```

This property can be very useful in assignments:

> 이 속성은 할당에 매우 유용할 수 있습니다:

```python
a[b] = 0  # All elements of `a` higher than 4 become 0
a
array([[0, 1, 2, 3],
       [4, 0, 0, 0],
       [0, 0, 0, 0]])
```

You can look at the following example to see how to use boolean indexing to generate an image of the Mandelbrot set:

> 다음 예제를 통해 불리언 인덱싱을 사용하여 만델브로트 집합의 이미지를 생성하는 방법을 확인할 수 있습니다:

```python
import numpy as np
import matplotlib.pyplot as plt
def mandelbrot(h, w, maxit=20, r=2):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    x = np.linspace(-2.5, 1.5, 4*h+1)
    y = np.linspace(-1.5, 1.5, 3*w+1)
    A, B = np.meshgrid(x, y)
    C = A + B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r                    # who is diverging
        div_now = diverge & (divtime == maxit)  # who is diverging now
        divtime[div_now] = i                    # note when
        z[diverge] = r                          # avoid diverging too much

    return divtime
plt.clf()
plt.imshow(mandelbrot(400, 400))
```

The second way of indexing with booleans is more similar to integer indexing; for each dimension of the array we give a 1D boolean array selecting the slices we want:

> 불리언으로 인덱싱하는 두 번째 방법은 정수 인덱싱과 더 유사합니다. 배열의 각 차원에 대해 원하는 슬라이스를 선택하는 1D 불리언 배열을 제공합니다:

```python
a = np.arange(12).reshape(3, 4)
b1 = np.array([False, True, True])         # first dim selection
b2 = np.array([True, False, True, False])  # second dim selection

a[b1, :]                                   # selecting rows
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

a[b1]                                      # same thing
array([[ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

a[:, b2]                                   # selecting columns
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])

a[b1, b2]                                  # a weird thing to do
array([ 4, 10])
```

Note that the length of the 1D boolean array must coincide with the length of the dimension (or axis) you want to slice. In the previous example, b1 has length 3 (the number of rows in a), and b2 (of length 4) is suitable to index the 2nd axis (columns) of a.

> 1D 불리언 배열의 길이는 슬라이스하려는 차원(또는 축)의 길이와 일치해야 합니다. 이전 예제에서 b1의 길이는 3(a의 행 수)이고, b2(길이 4)는 a의 2번째 축(열)을 인덱싱하는 데 적합합니다.

### The ix_() function
### ix_() 함수

The ix_ function can be used to combine different vectors so as to obtain the result for each n-uplet. For example, if you want to compute all the a+b*c for all the triplets taken from each of the vectors a, b and c:

> ix_ 함수는 서로 다른 벡터들을 결합하여 각 n-튜플에 대한 결과를 얻는 데 사용할 수 있습니다. 예를 들어, 벡터 a, b, c에서 가져온 모든 삼중항에 대해 a+b*c를 계산하려면 다음과 같이 할 수 있습니다:

```python
a = np.array([2, 3, 4, 5])
b = np.array([8, 5, 4])
c = np.array([5, 4, 6, 8, 3])
ax, bx, cx = np.ix_(a, b, c)
ax
array([[[2]],

       [[3]],

       [[4]],

       [[5]]])
bx
array([[[8],
        [5],
        [4]]])
cx
array([[[5, 4, 6, 8, 3]]])
ax.shape, bx.shape, cx.shape
((4, 1, 1), (1, 3, 1), (1, 1, 5))
result = ax + bx * cx
result
array([[[42, 34, 50, 66, 26],
        [27, 22, 32, 42, 17],
        [22, 18, 26, 34, 14]],

       [[43, 35, 51, 67, 27],
        [28, 23, 33, 43, 18],
        [23, 19, 27, 35, 15]],

       [[44, 36, 52, 68, 28],
        [29, 24, 34, 44, 19],
        [24, 20, 28, 36, 16]],

       [[45, 37, 53, 69, 29],
        [30, 25, 35, 45, 20],
        [25, 21, 29, 37, 17]]])
result[3, 2, 4]
17
a[3] + b[2] * c[4]
17
```

You could also implement the reduce as follows:

> 다음과 같이 reduce를 구현할 수도 있습니다:

```python
def ufunc_reduce(ufct, *vectors):
   vs = np.ix_(*vectors)
   r = ufct.identity
   for v in vs:
       r = ufct(r, v)
   return r
```

and then use it as:

> 그리고 다음과 같이 사용할 수 있습니다:

```python
ufunc_reduce(np.add, a, b, c)
array([[[15, 14, 16, 18, 13],
        [12, 11, 13, 15, 10],
        [11, 10, 12, 14,  9]],

       [[16, 15, 17, 19, 14],
        [13, 12, 14, 16, 11],
        [12, 11, 13, 15, 10]],

       [[17, 16, 18, 20, 15],
        [14, 13, 15, 17, 12],
        [13, 12, 14, 16, 11]],

       [[18, 17, 19, 21, 16],
        [15, 14, 16, 18, 13],
        [14, 13, 15, 17, 12]]])
```

The advantage of this version of reduce compared to the normal ufunc.reduce is that it makes use of the broadcasting rules in order to avoid creating an argument array the size of the output times the number of vectors.

> 이 버전의 reduce가 일반적인 ufunc.reduce에 비해 갖는 장점은 출력 크기와 벡터 수를 곱한 크기의 인수 배열을 생성하지 않기 위해 브로드캐스팅 규칙을 활용한다는 것입니다.

### Histograms
### 히스토그램

The NumPy histogram function applied to an array returns a pair of vectors: the histogram of the array and a vector of the bin edges. Beware: matplotlib also has a function to build histograms (called hist, as in Matlab) that differs from the one in NumPy. The main difference is that pylab.hist plots the histogram automatically, while numpy.histogram only generates the data.

> 배열에 적용된 NumPy histogram 함수는 두 개의 벡터를 반환합니다: 배열의 히스토그램과 구간 경계의 벡터입니다. 주의: matplotlib도 히스토그램을 만드는 함수(Matlab처럼 hist라고 불림)가 있으며, 이는 NumPy에 있는 함수와 다릅니다. 주요 차이점은 pylab.hist는 히스토그램을 자동으로 그리는 반면, numpy.histogram은 데이터만 생성한다는 것입니다.

```python
import numpy as np
rg = np.random.default_rng(1)
import matplotlib.pyplot as plt
# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
mu, sigma = 2, 0.5
v = rg.normal(mu, sigma, 10000)
# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, density=True)       # matplotlib version (plot)
(array...)
# Compute the histogram with numpy and then plot it
(n, bins) = np.histogram(v, bins=50, density=True)  # NumPy version (no plot)
plt.plot(.5 * (bins[1:] + bins[:-1]), n) 
```

With Matplotlib >=3.4 you can also use plt.stairs(n, bins).

> Matplotlib >=3.4에서는 plt.stairs(n, bins)를 사용할 수도 있습니다.

