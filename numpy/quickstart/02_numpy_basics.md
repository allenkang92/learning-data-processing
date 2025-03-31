# NumPy: the absolute basics for beginners
# NumPy: 초보자를 위한 기본 사항

Welcome to the absolute beginner's guide to NumPy!

> NumPy 초보자 가이드에 오신 것을 환영합니다!

NumPy (Numerical Python) is an open source Python library that's widely used in science and engineering. The NumPy library contains multidimensional array data structures, such as the homogeneous, N-dimensional ndarray, and a large library of functions that operate efficiently on these data structures. Learn more about NumPy at What is NumPy, and if you have comments or suggestions, please reach out!

> NumPy(Numerical Python)는 과학 및 공학 분야에서 널리 사용되는 오픈 소스 파이썬 라이브러리입니다. NumPy 라이브러리는 동질적이고 N차원인 ndarray와 같은 다차원 배열 데이터 구조와 이러한 데이터 구조에 효율적으로 작동하는 대규모 함수 라이브러리를 포함합니다. NumPy에 대한 자세한 내용은 'NumPy란 무엇인가'에서 확인하시고, 의견이나 제안 사항이 있으시면 연락주세요!

## How to import NumPy
## NumPy 불러오기

After installing NumPy, it may be imported into Python code like:

> NumPy를 설치한 후에는 다음과 같이 Python 코드로 불러올 수 있습니다:

```python
import numpy as np
```

This widespread convention allows access to NumPy features with a short, recognizable prefix (np.) while distinguishing NumPy features from others that have the same name.

> 이 널리 사용되는 관례는 NumPy 기능에 짧고 인식하기 쉬운 접두사(np.)를 통해 접근할 수 있게 하면서도, 동일한 이름을 가진 다른 기능들과 NumPy 기능을 구별할 수 있게 합니다.

## Reading the example code
## 예제 코드 읽기

Throughout the NumPy documentation, you will find blocks that look like:

> NumPy 문서 전반에 걸쳐 다음과 같은 블록을 발견할 수 있습니다:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
a.shape
(2, 3)
```

Text preceded by >>> or ... is input, the code that you would enter in a script or at a Python prompt. Everything else is output, the results of running your code. Note that >>> and ... are not part of the code and may cause an error if entered at a Python prompt.

> >>> 또는 ...로 시작하는 텍스트는 입력으로, 스크립트나 파이썬 프롬프트에 입력할 코드입니다. 나머지는 출력으로, 코드를 실행한 결과입니다. >>> 및 ...는 코드의 일부가 아니며, 파이썬 프롬프트에 입력하면 오류가 발생할 수 있습니다.

## Why use NumPy?
## 왜 NumPy를 사용할까요?

Python lists are excellent, general-purpose containers. They can be "heterogeneous", meaning that they can contain elements of a variety of types, and they are quite fast when used to perform individual operations on a handful of elements.

> 파이썬 리스트는 우수한 범용 컨테이너입니다. 다양한 유형의 요소를 포함할 수 있는 "이종(heterogeneous)" 자료형이며, 소수의 요소에 대해 개별 작업을 수행할 때 꽤 빠릅니다.

Depending on the characteristics of the data and the types of operations that need to be performed, other containers may be more appropriate; by exploiting these characteristics, we can improve speed, reduce memory consumption, and offer a high-level syntax for performing a variety of common processing tasks. NumPy shines when there are large quantities of "homogeneous" (same-type) data to be processed on the CPU.

> 데이터의 특성과 수행해야 할 작업의 유형에 따라 다른 컨테이너가 더 적합할 수 있습니다. 이러한 특성을 활용하여 속도를 개선하고, 메모리 소비를 줄이며, 다양한 일반적인 처리 작업을 수행하기 위한 고수준 구문을 제공할 수 있습니다. NumPy는 CPU에서 처리해야 할 "동질적인(homogeneous)" (동일 유형) 데이터가 대량으로 있을 때 빛을 발합니다.

## What is an "array"?
## "배열"이란 무엇인가요?

In computer programming, an array is a structure for storing and retrieving data. We often talk about an array as if it were a grid in space, with each cell storing one element of the data. For instance, if each element of the data were a number, we might visualize a "one-dimensional" array like a list:

> 컴퓨터 프로그래밍에서 배열은 데이터를 저장하고 검색하기 위한 구조입니다. 우리는 종종 배열을 공간상의 그리드처럼 이야기하며, 각 셀은 데이터의 한 요소를 저장합니다. 예를 들어, 데이터의 각 요소가 숫자라면, "1차원" 배열을 리스트처럼 시각화할 수 있습니다:

A two-dimensional array would be like a table:

> 2차원 배열은 표처럼 보일 것입니다:

A three-dimensional array would be like a set of tables, perhaps stacked as though they were printed on separate pages. In NumPy, this idea is generalized to an arbitrary number of dimensions, and so the fundamental array class is called ndarray: it represents an "N-dimensional array".

> 3차원 배열은 여러 테이블의 집합처럼 보일 것이며, 아마도 별도의 페이지에 인쇄된 것처럼 쌓여 있을 것입니다. NumPy에서 이 개념은 임의의 차원 수로 일반화되며, 따라서 기본 배열 클래스는 ndarray라고 불립니다: 이는 "N차원 배열"을 나타냅니다.

Most NumPy arrays have some restrictions. For instance:

* All elements of the array must be of the same type of data.
* Once created, the total size of the array can't change.
* The shape must be "rectangular", not "jagged"; e.g., each row of a two-dimensional array must have the same number of columns.

> 대부분의 NumPy 배열에는 몇 가지 제한이 있습니다. 예를 들면:
>
> * 배열의 모든 요소는 동일한 데이터 유형이어야 합니다.
> * 생성된 후에는 배열의 전체 크기를 변경할 수 없습니다.
> * 모양은 "직사각형"이어야 하며 "들쭉날쭉"해서는 안 됩니다; 예를 들어, 2차원 배열의 각 행은 동일한 수의 열을 가져야 합니다.

When these conditions are met, NumPy exploits these characteristics to make the array faster, more memory efficient, and more convenient to use than less restrictive data structures.

> 이러한 조건이 충족되면, NumPy는 이러한 특성을 활용하여 배열을 더 빠르고, 메모리 효율적으로 만들며, 덜 제한적인 데이터 구조보다 더 편리하게 사용할 수 있게 합니다.

For the remainder of this document, we will use the word "array" to refer to an instance of ndarray.

> 이 문서의 나머지 부분에서는 "배열"이라는 단어를 ndarray의 인스턴스를 지칭하는 데 사용할 것입니다.

## Array fundamentals
## 배열의 기본 사항

One way to initialize an array is using a Python sequence, such as a list. For example:

> 배열을 초기화하는 한 가지 방법은 리스트와 같은 파이썬 시퀀스를 사용하는 것입니다. 예를 들어:

```python
a = np.array([1, 2, 3, 4, 5, 6])
a
array([1, 2, 3, 4, 5, 6])
```

Elements of an array can be accessed in various ways. For instance, we can access an individual element of this array as we would access an element in the original list: using the integer index of the element within square brackets.

> 배열의 요소는 다양한 방식으로 접근할 수 있습니다. 예를 들어, 원래 리스트의 요소에 접근하는 것처럼 이 배열의 개별 요소에 접근할 수 있습니다: 대괄호 안에 요소의 정수 인덱스를 사용합니다.

```python
a[0]
1
```

Note

As with built-in Python sequences, NumPy arrays are "0-indexed": the first element of the array is accessed using index 0, not 1.

> 참고
>
> 내장 파이썬 시퀀스와 마찬가지로, NumPy 배열은 "0부터 인덱싱"됩니다: 배열의 첫 번째 요소는 인덱스 1이 아닌 0을 사용하여 접근합니다.

Like the original list, the array is mutable.

> 원래 리스트처럼 배열도 변경 가능합니다.

```python
a[0] = 10
a
array([10,  2,  3,  4,  5,  6])
```

Also like the original list, Python slice notation can be used for indexing.

> 또한 원래 리스트처럼 파이썬 슬라이스 표기법을 인덱싱에 사용할 수 있습니다.

```python
a[:3]
array([10, 2, 3])
```

One major difference is that slice indexing of a list copies the elements into a new list, but slicing an array returns a view: an object that refers to the data in the original array. The original array can be mutated using the view.

> 한 가지 중요한 차이점은 리스트의 슬라이스 인덱싱은 요소를 새 리스트에 복사하지만, 배열을 슬라이싱하면 뷰가 반환된다는 것입니다: 뷰는 원래 배열의 데이터를 참조하는 개체입니다. 뷰를 사용하여 원래 배열을 변경할 수 있습니다.

```python
b = a[3:]
b
array([4, 5, 6])
b[0] = 40
a
array([ 10,  2,  3, 40,  5,  6])
```

See Copies and views for a more comprehensive explanation of when array operations return views rather than copies.

> 배열 연산이 언제 복사본이 아닌 뷰를 반환하는지에 대한 더 포괄적인 설명은 복사본 및 뷰를 참조하세요.

Two- and higher-dimensional arrays can be initialized from nested Python sequences:

> 2차원 및 더 높은 차원의 배열은 중첩된 파이썬 시퀀스로부터 초기화할 수 있습니다:

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
```

In NumPy, a dimension of an array is sometimes referred to as an "axis". This terminology may be useful to disambiguate between the dimensionality of an array and the dimensionality of the data represented by the array. For instance, the array a could represent three points, each lying within a four-dimensional space, but a has only two "axes".

> NumPy에서 배열의 차원은 때때로 "축(axis)"이라고도 합니다. 이 용어는 배열의 차원성과 배열이 나타내는 데이터의 차원성을 명확히 구분하는 데 유용할 수 있습니다. 예를 들어, 배열 a는 4차원 공간 내에 있는 세 개의 점을 나타낼 수 있지만, a는 두 개의 "축"만 가지고 있습니다.

Another difference between an array and a list of lists is that an element of the array can be accessed by specifying the index along each axis within a single set of square brackets, separated by commas. For instance, the element 8 is in row 1 and column 3:

> 배열과 리스트의 리스트 간의 또 다른 차이점은 배열의 요소가 쉼표로 구분된 단일 대괄호 세트 내에서 각 축을 따라 인덱스를 지정하여 접근할 수 있다는 것입니다. 예를 들어, 요소 8은 행 1과 열 3에 있습니다:

```python
a[1, 3]
8
```

Note

It is familiar practice in mathematics to refer to elements of a matrix by the row index first and the column index second. This happens to be true for two-dimensional arrays, but a better mental model is to think of the column index as coming last and the row index as second to last. This generalizes to arrays with any number of dimensions.

> 참고
>
> 수학에서는 행 인덱스를 먼저, 열 인덱스를 그 다음에 지정하여 행렬의 요소를 참조하는 것이 익숙한 관행입니다. 이는 2차원 배열에서는 사실이지만, 더 나은 정신적 모델은 열 인덱스가 마지막에 오고 행 인덱스가 마지막에서 두 번째로 온다고 생각하는 것입니다. 이는 임의의 차원 수를 가진 배열로 일반화됩니다.

Note

You might hear of a 0-D (zero-dimensional) array referred to as a "scalar", a 1-D (one-dimensional) array as a "vector", a 2-D (two-dimensional) array as a "matrix", or an N-D (N-dimensional, where "N" is typically an integer greater than 2) array as a "tensor". For clarity, it is best to avoid the mathematical terms when referring to an array because the mathematical objects with these names behave differently than arrays (e.g. "matrix" multiplication is fundamentally different from "array" multiplication), and there are other objects in the scientific Python ecosystem that have these names (e.g. the fundamental data structure of PyTorch is the "tensor").

> 참고
>
> 0-D(0차원) 배열은 "스칼라", 1-D(1차원) 배열은 "벡터", 2-D(2차원) 배열은 "행렬", N-D(N차원, 여기서 "N"은 일반적으로 2보다 큰 정수)는 "텐서"라고 불리는 것을 들어본 적이 있을 것입니다. 명확성을 위해, 배열을 지칭할 때는 수학적 용어를 피하는 것이 가장 좋습니다. 이러한 이름을 가진 수학적 객체는 배열과 다르게 동작하기 때문입니다(예를 들어, "행렬" 곱셈은 "배열" 곱셈과 근본적으로 다릅니다). 또한 과학적 Python 생태계에는 이러한 이름을 가진 다른 객체들도 있습니다(예를 들어, PyTorch의 기본 데이터 구조는 "텐서"입니다).

## Array attributes
## 배열 속성

This section covers the ndim, shape, size, and dtype attributes of an array.

> 이 섹션은 배열의 ndim, shape, size, dtype 속성을 다룹니다.

The number of dimensions of an array is contained in the ndim attribute.

> 배열의 차원 수는 ndim 속성에 포함됩니다.

```python
a.ndim
2
```

The shape of an array is a tuple of non-negative integers that specify the number of elements along each dimension.

> 배열의 형태(shape)는 각 차원을 따라 요소 수를 지정하는 음이 아닌 정수의 튜플입니다.

```python
a.shape
(3, 4)
len(a.shape) == a.ndim
True
```

The fixed, total number of elements in array is contained in the size attribute.

> 배열의 고정된 총 요소 수는 size 속성에 포함됩니다.

```python
a.size
12
import math
a.size == math.prod(a.shape)
True
```

Arrays are typically "homogeneous", meaning that they contain elements of only one "data type". The data type is recorded in the dtype attribute.

> 배열은 일반적으로 "동질적(homogeneous)"입니다. 즉, 하나의 "데이터 유형"의 요소만 포함합니다. 데이터 유형은 dtype 속성에 기록됩니다.

```python
a.dtype
dtype('int64')  # "int" for integer, "64" for 64-bit
```

Read more about array attributes here and learn about array objects here.

> 배열 속성에 대해 자세히 알아보려면 여기를 참고하고, 배열 객체에 대해 알아보려면 여기를 참조하세요.

## How to create a basic array
## 기본 배열 생성 방법

This section covers np.zeros(), np.ones(), np.empty(), np.arange(), np.linspace()

> 이 섹션은 np.zeros(), np.ones(), np.empty(), np.arange(), np.linspace() 함수를 다룹니다.

Besides creating an array from a sequence of elements, you can easily create an array filled with 0's:

> 요소 시퀀스에서 배열을 생성하는 것 외에도, 0으로 채워진 배열을 쉽게 생성할 수 있습니다:

```python
np.zeros(2)
array([0., 0.])
```

Or an array filled with 1's:

> 또는 1로 채워진 배열:

```python
np.ones(2)
array([1., 1.])
```

Or even an empty array! The function empty creates an array whose initial content is random and depends on the state of the memory. The reason to use empty over zeros (or something similar) is speed - just make sure to fill every element afterwards!

> 심지어 빈 배열도 가능합니다! empty 함수는 초기 내용이 무작위이고 메모리 상태에 따라 달라지는 배열을 생성합니다. zeros(또는 이와 유사한 것) 대신 empty를 사용하는 이유는 속도 때문입니다 - 다만 나중에 모든 요소를 채워야 한다는 점을 명심하세요!

```python
# Create an empty array with 2 elements
np.empty(2) 
array([3.14, 42.  ])  # may vary
```

You can create an array with a range of elements:

> 일련의 요소가 있는 배열을 생성할 수 있습니다:

```python
np.arange(4)
array([0, 1, 2, 3])
```

And even an array that contains a range of evenly spaced intervals. To do this, you will specify the first number, last number, and the step size.

> 그리고 균등하게 간격을 둔 범위를 포함하는 배열도 가능합니다. 이를 위해 첫 번째 숫자, 마지막 숫자 및 단계 크기를 지정합니다.

```python
np.arange(2, 9, 2)
array([2, 4, 6, 8])
```

You can also use np.linspace() to create an array with values that are spaced linearly in a specified interval:

> 또한 np.linspace()를 사용하여 지정된 간격에서 선형으로 간격이 있는 값을 가진 배열을 생성할 수 있습니다:

```python
np.linspace(0, 10, num=5)
array([ 0. ,  2.5,  5. ,  7.5, 10. ])
```

### Specifying your data type
### 데이터 유형 지정하기

While the default data type is floating point (np.float64), you can explicitly specify which data type you want using the dtype keyword.

> 기본 데이터 유형은 부동 소수점(np.float64)이지만, dtype 키워드를 사용하여 원하는 데이터 유형을 명시적으로 지정할 수 있습니다.

```python
x = np.ones(2, dtype=np.int64)
x
array([1, 1])
```

Learn more about creating arrays here

> 배열 생성에 대해 자세히 알아보려면 여기를 참조하세요.

## Adding, removing, and sorting elements
## 요소 추가, 제거 및 정렬

This section covers np.sort(), np.concatenate()

> 이 섹션은 np.sort(), np.concatenate() 함수를 다룹니다.

Sorting an array is simple with np.sort(). You can specify the axis, kind, and order when you call the function.

> np.sort()로 배열을 정렬하는 것은 간단합니다. 함수를 호출할 때 축(axis), 종류(kind) 및 순서(order)를 지정할 수 있습니다.

If you start with this array:

> 만약 이런 배열로 시작한다면:

```python
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
```

You can quickly sort the numbers in ascending order with:

> 다음과 같이 숫자를 오름차순으로 빠르게 정렬할 수 있습니다:

```python
np.sort(arr)
array([1, 2, 3, 4, 5, 6, 7, 8])
```

In addition to sort, which returns a sorted copy of an array, you can use:

* argsort, which is an indirect sort along a specified axis,
* lexsort, which is an indirect stable sort on multiple keys,
* searchsorted, which will find elements in a sorted array, and
* partition, which is a partial sort.

> sort 외에도 배열의 정렬된 복사본을 반환하는 것 외에도 다음을 사용할 수 있습니다:
>
> * argsort: 지정된 축을 따라 간접 정렬합니다.
> * lexsort: 여러 키에 대한 간접적인 안정 정렬입니다.
> * searchsorted: 정렬된 배열에서 요소를 찾습니다.
> * partition: 부분 정렬입니다.

To read more about sorting an array, see: sort.

> 배열 정렬에 대해 자세히 알아보려면 sort를 참조하세요.

If you start with these arrays:

> 만약 이런 배열들로 시작한다면:

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
```

You can concatenate them with np.concatenate().

> np.concatenate()로 연결할 수 있습니다.

```python
np.concatenate((a, b))
array([1, 2, 3, 4, 5, 6, 7, 8])
```

Or, if you start with these arrays:

> 또는 이런 배열들로 시작한다면:

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
```

You can concatenate them with:

> 다음과 같이 연결할 수 있습니다:

```python
np.concatenate((x, y), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
```

In order to remove elements from an array, it's simple to use indexing to select the elements that you want to keep.

> 배열에서 요소를 제거하려면 유지하려는 요소를 선택하는 인덱싱을 사용하는 것이 간단합니다.

To read more about concatenate, see: concatenate.

> concatenate에 대해 자세히 알아보려면 concatenate를 참조하세요.

## How do you know the shape and size of an array?
## 배열의 형태와 크기를 어떻게 알 수 있나요?

This section covers ndarray.ndim, ndarray.size, ndarray.shape

> 이 섹션은 ndarray.ndim, ndarray.size, ndarray.shape를 다룹니다.

ndarray.ndim will tell you the number of axes, or dimensions, of the array.

> ndarray.ndim은 배열의 축 또는 차원의 수를 알려줍니다.

ndarray.size will tell you the total number of elements of the array. This is the product of the elements of the array's shape.

> ndarray.size는 배열의 총 요소 수를 알려줍니다. 이는 배열 형태의 요소들의 곱입니다.

ndarray.shape will display a tuple of integers that indicate the number of elements stored along each dimension of the array. If, for example, you have a 2-D array with 2 rows and 3 columns, the shape of your array is (2, 3).

> ndarray.shape는 배열의 각 차원을 따라 저장된 요소의 수를 나타내는 정수 튜플을 표시합니다. 예를 들어, 2개의 행과 3개의 열이 있는 2차원 배열이 있는 경우 배열의 shape는 (2, 3)입니다.

For example, if you create this array:

> 예를 들어, 이 배열을 생성하면:

```python
array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])
```

To find the number of dimensions of the array, run:

> 배열의 차원 수를 찾으려면 다음을 실행하세요:

```python
array_example.ndim
3
```

To find the total number of elements in the array, run:

> 배열의 총 요소 수를 찾으려면 다음을 실행하세요:

```python
array_example.size
24
```

And to find the shape of your array, run:

> 그리고 배열의 형태를 찾으려면 다음을 실행하세요:

```python
array_example.shape
(3, 2, 4)
```

## Can you reshape an array?
## 배열 형태를 바꿀 수 있나요?

This section covers arr.reshape()

> 이 섹션은 arr.reshape()를 다룹니다.

Yes!

> 네!

Using arr.reshape() will give a new shape to an array without changing the data. Just remember that when you use the reshape method, the array you want to produce needs to have the same number of elements as the original array. If you start with an array with 12 elements, you'll need to make sure that your new array also has a total of 12 elements.

> arr.reshape()를 사용하면 데이터를 변경하지 않고 배열에 새로운 형태를 부여할 수 있습니다. 그러나 reshape 메서드를 사용할 때는 생성하려는 배열이 원본 배열과 동일한 수의 요소를 가져야 한다는 점을 기억하세요. 12개의 요소가 있는 배열로 시작하면, 새 배열도 총 12개의 요소를 가지도록 해야 합니다.

If you start with this array:

> 이런 배열로 시작한다면:

```python
a = np.arange(6)
print(a)
[0 1 2 3 4 5]
```

You can use reshape() to reshape your array. For example, you can reshape this array to an array with three rows and two columns:

> reshape()를 사용하여 배열의 형태를 바꿀 수 있습니다. 예를 들어, 이 배열을 세 개의 행과 두 개의 열을 가진 배열로 재구성할 수 있습니다:

```python
b = a.reshape(3, 2)
print(b)
[[0 1]
 [2 3]
 [4 5]]
```

With np.reshape, you can specify a few optional parameters:

> np.reshape를 사용할 때 몇 가지 선택적 매개변수를 지정할 수 있습니다:

```python
np.reshape(a, shape=(1, 6), order='C')
array([[0, 1, 2, 3, 4, 5]])
```

a is the array to be reshaped.

shape is the new shape you want. You can specify an integer or a tuple of integers. If you specify an integer, the result will be an array of that length. The shape should be compatible with the original shape.

order: C means to read/write the elements using C-like index order, F means to read/write the elements using Fortran-like index order, A means to read/write the elements in Fortran-like index order if a is Fortran contiguous in memory, C-like order otherwise. (This is an optional parameter and doesn't need to be specified.)

> a는 재구성할 배열입니다.
> 
> shape는 원하는 새 모양입니다. 정수나 정수의 튜플을 지정할 수 있습니다. 정수를 지정하면 결과는 해당 길이의 배열이 됩니다. 모양은 원래 모양과 호환되어야 합니다.
> 
> order: C는 C와 같은 인덱스 순서로 요소를 읽고/쓰는 것을 의미하고, F는 Fortran과 같은 인덱스 순서로 요소를 읽고/쓰는 것을 의미하며, A는 a가 메모리에서 Fortran 연속적인 경우 Fortran과 같은 순서로 요소를 읽고/쓰고, 그렇지 않으면 C와 같은 순서로 읽고/쓰는 것을 의미합니다. (이는 선택적 매개변수이며 지정할 필요가 없습니다.)

If you want to learn more about C and Fortran order, you can read more about the internal organization of NumPy arrays here. Essentially, C and Fortran orders have to do with how indices correspond to the order the array is stored in memory. In Fortran, when moving through the elements of a two-dimensional array as it is stored in memory, the first index is the most rapidly varying index. As the first index moves to the next row as it changes, the matrix is stored one column at a time. This is why Fortran is thought of as a Column-major language. In C on the other hand, the last index changes the most rapidly. The matrix is stored by rows, making it a Row-major language. What you do for C or Fortran depends on whether it's more important to preserve the indexing convention or not reorder the data.

> C 및 Fortran 순서에 대해 더 알고 싶다면, 여기에서 NumPy 배열의 내부 구성에 대해 더 읽을 수 있습니다. 기본적으로 C와 Fortran 순서는 인덱스가 메모리에 저장된 배열 순서와 어떻게 대응하는지에 관련됩니다. Fortran에서는 메모리에 저장된 2차원 배열의 요소를 통과할 때 첫 번째 인덱스가 가장 빠르게 변하는 인덱스입니다. 첫 번째 인덱스가 변경됨에 따라 다음 행으로 이동하면, 행렬은 한 번에 한 열씩 저장됩니다. 이것이 Fortran이 열 중심(Column-major) 언어로 여겨지는 이유입니다. 반면 C에서는 마지막 인덱스가 가장 빠르게 변합니다. 행렬은 행별로 저장되므로 행 중심(Row-major) 언어입니다. C나 Fortran을 선택하는 것은 인덱싱 규칙을 유지하는 것이 중요한지 아니면 데이터의 순서를 재배치하지 않는 것이 중요한지에 따라 달라집니다.

Learn more about shape manipulation here.

> 모양 조작에 대해 자세히 알아보려면 여기를 참조하세요.

## How to convert a 1D array into a 2D array (how to add a new axis to an array)
## 1D 배열을 2D 배열로 변환하는 방법 (배열에 새 축을 추가하는 방법)

This section covers np.newaxis, np.expand_dims

> 이 섹션은 np.newaxis, np.expand_dims를 다룹니다.

You can use np.newaxis and np.expand_dims to increase the dimensions of your existing array.

> 기존 배열의 차원을 늘리기 위해 np.newaxis와 np.expand_dims를 사용할 수 있습니다.

Using np.newaxis will increase the dimensions of your array by one dimension when used once. This means that a 1D array will become a 2D array, a 2D array will become a 3D array, and so on.

> np.newaxis를 한 번 사용하면 배열의 차원이 하나 증가합니다. 이는 1D 배열이 2D 배열이 되고, 2D 배열이 3D 배열이 되는 등의 의미입니다.

For example, if you start with this array:

> 예를 들어, 이런 배열로 시작한다면:

```python
a = np.array([1, 2, 3, 4, 5, 6])
a.shape
(6,)
```

You can use np.newaxis to add a new axis:

> np.newaxis를 사용하여 새 축을 추가할 수 있습니다:

```python
a2 = a[np.newaxis, :]
a2.shape
(1, 6)
```

You can explicitly convert a 1D array to either a row vector or a column vector using np.newaxis. For example, you can convert a 1D array to a row vector by inserting an axis along the first dimension:

> np.newaxis를 사용하여 1D 배열을 행 벡터 또는 열 벡터로 명시적으로 변환할 수 있습니다. 예를 들어, 첫 번째 차원을 따라 축을 삽입하여 1D 배열을 행 벡터로 변환할 수 있습니다:

```python
row_vector = a[np.newaxis, :]
row_vector.shape
(1, 6)
```

Or, for a column vector, you can insert an axis along the second dimension:

> 또는 열 벡터의 경우 두 번째 차원을 따라 축을 삽입할 수 있습니다:

```python
col_vector = a[:, np.newaxis]
col_vector.shape
(6, 1)
```

You can also expand an array by inserting a new axis at a specified position with np.expand_dims.

> np.expand_dims를 사용하여 지정된 위치에 새 축을 삽입하여 배열을 확장할 수도 있습니다.

For example, if you start with this array:

> 예를 들어, 이런 배열로 시작한다면:

```python
a = np.array([1, 2, 3, 4, 5, 6])
a.shape
(6,)
```

You can use np.expand_dims to add an axis at index position 1 with:

> np.expand_dims를 사용하여 인덱스 위치 1에 축을 추가할 수 있습니다:

```python
b = np.expand_dims(a, axis=1)
b.shape
(6, 1)
```

You can add an axis at index position 0 with:

> 인덱스 위치 0에 축을 추가할 수 있습니다:

```python
c = np.expand_dims(a, axis=0)
c.shape
(1, 6)
```

Find more information about newaxis here and expand_dims at expand_dims.

> newaxis에 대한 자세한 정보는 여기에서, expand_dims에 대한 정보는 expand_dims에서 확인할 수 있습니다.

## Indexing and slicing
## 인덱싱과 슬라이싱

You can index and slice NumPy arrays in the same ways you can slice Python lists.

> NumPy 배열은 파이썬 리스트를 슬라이싱하는 것과 같은 방식으로 인덱싱하고 슬라이싱할 수 있습니다.

```python
data = np.array([1, 2, 3])

data[1]
2
data[0:2]
array([1, 2])
data[1:]
array([2, 3])
data[-2:]
array([2, 3])
```

You may want to take a section of your array or specific array elements to use in further analysis or additional operations. To do that, you'll need to subset, slice, and/or index your arrays.

> 추가 분석이나 작업에 사용할 배열의 일부 또는 특정 배열 요소를 가져오고 싶을 수 있습니다. 이를 위해 배열을 부분집합화하거나, 슬라이스하거나, 인덱싱해야 합니다.

If you want to select values from your array that fulfill certain conditions, it's straightforward with NumPy.

> NumPy를 사용하면 특정 조건을 충족하는 배열의 값을 선택하는 것이 간단합니다.

For example, if you start with this array:

> 예를 들어, 이런 배열로 시작한다면:

```python
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

You can easily print all of the values in the array that are less than 5.

> 5보다 작은 배열의 모든 값을 쉽게 출력할 수 있습니다.

```python
print(a[a < 5])
[1 2 3 4]
```

You can also select, for example, numbers that are equal to or greater than 5, and use that condition to index an array.

> 또한 5보다 크거나 같은 숫자를 선택하고, 그 조건을 사용하여 배열을 인덱싱할 수 있습니다.

```python
five_up = (a >= 5)
print(a[five_up])
[ 5  6  7  8  9 10 11 12]
```

You can select elements that are divisible by 2:

> 2로 나눌 수 있는 요소를 선택할 수 있습니다:

```python
divisible_by_2 = a[a%2==0]
print(divisible_by_2)
[ 2  4  6  8 10 12]
```

Or you can select elements that satisfy two conditions using the & and | operators:

> 또는 & 및 | 연산자를 사용하여 두 가지 조건을 만족하는 요소를 선택할 수 있습니다:

```python
c = a[(a > 2) & (a < 11)]
print(c)
[ 3  4  5  6  7  8  9 10]
```

You can also make use of the logical operators & and | in order to return boolean values that specify whether or not the values in an array fulfill a certain condition. This can be useful with arrays that contain names or other categorical values.

> 논리 연산자 & 및 |를 사용하여 배열의 값이 특정 조건을 충족하는지 여부를 지정하는 부울 값을 반환할 수도 있습니다. 이는 이름이나 다른 범주형 값을 포함하는 배열에 유용할 수 있습니다.

```python
five_up = (a > 5) | (a == 5)
print(five_up)
[[False False False False]
 [ True  True  True  True]
 [ True  True  True True]]
```

You can also use np.nonzero() to select elements or indices from an array.

> np.nonzero()를 사용하여 배열에서 요소나 인덱스를 선택할 수도 있습니다.

Starting with this array:

> 이런 배열로 시작합니다:

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

You can use np.nonzero() to print the indices of elements that are, for example, less than 5:

> 예를 들어, 5보다 작은 요소의 인덱스를 출력하기 위해 np.nonzero()를 사용할 수 있습니다:

```python
b = np.nonzero(a < 5)
print(b)
(array([0, 0, 0, 0]), array([0, 1, 2, 3]))
```

In this example, a tuple of arrays was returned: one for each dimension. The first array represents the row indices where these values are found, and the second array represents the column indices where the values are found.

> 이 예제에서는 배열의 튜플이 반환되었습니다: 각 차원마다 하나씩입니다. 첫 번째 배열은 이러한 값이 발견되는 행 인덱스를 나타내고, 두 번째 배열은 값이 발견되는 열 인덱스를 나타냅니다.

If you want to generate a list of coordinates where the elements exist, you can zip the arrays, iterate over the list of coordinates, and print them. For example:

> 요소가 존재하는 좌표 목록을 생성하려면 배열을 압축하고, 좌표 목록을 반복하고, 출력할 수 있습니다. 예를 들어:

```python
list_of_coordinates= list(zip(b[0], b[1]))

for coord in list_of_coordinates:
    print(coord)
(np.int64(0), np.int64(0))
(np.int64(0), np.int64(1))
(np.int64(0), np.int64(2))
(np.int64(0), np.int64(3))
```

You can also use np.nonzero() to print the elements in an array that are less than 5 with:

> 또한 np.nonzero()를 사용하여 5보다 작은 배열의 요소를 다음과 같이 출력할 수 있습니다:

```python
print(a[b])
[1 2 3 4]
```

If the element you're looking for doesn't exist in the array, then the returned array of indices will be empty. For example:

> 찾고 있는 요소가 배열에 존재하지 않으면 반환된 인덱스 배열은 비어 있습니다. 예를 들어:

```python
not_there = np.nonzero(a == 42)
print(not_there)
(array([], dtype=int64), array([], dtype=int64))
```

Learn more about indexing and slicing here and here.

Read more about using the nonzero function at: nonzero.

> 인덱싱과 슬라이싱에 대해 더 알아보려면 여기와 여기를 참조하세요.
> 
> nonzero 함수 사용에 대해 자세히 알아보려면 nonzero를 참조하세요.

## How to create an array from existing data
## 기존 데이터에서 배열을 생성하는 방법

This section covers slicing and indexing, np.vstack(), np.hstack(), np.hsplit(), .view(), copy()

> 이 섹션은 슬라이싱과 인덱싱, np.vstack(), np.hstack(), np.hsplit(), .view(), copy()를 다룹니다.

You can easily create a new array from a section of an existing array.

> 기존 배열의 일부에서 새 배열을 쉽게 생성할 수 있습니다.

Let's say you have this array:

> 이런 배열이 있다고 가정해 봅시다:

```python
a = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

You can create a new array from a section of your array any time by specifying where you want to slice your array.

> 배열을 슬라이스하고 싶은 위치를 지정하여 배열의 섹션에서 언제든지 새 배열을 만들 수 있습니다.

```python
arr1 = a[3:8]
arr1
array([4, 5, 6, 7, 8])
```

Here, you grabbed a section of your array from index position 3 through index position 8 but not including position 8 itself.

> 여기서는 인덱스 위치 3부터 인덱스 위치 8까지(위치 8 자체는 포함하지 않음) 배열의 섹션을 가져왔습니다.

Reminder: Array indexes begin at 0. This means the first element of the array is at index 0, the second element is at index 1, and so on.

> 참고: 배열 인덱스는 0부터 시작합니다. 이는 배열의 첫 번째 요소가 인덱스 0에, 두 번째 요소가 인덱스 1에 있다는 것을 의미합니다.

You can also stack two existing arrays, both vertically and horizontally. Let's say you have two arrays, a1 and a2:

> 두 개의 기존 배열을 수직 및 수평으로 쌓을 수도 있습니다. a1과 a2라는 두 배열이 있다고 가정해 봅시다:

```python
a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])
```

You can stack them vertically with vstack:

> vstack을 사용하여 수직으로 쌓을 수 있습니다:

```python
np.vstack((a1, a2))
array([[1, 1],
       [2, 2],
       [3, 3],
       [4, 4]])
```

Or stack them horizontally with hstack:

> 또는 hstack을 사용하여 수평으로 쌓을 수 있습니다:

```python
np.hstack((a1, a2))
array([[1, 1, 3, 3],
       [2, 2, 4, 4]])
```

You can split an array into several smaller arrays using hsplit. You can specify either the number of equally shaped arrays to return or the columns after which the division should occur.

> hsplit을 사용하여 배열을 여러 개의 작은 배열로 분할할 수 있습니다. 반환할 동일한 형태의 배열 수 또는 분할이 발생해야 하는 열을 지정할 수 있습니다.

Let's say you have this array:

> 이런 배열이 있다고 가정해 봅시다:

```python
x = np.arange(1, 25).reshape(2, 12)
x
array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
       [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])
```

If you wanted to split this array into three equally shaped arrays, you would run:

> 이 배열을 동일한 형태의 세 배열로 분할하려면 다음을 실행하면 됩니다:

```python
np.hsplit(x, 3)
  [array([[ 1,  2,  3,  4],
         [13, 14, 15, 16]]), array([[ 5,  6,  7,  8],
         [17, 18, 19, 20]]), array([[ 9, 10, 11, 12],
         [21, 22, 23, 24]])]
```

If you wanted to split your array after the third and fourth column, you'd run:

> 세 번째 열과 네 번째 열 이후에 배열을 분할하려면 다음을 실행하면 됩니다:

```python
np.hsplit(x, (3, 4))
  [array([[ 1,  2,  3],
         [13, 14, 15]]), array([[ 4],
         [16]]), array([[ 5,  6,  7,  8,  9, 10, 11, 12],
         [17, 18, 19, 20, 21, 22, 23, 24]])]
```

Learn more about stacking and splitting arrays here.

> 배열 쌓기 및 분할에 대해 자세히 알아보려면 여기를 참조하세요.

You can use the view method to create a new array object that looks at the same data as the original array (a shallow copy).

> view 메서드를 사용하여 원본 배열과 동일한 데이터를 보는 새 배열 객체(얕은 복사)를 생성할 수 있습니다.

Views are an important NumPy concept! NumPy functions, as well as operations like indexing and slicing, will return views whenever possible. This saves memory and is faster (no copy of the data has to be made). However it's important to be aware of this - modifying data in a view also modifies the original array!

> 뷰는 중요한 NumPy 개념입니다! NumPy 함수와 인덱싱 및 슬라이싱과 같은 작업은 가능한 경우 항상 뷰를 반환합니다. 이는 메모리를 절약하고 더 빠릅니다(데이터의 복사본을 만들 필요가 없음). 그러나 뷰의 데이터를 수정하면 원본 배열도 수정된다는 점을 인식하는 것이 중요합니다!

Let's say you create this array:

> 이런 배열을 만든다고 가정해 봅시다:

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

Now we create an array b1 by slicing a and modify the first element of b1. This will modify the corresponding element in a as well!

> 이제 a를 슬라이싱하여 배열 b1을 만들고 b1의 첫 번째 요소를 수정해 보겠습니다. 이렇게 하면 a의 해당 요소도 수정됩니다!

```python
b1 = a[0, :]
b1
array([1, 2, 3, 4])
b1[0] = 99
b1
array([99,  2,  3,  4])
a
array([[99,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])
```

Using the copy method will make a complete copy of the array and its data (a deep copy). To use this on your array, you could run:

> copy 메서드를 사용하면 배열과 그 데이터의 완전한 복사본(깊은 복사)을 만들 수 있습니다. 배열에 이 방법을 사용하려면 다음을 실행하면 됩니다:

```python
b2 = a.copy()
```

## Basic array operations
## 기본 배열 연산

This section covers addition, subtraction, multiplication, division, and more

> 이 섹션에서는 덧셈, 뺄셈, 곱셈, 나눗셈 등을 다룹니다.

Once you've created your arrays, you can start to work with them. Let's say, for example, that you've created two arrays, one called "data" and one called "ones"

> 배열을 생성한 후에는 이를 가지고 작업을 시작할 수 있습니다. 예를 들어, "data"와 "ones"라는 두 배열을 생성했다고 가정해 봅시다.

You can add the arrays together with the plus sign.

> 플러스 기호로 배열을 더할 수 있습니다.

```python
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
data + ones
array([2, 3])
```

You can, of course, do more than just addition!

> 물론 덧셈 외에도 더 많은 연산을 수행할 수 있습니다!

```python
data - ones
array([0, 1])
data * data
array([1, 4])
data / data
array([1., 1.])
```

Basic operations are simple with NumPy. If you want to find the sum of the elements in an array, you'd use sum(). This works for 1D arrays, 2D arrays, and arrays in higher dimensions.

> NumPy를 사용하면 기본 연산이 간단합니다. 배열의 요소들의 합을 구하고 싶다면 sum()을 사용하면 됩니다. 이는 1D 배열, 2D 배열 및 더 높은 차원의 배열에서도 작동합니다.

```python
a = np.array([1, 2, 3, 4])

a.sum()
10
```

To add the rows or the columns in a 2D array, you would specify the axis.

> 2D 배열에서 행이나 열을 더하려면 축(axis)을 지정해야 합니다.

If you start with this array:

> 이런 배열로 시작한다면:

```python
b = np.array([[1, 1], [2, 2]])
```

You can sum over the axis of rows with:

> 다음과 같이 행의 축을 따라 합계를 구할 수 있습니다:

```python
b.sum(axis=0)
array([3, 3])
```

You can sum over the axis of columns with:

> 다음과 같이 열의 축을 따라 합계를 구할 수 있습니다:

```python
b.sum(axis=1)
array([2, 4])
```

## Broadcasting
## 브로드캐스팅

There are times when you might want to carry out an operation between an array and a single number (also called an operation between a vector and a scalar) or between arrays of two different sizes. For example, your array (we'll call it "data") might contain information about distance in miles but you want to convert the information to kilometers. You can perform this operation with:

> 배열과 단일 숫자(벡터와 스칼라 간의 연산이라고도 함) 사이 또는 크기가 다른 두 배열 사이에 연산을 수행하고 싶을 때가 있습니다. 예를 들어, 배열("data"라고 부르겠습니다)에 마일 단위로 된 거리 정보가 포함되어 있지만 이 정보를 킬로미터로 변환하고 싶을 수 있습니다. 이 연산은 다음과 같이 수행할 수 있습니다:

```python
data = np.array([1.0, 2.0])
data * 1.6
array([1.6, 3.2])
```

NumPy understands that the multiplication should happen with each cell. That concept is called broadcasting. Broadcasting is a mechanism that allows NumPy to perform operations on arrays of different shapes. The dimensions of your array must be compatible, for example, when the dimensions of both arrays are equal or when one of them is 1. If the dimensions are not compatible, you will get a ValueError.

> NumPy는 곱셈이 각 셀마다 발생해야 한다는 것을 이해합니다. 이 개념을 브로드캐스팅이라고 합니다. 브로드캐스팅은 NumPy가 서로 다른 형태의 배열에 대해 연산을 수행할 수 있게 해주는 메커니즘입니다. 배열의 차원은 호환되어야 합니다. 예를 들어, 두 배열의 차원이 동일하거나 그 중 하나가 1일 때입니다. 차원이 호환되지 않으면 ValueError가 발생합니다.

Learn more about broadcasting here.

> 브로드캐스팅에 대해 더 알아보려면 여기를 참조하세요.

## More useful array operations
## 더 유용한 배열 연산

This section covers maximum, minimum, sum, mean, product, standard deviation, and more

> 이 섹션에서는 최대값, 최소값, 합계, 평균, 곱, 표준편차 등을 다룹니다.

NumPy also performs aggregation functions. In addition to min, max, and sum, you can easily run mean to get the average, prod to get the result of multiplying the elements together, std to get the standard deviation, and more.

> NumPy는 또한 집계 함수를 수행합니다. min, max, sum 외에도 평균을 구하기 위한 mean, 요소들을 곱한 결과를 얻기 위한 prod, 표준편차를 구하기 위한 std 등을 쉽게 실행할 수 있습니다.

```python
data.max()
2.0
data.min()
1.0
data.sum()
3.0
```

Let's start with this array, called "a"

> "a"라는 이 배열로 시작해 보겠습니다.

```python
a = np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
              [0.54627315, 0.05093587, 0.40067661, 0.55645993],
              [0.12697628, 0.82485143, 0.26590556, 0.56917101]])
```

It's very common to want to aggregate along a row or column. By default, every NumPy aggregation function will return the aggregate of the entire array. To find the sum or the minimum of the elements in your array, run:

> 행이나 열을 따라 집계하는 것은 매우 일반적입니다. 기본적으로 모든 NumPy 집계 함수는 전체 배열의 집계 결과를 반환합니다. 배열 요소의 합계나 최소값을 찾으려면 다음을 실행하세요:

```python
a.sum()
4.8595784
```

Or:

> 또는:

```python
a.min()
0.05093587
```

You can specify on which axis you want the aggregation function to be computed. For example, you can find the minimum value within each column by specifying axis=0.

> 집계 함수를 계산할 축을 지정할 수 있습니다. 예를 들어, axis=0을 지정하여 각 열 내의 최소값을 찾을 수 있습니다.

```python
a.min(axis=0)
array([0.12697628, 0.05093587, 0.26590556, 0.5510652 ])
```

The four values listed above correspond to the number of columns in your array. With a four-column array, you will get four values as your result.

> 위에 나열된 네 개의 값은 배열의 열 수에 해당합니다. 네 개의 열이 있는 배열의 경우, 결과로 네 개의 값을 얻게 됩니다.

## Creating matrices
## 행렬 생성하기

You can pass Python lists of lists to create a 2-D array (or "matrix") to represent them in NumPy.

> NumPy에서 2차원 배열("행렬")을 생성하기 위해 파이썬 리스트의 리스트를 전달할 수 있습니다.

```python
data = np.array([[1, 2], [3, 4], [5, 6]])
data
array([[1, 2],
       [3, 4],
       [5, 6]])
```

Indexing and slicing operations are useful when you're manipulating matrices:

> 행렬을 조작할 때 인덱싱과 슬라이싱 연산이 유용합니다:

```python
data[0, 1]
2
data[1:3]
array([[3, 4],
       [5, 6]])
data[0:2, 0]
array([1, 3])
```

You can aggregate matrices the same way you aggregated vectors:

> 벡터를 집계한 것과 같은 방식으로 행렬을 집계할 수 있습니다:

```python
data.max()
6
data.min()
1
data.sum()
21
```

You can aggregate all the values in a matrix and you can aggregate them across columns or rows using the axis parameter. To illustrate this point, let's look at a slightly modified dataset:

> 행렬의 모든 값을 집계할 수 있으며, axis 매개변수를 사용하여 열이나 행에 걸쳐 집계할 수도 있습니다. 이 점을 설명하기 위해 약간 수정된 데이터셋을 살펴보겠습니다:

```python
data = np.array([[1, 2], [5, 3], [4, 6]])
data
array([[1, 2],
       [5, 3],
       [4, 6]])
data.max(axis=0)
array([5, 6])
data.max(axis=1)
array([2, 5, 6])
```

Once you've created your matrices, you can add and multiply them using arithmetic operators if you have two matrices that are the same size.

> 행렬을 생성한 후에는 같은 크기의 두 행렬이 있는 경우 산술 연산자를 사용하여 더하고 곱할 수 있습니다.

```python
data = np.array([[1, 2], [3, 4]])
ones = np.array([[1, 1], [1, 1]])
data + ones
array([[2, 3],
       [4, 5]])
```

You can do these arithmetic operations on matrices of different sizes, but only if one matrix has only one column or one row. In this case, NumPy will use its broadcast rules for the operation.

> 크기가 다른 행렬에서도 이러한 산술 연산을 수행할 수 있지만, 한 행렬이 하나의 열이나 하나의 행만 가지고 있는 경우에만 가능합니다. 이 경우 NumPy는 작업에 브로드캐스트 규칙을 사용합니다.

```python
data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
data + ones_row
array([[2, 3],
       [4, 5],
       [6, 7]])
```

Be aware that when NumPy prints N-dimensional arrays, the last axis is looped over the fastest while the first axis is the slowest. For instance:

> NumPy가 N차원 배열을 출력할 때 마지막 축이 가장 빠르게 반복되고 첫 번째 축이 가장 느리게 반복된다는 점에 유의하세요. 예를 들면:

```python
np.ones((4, 3, 2))
array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])
```

There are often instances where we want NumPy to initialize the values of an array. NumPy offers functions like ones() and zeros(), and the random.Generator class for random number generation for that. All you need to do is pass in the number of elements you want it to generate:

> 배열의 값을 초기화하려는 경우가 자주 있습니다. NumPy는 ones()와 zeros() 같은 함수와 난수 생성을 위한 random.Generator 클래스를 제공합니다. 생성하려는 요소 수를 전달하기만 하면 됩니다:

```python
np.ones(3)
array([1., 1., 1.])
np.zeros(3)
array([0., 0., 0.])
rng = np.random.default_rng()  # the simplest way to generate random numbers
rng.random(3) 
array([0.63696169, 0.26978671, 0.04097352])
```

You can also use ones(), zeros(), and random() to create a 2D array if you give them a tuple describing the dimensions of the matrix:

> 행렬의 차원을 설명하는 튜플을 제공하면 ones(), zeros() 및 random()을 사용하여 2D 배열을 생성할 수도 있습니다:

```python
np.ones((3, 2))
array([[1., 1.],
       [1., 1.],
       [1., 1.]])
np.zeros((3, 2))
array([[0., 0.],
       [0., 0.],
       [0., 0.]])
rng.random((3, 2)) 
array([[0.01652764, 0.81327024],
       [0.91275558, 0.60663578],
       [0.72949656, 0.54362499]])  # may vary
```

Read more about creating arrays, filled with 0's, 1's, other values or uninitialized, at array creation routines.

> 0, 1, 기타 값 또는 초기화되지 않은 값으로 채워진 배열 생성에 대해 자세히 알아보려면 배열 생성 루틴을 참조하세요.

## Generating random numbers
## 난수 생성하기

The use of random number generation is an important part of the configuration and evaluation of many numerical and machine learning algorithms. Whether you need to randomly initialize weights in an artificial neural network, split data into random sets, or randomly shuffle your dataset, being able to generate random numbers (actually, repeatable pseudo-random numbers) is essential.

> 난수 생성의 사용은 많은 수치 및 기계 학습 알고리즘의 구성 및 평가에서 중요한 부분입니다. 인공 신경망에서 가중치를 무작위로 초기화하거나, 데이터를 무작위 세트로 분할하거나, 데이터셋을 무작위로 섞어야 하는 경우 난수(실제로는 반복 가능한 의사 난수)를 생성할 수 있는 능력이 필수적입니다.

With Generator.integers, you can generate random integers from low (remember that this is inclusive with NumPy) to high (exclusive). You can set endpoint=True to make the high number inclusive.

> Generator.integers를 사용하면 하한값(NumPy에서는 포함됨)부터 상한값(포함되지 않음)까지의 무작위 정수를 생성할 수 있습니다. endpoint=True로 설정하여 상한값을 포함시킬 수 있습니다.

You can generate a 2 x 4 array of random integers between 0 and 4 with:

> 0에서 4 사이의 무작위 정수로 이루어진 2 x 4 배열을 다음과 같이 생성할 수 있습니다:

```python
rng.integers(5, size=(2, 4)) 
array([[2, 1, 1, 0],
       [0, 0, 0, 4]])  # may vary
```

## How to get unique items and counts
## 고유 항목 및 개수 얻기

This section covers np.unique()

> 이 섹션은 np.unique()를 다룹니다.

You can find the unique elements in an array easily with np.unique.

> np.unique를 사용하여 배열에서 고유한 요소를 쉽게 찾을 수 있습니다.

For example, if you start with this array:

> 예를 들어, 이런 배열로 시작한다면:

```python
a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
```

you can use np.unique to print the unique values in your array:

> np.unique를 사용하여 배열의 고유 값을 출력할 수 있습니다:

```python
unique_values = np.unique(a)
print(unique_values)
[11 12 13 14 15 16 17 18 19 20]
```

To get the indices of unique values in a NumPy array (an array of first index positions of unique values in the array), just pass the return_index argument in np.unique() as well as your array.

> NumPy 배열에서 고유 값의 인덱스(배열에서 고유 값의 첫 번째 인덱스 위치 배열)를 얻으려면, 배열과 함께 np.unique()에 return_index 인자를 전달하면 됩니다.

```python
unique_values, indices_list = np.unique(a, return_index=True)
print(indices_list)
[ 0  2  3  4  5  6  7 12 13 14]
```

You can pass the return_counts argument in np.unique() along with your array to get the frequency count of unique values in a NumPy array.

> 배열과 함께 np.unique()에 return_counts 인자를 전달하여 NumPy 배열에서 고유 값의 빈도 수를 얻을 수 있습니다.

```python
unique_values, occurrence_count = np.unique(a, return_counts=True)
print(occurrence_count)
[3 2 2 2 1 1 1 1 1 1]
```

This also works with 2D arrays! If you start with this array:

> 이것은 2D 배열에서도 작동합니다! 이런 배열로 시작한다면:

```python
a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
```

You can find unique values with:

> 다음과 같이 고유 값을 찾을 수 있습니다:

```python
unique_values = np.unique(a_2d)
print(unique_values)
[ 1  2  3  4  5  6  7  8  9 10 11 12]
```

If the axis argument isn't passed, your 2D array will be flattened.

> axis 인자가 전달되지 않으면 2D 배열은 평탄화됩니다.

If you want to get the unique rows or columns, make sure to pass the axis argument. To find the unique rows, specify axis=0 and for columns, specify axis=1.

> 고유한 행이나 열을 얻으려면 axis 인자를 전달해야 합니다. 고유한 행을 찾으려면 axis=0을 지정하고, 열은 axis=1을 지정합니다.

```python
unique_rows = np.unique(a_2d, axis=0)
print(unique_rows)
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
```

To get the unique rows, index position, and occurrence count, you can use:

> 고유한 행, 인덱스 위치 및 발생 횟수를 얻으려면 다음을 사용할 수 있습니다:

```python
unique_rows, indices, occurrence_count = np.unique(
     a_2d, axis=0, return_counts=True, return_index=True)
print(unique_rows)
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
print(indices)
[0 1 2]
print(occurrence_count)
[2 1 1]
```

To learn more about finding the unique elements in an array, see unique.

> 배열에서 고유 요소를 찾는 방법에 대해 자세히 알아보려면 unique를 참조하세요.

## Transposing and reshaping a matrix
## 행렬 전치 및 리셰이핑

This section covers arr.reshape(), arr.transpose(), arr.T

> 이 섹션은 arr.reshape(), arr.transpose(), arr.T를 다룹니다.

It's common to need to transpose your matrices. NumPy arrays have the property T that allows you to transpose a matrix.

> 행렬을 전치해야 하는 경우가 흔합니다. NumPy 배열에는 행렬을 전치할 수 있는 T 속성이 있습니다.

You may also need to switch the dimensions of a matrix. This can happen when, for example, you have a model that expects a certain input shape that is different from your dataset. This is where the reshape method can be useful. You simply need to pass in the new dimensions that you want for the matrix.

> 또한 행렬의 차원을 전환해야 할 수도 있습니다. 예를 들어, 데이터셋과 다른 특정 입력 형태를 기대하는 모델이 있을 때 이런 일이 발생할 수 있습니다. 이때 reshape 메서드가 유용합니다. 행렬에 원하는 새 차원을 전달하기만 하면 됩니다.

```python
data.reshape(2, 3)
array([[1, 2, 3],
       [4, 5, 6]])
data.reshape(3, 2)
array([[1, 2],
       [3, 4],
       [5, 6]])
```

You can also use .transpose() to reverse or change the axes of an array according to the values you specify.

> 또한 .transpose()를 사용하여 지정한 값에 따라 배열의 축을 반전하거나 변경할 수 있습니다.

If you start with this array:

> 이런 배열로 시작한다면:

```python
arr = np.arange(6).reshape((2, 3))
arr
array([[0, 1, 2],
       [3, 4, 5]])
```

You can transpose your array with arr.transpose().

> arr.transpose()로 배열을 전치할 수 있습니다.

```python
arr.transpose()
array([[0, 3],
       [1, 4],
       [2, 5]])
```

You can also use arr.T:

> arr.T를 사용할 수도 있습니다:

```python
arr.T
array([[0, 3],
       [1, 4],
       [2, 5]])
```

To learn more about transposing and reshaping arrays, see transpose and reshape.

> 배열의 전치 및 리셰이핑에 대해 자세히 알아보려면 transpose와 reshape를 참조하세요.

## How to reverse an array
## 배열 뒤집기

This section covers np.flip()

> 이 섹션은 np.flip()을 다룹니다.

NumPy's np.flip() function allows you to flip, or reverse, the contents of an array along an axis. When using np.flip(), specify the array you would like to reverse and the axis. If you don't specify the axis, NumPy will reverse the contents along all of the axes of your input array.

> NumPy의 np.flip() 함수를 사용하면 축을 따라 배열의 내용을 뒤집거나 반전시킬 수 있습니다. np.flip()을 사용할 때는 뒤집고 싶은 배열과 축을 지정합니다. 축을 지정하지 않으면 NumPy는 입력 배열의 모든 축을 따라 내용을 뒤집습니다.

### Reversing a 1D array
### 1D 배열 뒤집기

If you begin with a 1D array like this one:

> 다음과 같은 1D 배열로 시작한다면:

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
```

You can reverse it with:

> 다음과 같이 뒤집을 수 있습니다:

```python
reversed_arr = np.flip(arr)
```

If you want to print your reversed array, you can run:

> 뒤집힌 배열을 출력하고 싶다면 다음을 실행할 수 있습니다:

```python
print('Reversed Array: ', reversed_arr)
Reversed Array:  [8 7 6 5 4 3 2 1]
```

### Reversing a 2D array
### 2D 배열 뒤집기

A 2D array works much the same way.

> 2D 배열도 거의 동일한 방식으로 작동합니다.

If you start with this array:

> 이런 배열로 시작한다면:

```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

You can reverse the content in all of the rows and all of the columns with:

> 모든 행과 모든 열의 내용을 다음과 같이 뒤집을 수 있습니다:

```python
reversed_arr = np.flip(arr_2d)
print(reversed_arr)
[[12 11 10  9]
 [ 8  7  6  5]
 [ 4  3  2  1]]
```

You can easily reverse only the rows with:

> 행만 쉽게 뒤집을 수 있습니다:

```python
reversed_arr_rows = np.flip(arr_2d, axis=0)
print(reversed_arr_rows)
[[ 9 10 11 12]
 [ 5  6  7  8]
 [ 1  2  3  4]]
```

Or reverse only the columns with:

> 또는 열만 뒤집을 수 있습니다:

```python
reversed_arr_columns = np.flip(arr_2d, axis=1)
print(reversed_arr_columns)
[[ 4  3  2  1]
 [ 8  7  6  5]
 [12 11 10  9]]
```

You can also reverse the contents of only one column or row. For example, you can reverse the contents of the row at index position 1 (the second row):

> 하나의 열이나 행의 내용만 뒤집을 수도 있습니다. 예를 들어, 인덱스 위치 1(두 번째 행)의 행 내용을 뒤집을 수 있습니다:

```python
arr_2d[1] = np.flip(arr_2d[1])
print(arr_2d)
[[ 1  2  3  4]
 [ 8  7  6  5]
 [ 9 10 11 12]]
```

You can also reverse the column at index position 1 (the second column):

> 인덱스 위치 1(두 번째 열)의 열도 뒤집을 수 있습니다:

```python
arr_2d[:,1] = np.flip(arr_2d[:,1])
print(arr_2d)
[[ 1 10  3  4]
 [ 8  7  6  5]
 [ 9  2 11 12]]
```

Read more about reversing arrays at flip.

> 배열 뒤집기에 대해 더 자세히 알아보려면 flip을 참조하세요.

## Reshaping and flattening multidimensional arrays
## 다차원 배열 리셰이핑 및 평탄화

This section covers .flatten(), ravel()

> 이 섹션은 .flatten(), ravel()을 다룹니다.

There are two popular ways to flatten an array: .flatten() and .ravel(). The primary difference between the two is that the new array created using ravel() is actually a reference to the parent array (i.e., a "view"). This means that any changes to the new array will affect the parent array as well. Since ravel does not create a copy, it's memory efficient.

> 배열을 평탄화하는 두 가지 인기 있는 방법이 있습니다: .flatten()과 .ravel(). 둘의 주요 차이점은 ravel()을 사용하여 생성된 새 배열이 실제로 부모 배열에 대한 참조(즉, "뷰")라는 것입니다. 이는 새 배열의 변경 사항이 부모 배열에도 영향을 미친다는 의미입니다. ravel은 복사본을 만들지 않으므로 메모리 효율성이 좋습니다.

If you start with this array:

> 이런 배열로 시작한다면:

```python
x = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

You can use flatten to flatten your array into a 1D array.

> flatten을 사용하여 배열을 1D 배열로 평탄화할 수 있습니다.

```python
x.flatten()
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
```

When you use flatten, changes to your new array won't change the parent array.

> flatten을 사용할 때는 새 배열의 변경 사항이 부모 배열을 변경하지 않습니다.

For example:

> 예를 들면:

```python
a1 = x.flatten()
a1[0] = 99
print(x)  # Original array
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
print(a1)  # New array
[99  2  3  4  5  6  7  8  9 10 11 12]
```

But when you use ravel, the changes you make to the new array will affect the parent array.

> 하지만 ravel을 사용할 때는 새 배열에 대한 변경 사항이 부모 배열에 영향을 미칩니다.

For example:

> 예를 들면:

```python
a2 = x.ravel()
a2[0] = 98
print(x)  # Original array
[[98  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
print(a2)  # New array
[98  2  3  4  5  6  7  8  9 10 11 12]
```

Read more about flatten at ndarray.flatten and ravel at ravel.

> flatten에 대해 더 자세히 알아보려면 ndarray.flatten을 참조하고, ravel에 대해서는 ravel을 참조하세요.

## How to access the docstring for more information
## 더 많은 정보를 위한 docstring 접근 방법

This section covers help(), ?, ??

> 이 섹션은 help(), ?, ??를 다룹니다.

When it comes to the data science ecosystem, Python and NumPy are built with the user in mind. One of the best examples of this is the built-in access to documentation. Every object contains the reference to a string, which is known as the docstring. In most cases, this docstring contains a quick and concise summary of the object and how to use it. Python has a built-in help() function that can help you access this information. This means that nearly any time you need more information, you can use help() to quickly find the information that you need.

> 데이터 과학 생태계에 있어서 Python과 NumPy는 사용자를 염두에 두고 설계되었습니다. 이 중 가장 좋은 예는 문서에 대한 내장 접근 기능입니다. 모든 객체는 docstring이라고 알려진 문자열에 대한 참조를 포함하고 있습니다. 대부분의 경우, 이 docstring은 객체와 그 사용법에 대한 빠르고 간결한 요약을 포함하고 있습니다. Python에는 이 정보에 접근하는 데 도움이 되는 내장 help() 함수가 있습니다. 이는 더 많은 정보가 필요할 때마다 help()를 사용하여 필요한 정보를 빠르게 찾을 수 있다는 의미입니다.

For example:

> 예를 들면:

```python
help(max)
Help on built-in function max in module builtins:

max(...)
    max(iterable, *[, default=obj, key=func]) -> value
    max(arg1, arg2, *args, *[, key=func]) -> value

    With a single iterable argument, return its biggest item. The
    default keyword-only argument specifies an object to return if
    the provided iterable is empty.
    With two or more arguments, return the largest argument.
```

Because access to additional information is so useful, IPython uses the ? character as a shorthand for accessing this documentation along with other relevant information. IPython is a command shell for interactive computing in multiple languages. You can find more information about IPython here.

> 추가 정보에 대한 접근이 매우 유용하기 때문에, IPython은 ? 문자를 사용하여 이 문서와 다른 관련 정보에 접근하는 단축키로 사용합니다. IPython은 여러 언어로 대화형 컴퓨팅을 위한 명령 쉘입니다. IPython에 대한 자세한 정보는 여기에서 찾을 수 있습니다.

For example:

> 예를 들면:

```python
max?
max(iterable, *[, default=obj, key=func]) -> value
max(arg1, arg2, *args, *[, key=func]) -> value

With a single iterable argument, return its biggest item. The
default keyword-only argument specifies an object to return if
the provided iterable is empty.
With two or more arguments, return the largest argument.
Type:      builtin_function_or_method
```

You can even use this notation for object methods and objects themselves.

> 이 표기법을 객체 메서드와 객체 자체에도 사용할 수 있습니다.

Let's say you create this array:

> 다음과 같은 배열을 생성했다고 가정해 봅시다:

```python
a = np.array([1, 2, 3, 4, 5, 6])
```

Then you can obtain a lot of useful information (first details about a itself, followed by the docstring of ndarray of which a is an instance):

> 그러면 많은 유용한 정보를 얻을 수 있습니다(먼저 a 자체에 대한 세부 정보, 그 다음에 a가 인스턴스인 ndarray의 docstring):

```python
a?
Type:            ndarray
String form:     [1 2 3 4 5 6]
Length:          6
File:            ~/anaconda3/lib/python3.9/site-packages/numpy/__init__.py
Docstring:       <no docstring>
Class docstring:
ndarray(shape, dtype=float, buffer=None, offset=0,
        strides=None, order=None)

An array object represents a multidimensional, homogeneous array
of fixed-size items.  An associated data-type object describes the
format of each element in the array (its byte-order, how many bytes it
occupies in memory, whether it is an integer, a floating point number,
or something else, etc.)

Arrays should be constructed using `array`, `zeros` or `empty` (refer
to the See Also section below).  The parameters given here refer to
a low-level method (`ndarray(...)`) for instantiating an array.

For more information, refer to the `numpy` module and examine the
methods and attributes of an array.

Parameters
----------
(for the __new__ method; see Notes below)

shape : tuple of ints
        Shape of created array.
...
```

This also works for functions and other objects that you create. Just remember to include a docstring with your function using a string literal (""" """ or ''' ''' around your documentation).

> 이것은 여러분이 생성한 함수 및 기타 객체에도 작동합니다. 함수에 문자열 리터럴(문서 주변에 """ """ 또는 ''' ''')을 사용하여 docstring을 포함하는 것을 잊지 마세요.

For example, if you create this function:

> 예를 들어, 이 함수를 생성하면:

```python
def double(a):
  '''Return a * 2'''
  return a * 2
```

You can obtain information about the function:

> 함수에 대한 정보를 얻을 수 있습니다:

```python
double?
Signature: double(a)
Docstring: Return a * 2
File:      ~/Desktop/<ipython-input-23-b5adf20be596>
Type:      function
```

You can reach another level of information by reading the source code of the object you're interested in. Using a double question mark (??) allows you to access the source code.

> 관심 있는 객체의 소스 코드를 읽어 또 다른 수준의 정보에 도달할 수 있습니다. 이중 물음표(??)를 사용하면 소스 코드에 액세스할 수 있습니다.

For example:

> 예를 들면:

```python
double??
Signature: double(a)
Source:
def double(a):
    '''Return a * 2'''
    return a * 2
File:      ~/Desktop/<ipython-input-23-b5adf20be596>
Type:      function
```

If the object in question is compiled in a language other than Python, using ?? will return the same information as ?. You'll find this with a lot of built-in objects and types, for example:

> 만약 해당 객체가 Python 이외의 언어로 컴파일된 경우, ??를 사용하면 ?와 동일한 정보가 반환됩니다. 이것은 많은 내장 객체와 유형에서 볼 수 있습니다. 예를 들면:

```python
len?
Signature: len(obj, /)
Docstring: Return the number of items in a container.
Type:      builtin_function_or_method
```

and:

> 그리고:

```python
len??
Signature: len(obj, /)
Docstring: Return the number of items in a container.
Type:      builtin_function_or_method
```

have the same output because they were compiled in a programming language other than Python.

> Python 이외의 프로그래밍 언어로 컴파일되었기 때문에 동일한 출력을 가집니다.

## Working with mathematical formulas
## 수학 공식 활용하기

The ease of implementing mathematical formulas that work on arrays is one of the things that make NumPy so widely used in the scientific Python community.

> 배열에 작동하는 수학 공식을 쉽게 구현할 수 있다는 점은 NumPy가 과학 Python 커뮤니티에서 널리 사용되는 이유 중 하나입니다.

For example, this is the mean square error formula (a central formula used in supervised machine learning models that deal with regression):

> 예를 들어, 다음은 평균 제곱 오차 공식(회귀를 다루는 지도 학습 모델에서 사용되는 중심 공식)입니다:

MSE = (1/n) * Σ(predictions - labels)²

Implementing this formula is simple and straightforward in NumPy:

> NumPy에서 이 공식을 구현하는 것은 간단하고 명확합니다:

```python
# MSE implementation
mse = np.mean((predictions - labels)**2)
```

What makes this work so well is that predictions and labels can contain one or a thousand values. They only need to be the same size.

> 이것이 잘 작동하는 이유는 predictions와 labels가 하나 또는 천 개의 값을 포함할 수 있기 때문입니다. 동일한 크기만 되면 됩니다.

You can visualize it this way:

> 이를 다음과 같이 시각화할 수 있습니다:

When both the predictions and labels vectors contain three values, n has a value of three. After we carry out subtractions the values in the vector are squared. Then NumPy sums the values, divides by n, and your result is the error value for that prediction and a score for the quality of the model.

> predictions와 labels 벡터 모두 세 개의 값을 포함할 때, n은 3의 값을 갖습니다. 뺄셈을 수행한 후 벡터의 값이 제곱됩니다. 그런 다음 NumPy는 값들을 합산하고, n으로 나누며, 결과는 해당 예측에 대한 오차 값과 모델의 품질에 대한 점수가 됩니다.

## How to save and load NumPy objects
## NumPy 객체 저장 및 불러오기

This section covers np.save, np.savez, np.savetxt, np.load, np.loadtxt

> 이 섹션은 np.save, np.savez, np.savetxt, np.load, np.loadtxt를 다룹니다.

You will, at some point, want to save your arrays to disk and load them back without having to re-run the code. Fortunately, there are several ways to save and load objects with NumPy. The ndarray objects can be saved to and loaded from the disk files with loadtxt and savetxt functions that handle normal text files, load and save functions that handle NumPy binary files with a .npy file extension, and a savez function that handles NumPy files with a .npz file extension.

> 어느 시점에서는 코드를 다시 실행하지 않고도 배열을 디스크에 저장하고 다시 불러오고 싶을 것입니다. 다행히 NumPy로 객체를 저장하고 불러오는 여러 가지 방법이 있습니다. ndarray 객체는 일반 텍스트 파일을 처리하는 loadtxt와 savetxt 함수, .npy 파일 확장자를 가진 NumPy 바이너리 파일을 처리하는 load와 save 함수, 그리고 .npz 파일 확장자를 가진 NumPy 파일을 처리하는 savez 함수를 사용하여 디스크 파일에 저장하고 불러올 수 있습니다.

The .npy and .npz files store data, shape, dtype, and other information required to reconstruct the ndarray in a way that allows the array to be correctly retrieved, even when the file is on another machine with different architecture.

> .npy 및 .npz 파일은 데이터, 형태, dtype 및 ndarray를 재구성하는 데 필요한 기타 정보를 저장하여 파일이 다른 아키텍처의 머신에 있더라도 배열을 올바르게 검색할 수 있게 합니다.

If you want to store a single ndarray object, store it as a .npy file using np.save. If you want to store more than one ndarray object in a single file, save it as a .npz file using np.savez. You can also save several arrays into a single file in compressed npz format with savez_compressed.

> 단일 ndarray 객체를 저장하려면 np.save를 사용하여 .npy 파일로 저장하세요. 단일 파일에 여러 ndarray 객체를 저장하려면 np.savez를 사용하여 .npz 파일로 저장하세요. savez_compressed를 사용하여 여러 배열을 압축된 npz 형식의 단일 파일로 저장할 수도 있습니다.

It's easy to save and load an array with np.save(). Just make sure to specify the array you want to save and a file name. For example, if you create this array:

> np.save()로 배열을 저장하고 불러오는 것은 쉽습니다. 저장하려는 배열과 파일 이름을 지정하기만 하면 됩니다. 예를 들어, 이 배열을 생성한다면:

```python
a = np.array([1, 2, 3, 4, 5, 6])
```

You can save it as "filename.npy" with:

> "filename.npy"로 저장할 수 있습니다:

```python
np.save('filename', a)
```

You can use np.load() to reconstruct your array.

> np.load()를 사용하여 배열을 재구성할 수 있습니다.

```python
b = np.load('filename.npy')
```

If you want to check your array, you can run:

> 배열을 확인하고 싶다면 다음을 실행할 수 있습니다:

```python
print(b)
[1 2 3 4 5 6]
```

You can save a NumPy array as a plain text file like a .csv or .txt file with np.savetxt.

> np.savetxt를 사용하여 NumPy 배열을 .csv나 .txt 파일과 같은 일반 텍스트 파일로 저장할 수 있습니다.

For example, if you create this array:

> 예를 들어, 이 배열을 생성한다면:

```python
csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
```

You can easily save it as a .csv file with the name "new_file.csv" like this:

> "new_file.csv"라는 이름의 .csv 파일로 쉽게 저장할 수 있습니다:

```python
np.savetxt('new_file.csv', csv_arr)
```

You can quickly and easily load your saved text file using loadtxt():

> loadtxt()를 사용하여 저장된 텍스트 파일을 빠르고 쉽게 불러올 수 있습니다:

```python
np.loadtxt('new_file.csv')
array([1., 2., 3., 4., 5., 6., 7., 8.])
```

The savetxt() and loadtxt() functions accept additional optional parameters such as header, footer, and delimiter. While text files can be easier for sharing, .npy and .npz files are smaller and faster to read. If you need more sophisticated handling of your text file (for example, if you need to work with lines that contain missing values), you will want to use the genfromtxt function.

> savetxt() 및 loadtxt() 함수는 header, footer 및 delimiter와 같은 추가 선택적 매개변수를 허용합니다. 텍스트 파일은 공유하기 더 쉬울 수 있지만, .npy 및 .npz 파일은 더 작고 읽기 더 빠릅니다. 텍스트 파일의 더 복잡한 처리가 필요한 경우(예: 누락된 값을 포함하는 행을 처리해야 하는 경우), genfromtxt 함수를 사용하고 싶을 것입니다.

With savetxt, you can specify headers, footers, comments, and more.

> savetxt를 사용하면 헤더, 푸터, 주석 등을 지정할 수 있습니다.

## Importing and exporting a CSV
## CSV 가져오기 및 내보내기

It's simple to read in a CSV that contains existing information. The best and easiest way to do this is to use Pandas.

> 기존 정보가 포함된 CSV를 읽어들이는 것은 간단합니다. 이를 수행하는 가장 좋고 쉬운 방법은 Pandas를 사용하는 것입니다.

```python
import pandas as pd

# If all of your columns are the same type:
x = pd.read_csv('music.csv', header=0).values
print(x)
[['Billie Holiday' 'Jazz' 1300000 27000000]
 ['Jimmie Hendrix' 'Rock' 2700000 70000000]
 ['Miles Davis' 'Jazz' 1500000 48000000]
 ['SIA' 'Pop' 2000000 74000000]]

# You can also simply select the columns you need:
x = pd.read_csv('music.csv', usecols=['Artist', 'Plays']).values
print(x)
[['Billie Holiday' 27000000]
 ['Jimmie Hendrix' 70000000]
 ['Miles Davis' 48000000]
 ['SIA' 74000000]]
```

It's simple to use Pandas in order to export your array as well. If you are new to NumPy, you may want to create a Pandas dataframe from the values in your array and then write the data frame to a CSV file with Pandas.

> 배열을 내보내기 위해 Pandas를 사용하는 것도 간단합니다. NumPy를 처음 사용하는 경우, 배열의 값에서 Pandas 데이터프레임을 만든 다음 Pandas를 사용하여 데이터프레임을 CSV 파일로 작성할 수 있습니다.

If you created this array "a"

> "a"라는 이 배열을 생성했다면:

```python
a = np.array([[-2.58289208,  0.43014843, -1.24082018, 1.59572603],
              [ 0.99027828, 1.17150989,  0.94125714, -0.14692469],
              [ 0.76989341,  0.81299683, -0.95068423, 0.11769564],
              [ 0.20484034,  0.34784527,  1.96979195, 0.51992837]])
```

You could create a Pandas dataframe

> Pandas 데이터프레임을 만들 수 있습니다:

```python
df = pd.DataFrame(a)
print(df)
          0         1         2         3
0 -2.582892  0.430148 -1.240820  1.595726
1  0.990278  1.171510  0.941257 -0.146925
2  0.769893  0.812997 -0.950684  0.117696
3  0.204840  0.347845  1.969792  0.519928
```

You can easily save your dataframe with:

> 데이터프레임을 쉽게 저장할 수 있습니다:

```python
df.to_csv('pd.csv')
```

And read your CSV with:

> 그리고 CSV를 다음과 같이 읽을 수 있습니다:

```python
data = pd.read_csv('pd.csv')
```

You can also save your array with the NumPy savetxt method.

> NumPy savetxt 메서드를 사용하여 배열을 저장할 수도 있습니다.

```python
np.savetxt('np.csv', a, fmt='%.2f', delimiter=',', header='1,  2,  3,  4')
```

If you're using the command line, you can read your saved CSV any time with a command such as:

> 명령줄을 사용하는 경우 다음과 같은 명령으로 언제든지 저장된 CSV를 읽을 수 있습니다:

```bash
$ cat np.csv
#  1,  2,  3,  4
-2.58,0.43,-1.24,1.60
0.99,1.17,0.94,-0.15
0.77,0.81,-0.95,0.12
0.20,0.35,1.97,0.52
```

Or you can open the file any time with a text editor!

> 또는 언제든지 텍스트 편집기로 파일을 열 수 있습니다!

If you're interested in learning more about Pandas, take a look at the official Pandas documentation. Learn how to install Pandas with the official Pandas installation information.

> Pandas에 대해 더 자세히 알고 싶다면, 공식 Pandas 문서를 참조하세요. 공식 Pandas 설치 정보로 Pandas를 설치하는 방법을 알아보세요.

## Plotting arrays with Matplotlib
## Matplotlib을 사용한 배열 시각화

If you need to generate a plot for your values, it's very simple with Matplotlib.

> 값에 대한 그래프를 생성해야 하는 경우 Matplotlib을 사용하면 매우 간단합니다.

For example, you may have an array like this one:

> 예를 들어, 다음과 같은 배열이 있을 수 있습니다:

```python
a = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22])
```

If you already have Matplotlib installed, you can import it with:

> Matplotlib이 이미 설치되어 있다면 다음과 같이 가져올 수 있습니다:

```python
import matplotlib.pyplot as plt

# If you're using Jupyter Notebook, you may also want to run the following
# line of code to display your code in the notebook:

%matplotlib inline
```

All you need to do to plot your values is run:

> 값을 그래프로 그리기 위해 필요한 것은 다음을 실행하는 것뿐입니다:

```python
plt.plot(a)

# If you are running from a command line, you may need to do this:
# >>> plt.show()
```

For example, you can plot a 1D array like this:

> 예를 들어, 1D 배열을 다음과 같이 그릴 수 있습니다:

```python
x = np.linspace(0, 5, 20)
y = np.linspace(0, 10, 20)
plt.plot(x, y, 'purple') # line
plt.plot(x, y, 'o')      # dots
```

With Matplotlib, you have access to an enormous number of visualization options.

> Matplotlib을 사용하면 엄청나게 많은 시각화 옵션을 활용할 수 있습니다.

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X = np.arange(-5, 5, 0.15)
Y = np.arange(-5, 5, 0.15)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
```

