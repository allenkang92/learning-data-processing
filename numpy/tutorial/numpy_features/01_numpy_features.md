# NumPy Features

A collection of notebooks pertaining to built-in NumPy functionality.

NumPy 내장 기능에 관한 노트북 모음입니다.

## Contents

- [Linear algebra on n-dimensional arrays](#linear-algebra-on-n-dimensional-arrays)
  - [n차원 배열에서의 선형 대수학](#linear-algebra-on-n-dimensional-arrays)
- [Saving and sharing your NumPy arrays](#saving-and-sharing-your-numpy-arrays)
  - [NumPy 배열 저장 및 공유하기](#saving-and-sharing-your-numpy-arrays)
- [Masked Arrays](#masked-arrays)
  - [마스크된 배열](#masked-arrays)

## Linear algebra on n-dimensional arrays
## n차원 배열에서의 선형 대수학

### Prerequisites

Before reading this tutorial, you should know a bit of Python. If you would like to refresh your memory, take a look at the Python tutorial.

이 튜토리얼을 읽기 전에 Python에 대한 기본 지식이 있어야 합니다. 기억을 되살리고 싶다면 Python 튜토리얼을 참고하세요.

If you want to be able to run the examples in this tutorial, you should also have matplotlib and SciPy installed on your computer.

이 튜토리얼의 예제를 실행하려면 컴퓨터에 matplotlib과 SciPy가 설치되어 있어야 합니다.

### Learner profile


This tutorial is for people who have a basic understanding of linear algebra and arrays in NumPy and want to understand how n-dimensional (n-D) arrays are represented and can be manipulated. In particular, if you don't know how to apply common functions to n-dimensional arrays (without using for-loops), or if you want to understand axis and shape properties for n-dimensional arrays, this tutorial might be of help.

이 튜토리얼은 선형 대수학과 NumPy의 배열에 대한 기본적인 이해가 있으며 n차원(n-D) 배열이 어떻게 표현되고 조작될 수 있는지 이해하고자 하는 사람들을 위한 것입니다. 특히 for 루프를 사용하지 않고 n차원 배열에 일반적인 함수를 적용하는 방법을 모르거나, n차원 배열의 축과 형태 속성을 이해하고자 한다면 이 튜토리얼이 도움이 될 수 있습니다.

### Learning Objectives


After this tutorial, you should be able to:
이 튜토리얼을 마친 후에는 다음을 수행할 수 있어야 합니다:

- Understand the difference between one-, two- and n-dimensional arrays in NumPy;
- NumPy에서 1차원, 2차원 및 n차원 배열의 차이점을 이해합니다;
- Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;
- for 루프를 사용하지 않고 n차원 배열에 일부 선형 대수 연산을 적용하는 방법을 이해합니다;
- Understand axis and shape properties for n-dimensional arrays.
- n차원 배열의 축과 형태 속성을 이해합니다.

### Content
### 내용

In this tutorial, we will use a matrix decomposition from linear algebra, the Singular Value Decomposition, to generate a compressed approximation of an image. We'll use the face image from the scipy.datasets module:

이 튜토리얼에서는 선형 대수학의 행렬 분해인 특이값 분해(Singular Value Decomposition)를 사용하여 이미지의 압축된 근사치를 생성할 것입니다. scipy.datasets 모듈에서 얼굴 이미지를 사용하겠습니다:

```python
# TODO: Rm try-except with scipy 1.10 is the minimum supported version
try:
    from scipy.datasets import face
except ImportError:  # Data was in scipy.misc prior to scipy v1.10
    from scipy.misc import face

img = face()
```

Note: If you prefer, you can use your own image as you work through this tutorial. In order to transform your image into a NumPy array that can be manipulated, you can use the imread function from the matplotlib.pyplot submodule. Alternatively, you can use the imageio.imread function from the imageio library. Be aware that if you use your own image, you'll likely need to adapt the steps below. For more information on how images are treated when converted to NumPy arrays, see A crash course on NumPy for images from the scikit-image documentation.

참고: 원하는 경우 이 튜토리얼을 진행하면서 자신의 이미지를 사용할 수 있습니다. 이미지를 조작할 수 있는 NumPy 배열로 변환하려면 matplotlib.pyplot 서브모듈의 imread 함수를 사용할 수 있습니다. 또는 imageio 라이브러리의 imageio.imread 함수를 사용할 수도 있습니다. 자신의 이미지를 사용하는 경우 아래 단계를 적절히 조정해야 할 수 있음을 유의하세요. 이미지가 NumPy 배열로 변환될 때 어떻게 처리되는지에 대한 자세한 정보는 scikit-image 문서의 'NumPy for images에 대한 속성 과정'을 참조하세요.

Now, img is a NumPy array, as we can see when using the type function:
이제 img는 type 함수를 사용할 때 볼 수 있듯이 NumPy 배열입니다:

```python
type(img)
```
```
numpy.ndarray
```

We can see the image using the matplotlib.pyplot.imshow function & the special iPython command, %matplotlib inline to display plots inline:

matplotlib.pyplot.imshow 함수와 특별한 iPython 명령인 %matplotlib inline을 사용하여 인라인으로 플롯을 표시할 수 있습니다:

```python
import matplotlib.pyplot as plt

%matplotlib inline
plt.imshow(img)
plt.show()
```

### Shape, axis and array properties
### 형태, 축 및 배열 속성

Note that, in linear algebra, the dimension of a vector refers to the number of entries in an array. In NumPy, it instead defines the number of axes. For example, a 1D array is a vector such as [1, 2, 3], a 2D array is a matrix, and so forth.

선형 대수학에서 벡터의 차원은 배열의 항목 수를 의미합니다. NumPy에서는 대신 축의 수를 정의합니다. 예를 들어, 1D 배열은 [1, 2, 3]과 같은 벡터이고, 2D 배열은 행렬이며, 그 이상도 마찬가지입니다.

First, let's check for the shape of the data in our array. Since this image is two-dimensional (the pixels in the image form a rectangle), we might expect a two-dimensional array to represent it (a matrix). However, using the shape property of this NumPy array gives us a different result:

먼저, 배열의 데이터 형태를 확인해 봅시다. 이 이미지는 2차원(이미지의 픽셀이 직사각형을 형성)이므로 2차원 배열(행렬)로 표현될 것이라고 예상할 수 있습니다. 그러나 이 NumPy 배열의 shape 속성을 사용하면 다른 결과를 얻게 됩니다:

```python
img.shape
```
```
(768, 1024, 3)
```

The output is a tuple with three elements, which means that this is a three-dimensional array. In fact, since this is a color image, and we have used the imread function to read it, the data is organized in three 2D arrays, representing color channels (in this case, red, green and blue - RGB). You can see this by looking at the shape above: it indicates that we have an array of 3 matrices, each having shape 768x1024.

출력은 3개의 요소가 있는 튜플로, 이것이 3차원 배열임을 의미합니다. 실제로 이것은 컬러 이미지이며 imread 함수를 사용하여 읽었기 때문에 데이터는 색상 채널(이 경우 빨강, 녹색, 파랑 - RGB)을 나타내는 3개의 2D 배열로 구성됩니다. 위의 형태를 보면 알 수 있습니다: 각각 768x1024 형태를 가진 3개의 행렬 배열이 있음을 나타냅니다.

Furthermore, using the ndim property of this array, we can see that

또한 이 배열의 ndim 속성을 사용하면 다음을 확인할 수 있습니다:

```python
img.ndim
```
```
3
```

NumPy refers to each dimension as an axis. Because of how imread works, the first index in the 3rd axis is the red pixel data for our image. We can access this by using the syntax

NumPy는 각 차원을 축(axis)이라고 합니다. imread가 작동하는 방식 때문에 3번째 축의 첫 번째 인덱스는 이미지의 빨간색 픽셀 데이터입니다. 다음 구문을 사용하여 이 데이터에 접근할 수 있습니다:

```python
img[:, :, 0]
```
```
array([[121, 138, 153, ..., 119, 131, 139],
       [ 89, 110, 130, ..., 118, 134, 146],
       [ 73,  94, 115, ..., 117, 133, 144],
       ...,
       [ 87,  94, 107, ..., 120, 119, 119],
       [ 85,  95, 112, ..., 121, 120, 120],
       [ 85,  97, 111, ..., 120, 119, 118]],
      shape=(768, 1024), dtype=uint8)
```

From the output above, we can see that every value in img[:, :, 0] is an integer value between 0 and 255, representing the level of red in each corresponding image pixel (keep in mind that this might be different if you use your own image instead of scipy.datasets.face).

위의 출력에서 img[:, :, 0]의 모든 값은 0과 255 사이의 정수 값으로, 각 해당 이미지 픽셀의 빨간색 수준을 나타냅니다(scipy.datasets.face 대신 자신의 이미지를 사용하는 경우 다를 수 있음을 명심하세요).

As expected, this is a 768x1024 matrix:

예상대로 이것은 768x1024 행렬입니다:

```python
img[:, :, 0].shape
```
```
(768, 1024)
```

Since we are going to perform linear algebra operations on this data, it might be more interesting to have real numbers between 0 and 1 in each entry of the matrices to represent the RGB values. We can do that by setting

이 데이터에 선형 대수 연산을 수행할 것이므로, RGB 값을 나타내기 위해 행렬의 각 항목에 0과 1 사이의 실수를 갖는 것이 더 흥미로울 수 있습니다. 다음과 같이 설정할 수 있습니다:

```python
img_array = img / 255
```

This operation, dividing an array by a scalar, works because of NumPy's broadcasting rules. (Note that in real-world applications, it would be better to use, for example, the img_as_float utility function from scikit-image).

이 연산, 즉 배열을 스칼라로 나누는 것은 NumPy의 브로드캐스팅 규칙 때문에 작동합니다. (실제 응용 프로그램에서는 예를 들어 scikit-image의 img_as_float 유틸리티 함수를 사용하는 것이 더 좋습니다).

You can check that the above works by doing some tests; for example, inquiring about maximum and minimum values for this array:

위의 방법이 작동하는지 몇 가지 테스트를 통해 확인할 수 있습니다. 예를 들어, 이 배열의 최대 및 최소 값에 대해 문의하는 것입니다:

```python
img_array.max(), img_array.min()
```
```
(np.float64(1.0), np.float64(0.0))
```

or checking the type of data in the array:

또는 배열의 데이터 유형을 확인할 수 있습니다:

```python
img_array.dtype
```
```
dtype('float64')
```

Note that we can assign each color channel to a separate matrix using the slice syntax:

슬라이스 구문을 사용하여 각 색상 채널을 별도의 행렬에 할당할 수 있습니다:

```python
red_array = img_array[:, :, 0]
green_array = img_array[:, :, 1]
blue_array = img_array[:, :, 2]
```

### Operations on an axis

It is possible to use methods from linear algebra to approximate an existing set of data. Here, we will use the SVD (Singular Value Decomposition) to try to rebuild an image that uses less singular value information than the original one, while still retaining some of its features.

선형 대수학의 방법을 사용하여 기존 데이터 세트를 근사화하는 것이 가능합니다. 여기서는 SVD(특이값 분해)를 사용하여 원래 이미지보다 적은 특이값 정보를 사용하면서도 일부 특징을 유지하는 이미지를 재구성해 보겠습니다.

Note: We will use NumPy's linear algebra module, numpy.linalg, to perform the operations in this tutorial. Most of the linear algebra functions in this module can also be found in scipy.linalg, and users are encouraged to use the scipy module for real-world applications. However, some functions in the scipy.linalg module, such as the SVD function, only support 2D arrays. For more information on this, check the scipy.linalg page.

참고: 이 튜토리얼에서는 NumPy의 선형 대수 모듈인 numpy.linalg를 사용하여 연산을 수행할 것입니다. 이 모듈의 대부분의 선형 대수 함수는 scipy.linalg에서도 찾을 수 있으며, 실제 응용 프로그램에서는 scipy 모듈을 사용하는 것이 좋습니다. 그러나 scipy.linalg 모듈의 일부 함수(예: SVD 함수)는 2D 배열만 지원합니다. 이에 대한 자세한 정보는 scipy.linalg 페이지를 확인하세요.

To proceed, import the linear algebra submodule from NumPy:

계속하려면 NumPy에서 선형 대수 서브모듈을 가져옵니다:

```python
from numpy import linalg
```

In order to extract information from a given matrix, we can use the SVD to obtain 3 arrays which can be multiplied to obtain the original matrix. From the theory of linear algebra, given a matrix A, the following product can be computed:

주어진 행렬에서 정보를 추출하기 위해 SVD를 사용하여 원래 행렬을 얻기 위해 곱할 수 있는 3개의 배열을 얻을 수 있습니다. 선형 대수학 이론에 따르면, 행렬 A가 주어지면 다음 곱을 계산할 수 있습니다:

A = U @ Σ @ V^T

where U and V^T are square and Σ is the same size as A. Σ is a diagonal matrix and contains the singular values of A, organized from largest to smallest. These values are always non-negative and can be used as an indicator of the "importance" of some features represented by the matrix A.

여기서 U와 V^T는 정사각형이고 Σ는 A와 같은 크기입니다. Σ는 대각 행렬이며 A의 특이값을 가장 큰 값부터 가장 작은 값까지 정렬하여 포함합니다. 이 값들은 항상 음수가 아니며 행렬 A로 표현되는 일부 특징의 "중요성" 지표로 사용될 수 있습니다.

Let's see how this works in practice with just one matrix first. Note that according to colorimetry, it is possible to obtain a fairly reasonable grayscale version of our color image if we apply the formula

먼저 한 행렬로만 이것이 어떻게 작동하는지 살펴보겠습니다. 색채측정학에 따르면, 다음 공식을 적용하면 컬러 이미지의 상당히 합리적인 그레이스케일 버전을 얻을 수 있습니다:

G = 0.2126 R + 0.7152 G + 0.0722 B

where G is the array representing the grayscale image, and R, G and B are the red, green and blue channel arrays we had originally. Notice we can use the @ operator (the matrix multiplication operator for NumPy arrays, see numpy.matmul) for this:

여기서 G는 그레이스케일 이미지를 나타내는 배열이고, R, G, B는 원래 가지고 있던 빨강, 녹색, 파랑 채널 배열입니다. 이를 위해 @ 연산자(NumPy 배열의 행렬 곱셈 연산자, numpy.matmul 참조)를 사용할 수 있습니다:

```python
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
```

Now, img_gray has shape

이제 img_gray의 형태는 다음과 같습니다:

```python
img_gray.shape
```
```
(768, 1024)
```

To see if this makes sense in our image, we should use a colormap from matplotlib corresponding to the color we wish to see in out image (otherwise, matplotlib will default to a colormap that does not correspond to the real data).

이것이 우리 이미지에서 의미가 있는지 확인하기 위해, 우리가 이미지에서 보고 싶은 색상에 해당하는 matplotlib의 컬러맵을 사용해야 합니다(그렇지 않으면 matplotlib은 실제 데이터에 해당하지 않는 컬러맵을 기본값으로 사용합니다).

In our case, we are approximating the grayscale portion of the image, so we will use the colormap gray:

우리의 경우, 이미지의 그레이스케일 부분을 근사화하고 있으므로 gray 컬러맵을 사용할 것입니다:

```python
plt.imshow(img_gray, cmap="gray")
plt.show()
```

Now, applying the linalg.svd function to this matrix, we obtain the following decomposition:

이제 이 행렬에 linalg.svd 함수를 적용하면 다음과 같은 분해를 얻습니다:

```python
U, s, Vt = linalg.svd(img_gray)
```

Note: If you are using your own image, this command might take a while to run, depending on the size of your image and your hardware. Don't worry, this is normal! The SVD can be a pretty intensive computation.

참고: 자신의 이미지를 사용하는 경우 이미지 크기와 하드웨어에 따라 이 명령이 실행되는 데 시간이 걸릴 수 있습니다. 걱정하지 마세요, 이것은 정상입니다! SVD는 상당히 집약적인 계산일 수 있습니다.

Let's check that this is what we expected:

이것이 우리가 기대한 것인지 확인해 봅시다:

```python
U.shape, s.shape, Vt.shape
```
```
((768, 768), (768,), (1024, 1024))
```

Note that s has a particular shape: it has only one dimension. This means that some linear algebra functions that expect 2d arrays might not work. For example, from the theory, one might expect s and Vt to be compatible for multiplication. However, this is not true as s does not have a second axis. Executing

s의 특별한 형태에 주목하세요: 한 차원만 있습니다. 이는 2차원 배열을 기대하는 일부 선형 대수 함수가 작동하지 않을 수 있음을 의미합니다. 예를 들어, 이론상으로는 s와 Vt가 곱셈에 호환될 것이라고 예상할 수 있습니다. 그러나 s에는 두 번째 축이 없기 때문에 이것은 사실이 아닙니다. 다음을 실행하면:

```python
s @ Vt
```

results in a ValueError. This happens because having a one-dimensional array for s, in this case, is much more economic in practice than building a diagonal matrix with the same data. To reconstruct the original matrix, we can rebuild the diagonal matrix Σ with the elements of s in its diagonal and with the appropriate dimensions for multiplying: in our case, Σ should be 768x1024 since U is 768x768 and Vt is 1024x1024. In order to add the singular values to the diagonal of Sigma, we will use the fill_diagonal function from NumPy:

ValueError가 발생합니다. 이는 이 경우 s에 대해 1차원 배열을 갖는 것이 동일한 데이터로 대각 행렬을 구축하는 것보다 실제로 훨씬 더 경제적이기 때문입니다. 원래 행렬을 재구성하기 위해, 대각선에 s의 요소를 넣고 곱셈에 적합한 차원을 가진 대각 행렬 Σ를 재구축할 수 있습니다: 우리의 경우, U가 768x768이고 Vt가 1024x1024이므로 Σ는 768x1024여야 합니다. Sigma의 대각선에 특이값을 추가하기 위해 NumPy의 fill_diagonal 함수를 사용할 것입니다:

```python
import numpy as np

Sigma = np.zeros((U.shape[1], Vt.shape[0]))
np.fill_diagonal(Sigma, s)
```

Now, we want to check if the reconstructed U @ Sigma @ Vt is close to the original img_gray matrix.

이제 재구성된 U @ Sigma @ Vt가 원래 img_gray 행렬에 가까운지 확인하고자 합니다.

### Approximation
### 근사화

The linalg module includes a norm function, which computes the norm of a vector or matrix represented in a NumPy array. For example, from the SVD explanation above, we would expect the norm of the difference between img_gray and the reconstructed SVD product to be small. As expected, you should see something like
linalg 모듈에는 NumPy 배열로 표현된 벡터나 행렬의 노름(norm)을 계산하는 norm 함수가 포함되어 있습니다. 예를 들어, 위의 SVD 설명에서 우리는 img_gray와 재구성된 SVD 곱 사이의 차이의 노름이 작을 것으로 예상합니다. 예상대로 다음과 같은 결과가 나타날 것입니다:

```python
linalg.norm(img_gray - U @ Sigma @ Vt)
```
```
np.float64(1.43712046073728e-12)
```

(The actual result of this operation might be different depending on your architecture and linear algebra setup. Regardless, you should see a small number.)
(이 연산의 실제 결과는 아키텍처와 선형 대수 설정에 따라 다를 수 있습니다. 어쨌든 작은 숫자가 표시되어야 합니다.)

We could also have used the numpy.allclose function to make sure the reconstructed product is, in fact, close to our original matrix (the difference between the two arrays is small):

재구성된 곱이 실제로 원래 행렬과 가까운지(두 배열 간의 차이가 작은지) 확인하기 위해 numpy.allclose 함수를 사용할 수도 있습니다:

```python
np.allclose(img_gray, U @ Sigma @ Vt)
```
```
True
```

To see if an approximation is reasonable, we can check the values in s:
근사치가 합리적인지 확인하기 위해 s의 값을 확인할 수 있습니다:

```python
plt.plot(s)
plt.show()
```

In the graph, we can see that although we have 768 singular values in s, most of those (after the 150th entry or so) are pretty small. So it might make sense to use only the information related to the first (say, 50) singular values to build a more economical approximation to our image.

그래프에서 s에 768개의 특이값이 있지만, 그 중 대부분(약 150번째 항목 이후)은 매우 작다는 것을 알 수 있습니다. 따라서 이미지의 더 경제적인 근사치를 구축하기 위해 첫 번째(예: 50개) 특이값과 관련된 정보만 사용하는 것이 타당할 수 있습니다.

The idea is to consider all but the first k singular values in Sigma (which are the same as in s) as zeros, keeping U and Vt intact, and computing the product of these matrices as the approximation.

아이디어는 Sigma(s와 동일한)의 첫 번째 k개의 특이값을 제외한 모든 값을 0으로 간주하고, U와 Vt를 그대로 유지하면서 이러한 행렬의 곱을 근사치로 계산하는 것입니다.

For example, if we choose
예를 들어, 다음을 선택한다면:

```python
k = 10
```

we can build the approximation by doing

다음과 같이 근사치를 구축할 수 있습니다:

```python
approx = U @ Sigma[:, :k] @ Vt[:k, :]
```

Note that we had to use only the first k rows of Vt, since all other rows would be multiplied by the zeros corresponding to the singular values we eliminated from this approximation.

이 근사치에서 제거한 특이값에 해당하는 0으로 다른 모든 행이 곱해지기 때문에 Vt의 첫 k개 행만 사용해야 했음을 주목하세요.

```python
plt.imshow(approx, cmap="gray")
plt.show()
```

Now, you can go ahead and repeat this experiment with other values of k, and each of your experiments should give you a slightly better (or worse) image depending on the value you choose.

이제 다른 k 값으로 이 실험을 반복할 수 있으며, 선택한 값에 따라 각 실험에서 약간 더 좋거나(또는 나쁜) 이미지를 얻을 수 있습니다.

### Applying to all colors
### 모든 색상에 적용하기

Now we want to do the same kind of operation, but to all three colors. Our first instinct might be to repeat the same operation we did above to each color matrix individually. However, NumPy's broadcasting takes care of this for us.

이제 같은 종류의 연산을 세 가지 색상 모두에 적용하고자 합니다. 첫 번째 직관은 위에서 수행한 것과 동일한 연산을 각 색상 행렬에 개별적으로 반복하는 것일 수 있습니다. 그러나 NumPy의 브로드캐스팅이 이를 처리해 줍니다.

If our array has more than two dimensions, then the SVD can be applied to all axes at once. However, the linear algebra functions in NumPy expect to see an array of the form (n, M, N), where the first axis n represents the number of MxN matrices in the stack.

배열이 두 개 이상의 차원을 가진 경우 SVD는 모든 축에 한 번에 적용될 수 있습니다. 그러나 NumPy의 선형 대수 함수는 (n, M, N) 형태의 배열을 예상하며, 여기서 첫 번째 축 n은 스택에 있는 MxN 행렬의 수를 나타냅니다.

In our case,
우리의 경우:

```python
img_array.shape
```
```
(768, 1024, 3)
```

so we need to permutate the axis on this array to get a shape like (3, 768, 1024). Fortunately, the numpy.transpose function can do that for us:

따라서 (3, 768, 1024)와 같은 형태를 얻기 위해 이 배열의 축을 치환해야 합니다. 다행히도, numpy.transpose 함수가 이를 수행할 수 있습니다:

```python
np.transpose(x, axes=(i, j, k))
```

indicates that the axis will be reordered such that the final shape of the transposed array will be reordered according to the indices (i, j, k).

축이 재정렬되어 전치된 배열의 최종 형태가 인덱스 (i, j, k)에 따라 재정렬됨을 나타냅니다.

Let's see how this goes for our array:

우리 배열에 대해 이것이 어떻게 진행되는지 봅시다:

```python
img_array_transposed = np.transpose(img_array, (2, 0, 1))
img_array_transposed.shape
```
```
(3, 768, 1024)
```

Now we are ready to apply the SVD:

이제 SVD를 적용할 준비가 되었습니다:

```python
U, s, Vt = linalg.svd(img_array_transposed)
```

Finally, to obtain the full approximated image, we need to reassemble these matrices into the approximation. Now, note that

마지막으로, 완전한 근사 이미지를 얻기 위해 이러한 행렬을 근사치로 재조립해야 합니다. 이제 다음을 주목하세요:

```python
U.shape, s.shape, Vt.shape
```
```
((3, 768, 768), (3, 768), (3, 1024, 1024))
```

To build the final approximation matrix, we must understand how multiplication across different axes works.

최종 근사 행렬을 구축하기 위해 서로 다른 축 간의 곱셈이 어떻게 작동하는지 이해해야 합니다.

### Products with n-dimensional arrays
### n차원 배열과의 곱셈

If you have worked before with only one- or two-dimensional arrays in NumPy, you might use numpy.dot and numpy.matmul (or the @ operator) interchangeably. However, for n-dimensional arrays, they work in very different ways. For more details, check the documentation on numpy.matmul.

이전에 NumPy에서 1차원 또는 2차원 배열만 사용해 본 경우, numpy.dot과 numpy.matmul(또는 @ 연산자)을 서로 바꿔 사용할 수 있습니다. 그러나 n차원 배열의 경우 매우 다른 방식으로 작동합니다. 자세한 내용은 numpy.matmul에 대한 문서를 확인하세요.

Now, to build our approximation, we first need to make sure that our singular values are ready for multiplication, so we build our Sigma matrix similarly to what we did before. The Sigma array must have dimensions (3, 768, 1024). In order to add the singular values to the diagonal of Sigma, we will again use the fill_diagonal function, using each of the 3 rows in s as the diagonal for each of the 3 matrices in Sigma:

이제 근사치를 구축하기 위해 먼저 특이값이 곱셈 준비가 되어 있는지 확인해야 합니다. 따라서 이전과 유사하게 Sigma 행렬을 구축합니다. Sigma 배열은 (3, 768, 1024) 차원을 가져야 합니다. Sigma의 대각선에 특이값을 추가하기 위해 다시 fill_diagonal 함수를 사용하여 s의 3개 행 각각을 Sigma의 3개 행렬 각각의 대각선으로 사용합니다:

```python
Sigma = np.zeros((3, 768, 1024))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])
```

Now, if we wish to rebuild the full SVD (with no approximation), we can do

이제 완전한 SVD(근사 없음)를 다시 구축하고자 한다면 다음과 같이 할 수 있습니다:

```python
reconstructed = U @ Sigma @ Vt
```

Note that
다음을 주목하세요:

```python
reconstructed.shape
```
```
(3, 768, 1024)
```

The reconstructed image should be indistinguishable from the original one, except for differences due to floating point errors from the reconstruction. Recall that our original image consisted of floating point values in the range [0., 1.]. The accumulation of floating point error from the reconstruction can result in values slightly outside this original range:

재구성된 이미지는 재구성으로 인한 부동소수점 오류로 인한 차이를 제외하고는 원본과 구별할 수 없어야 합니다. 원래 이미지가 [0., 1.] 범위의 부동소수점 값으로 구성되어 있었음을 상기하세요. 재구성으로 인한 부동소수점 오류의 누적으로 인해 이 원래 범위를 약간 벗어나는 값이 발생할 수 있습니다:

```python
reconstructed.min(), reconstructed.max()
```
```
(np.float64(-5.558487697898684e-15), np.float64(1.0000000000000053))
```

Since imshow expects values in the range, we can use clip to excise the floating point error:
imshow가 범위 내의 값을 기대하므로, clip을 사용하여 부동소수점 오류를 제거할 수 있습니다:

```python
reconstructed = np.clip(reconstructed, 0, 1)
plt.imshow(np.transpose(reconstructed, (1, 2, 0)))
plt.show()
```

In fact, imshow peforms this clipping under-the-hood, so if you skip the first line in the previous code cell, you might see a warning message saying "Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)."

실제로 imshow는 이 클리핑을 내부적으로 수행하므로, 이전 코드 셀의 첫 번째 줄을 건너뛰면 "RGB 데이터에 대한 imshow의 유효 범위로 입력 데이터 클리핑 중([0..1], 부동소수점 또는 [0..255], 정수)"이라는 경고 메시지가 표시될 수 있습니다.

Now, to do the approximation, we must choose only the first k singular values for each color channel. This can be done using the following syntax:

이제 근사를 수행하기 위해 각 색상 채널에 대해 첫 번째 k개의 특이값만 선택해야 합니다. 이는 다음 구문을 사용하여 수행할 수 있습니다:

```python
approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]
```

You can see that we have selected only the first k components of the last axis for Sigma (this means that we have used only the first k columns of each of the three matrices in the stack), and that we have selected only the first k components in the second-to-last axis of Vt (this means we have selected only the first k rows from every matrix in the stack Vt and all columns). If you are unfamiliar with the ellipsis syntax, it is a placeholder for other axes. For more details, see the documentation on Indexing.

Sigma의 마지막 축에서 첫 번째 k개 성분만 선택했음을 알 수 있습니다(이는 스택의 세 행렬 각각에서 첫 번째 k개 열만 사용했음을 의미합니다). 또한 Vt의 뒤에서 두 번째 축에서 첫 번째 k개 성분만 선택했습니다(이는 스택 Vt의 모든 행렬에서 첫 번째 k개 행과 모든 열만 선택했음을 의미합니다). 생략 구문에 익숙하지 않다면, 이는 다른 축을 위한 자리 표시자입니다. 자세한 내용은 인덱싱에 관한 문서를 참조하세요.

Now,
이제:

```python
approx_img.shape
```
```
(3, 768, 1024)
```

which is not the right shape for showing the image. Finally, reordering the axes back to our original shape of (768, 1024, 3), we can see our approximation:
이것은 이미지를 표시하기 위한 올바른 형태가 아닙니다. 마지막으로, 축을 원래 형태인 (768, 1024, 3)으로 다시 정렬하면 우리의 근사치를 볼 수 있습니다:

```python
plt.imshow(np.transpose(approx_img, (1, 2, 0)))
plt.show()
```

Even though the image is not as sharp, using a small number of k singular values (compared to the original set of 768 values), we can recover many of the distinguishing features from this image.

이미지가 선명하지는 않지만, 적은 수의 k 특이값(원래 768개 값 세트에 비해)을 사용하여 이 이미지의 많은 구별 특징을 복구할 수 있습니다.

### Final words
### 마무리 말

Of course, this is not the best method to approximate an image. However, there is, in fact, a result in linear algebra that says that the approximation we built above is the best we can get to the original matrix in terms of the norm of the difference. For more information, see G. H. Golub and C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University Press, 1985.

물론, 이것은 이미지를 근사화하는 최선의 방법은 아닙니다. 그러나 사실 선형 대수학에는 우리가 위에서 구축한 근사치가 차이의 노름 측면에서 원래 행렬에 얻을 수 있는 최선이라고 말하는 결과가 있습니다. 자세한 정보는 G. H. Golub 및 C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University Press, 1985를 참조하세요.
