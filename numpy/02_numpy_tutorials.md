# Linear algebra on n-dimensional arrays

> # 다차원 배열에서의 선형대수학

+++

## Prerequisites

Before reading this tutorial, you should know a bit of Python. If you would like to refresh your memory, take a look at the [Python tutorial](https://docs.python.org/3/tutorial/).

If you want to be able to run the examples in this tutorial, you should also have [matplotlib](https://matplotlib.org/) and [SciPy](https://scipy.org) installed on your computer.

> ## 사전 요구사항
>
> 이 튜토리얼을 읽기 전에 Python에 대한 기본 지식이 있어야 합니다. 기억을 새롭게 하고 싶다면 [Python 튜토리얼](https://docs.python.org/3/tutorial/)을 참고하세요.
>
> 이 튜토리얼의 예제를 실행하려면 컴퓨터에 [matplotlib](https://matplotlib.org/)와 [SciPy](https://scipy.org)가 설치되어 있어야 합니다.

## Learner profile

This tutorial is for people who have a basic understanding of linear algebra and arrays in NumPy and want to understand how n-dimensional ($n>=2$) arrays are represented and can be manipulated. In particular, if you don't know how to apply common functions to n-dimensional arrays (without using for-loops), or if you want to understand axis and shape properties for n-dimensional arrays, this tutorial might be of help.

> ## 학습자 프로필
>
> 이 튜토리얼은 선형대수학과 NumPy 배열에 대한 기본적인 이해가 있으며, n차원($n>=2$) 배열이 어떻게 표현되고 조작될 수 있는지 이해하고자 하는 사람들을 위한 것입니다. 특히, n차원 배열에 일반적인 함수를 어떻게 적용하는지(for 루프를 사용하지 않고) 모르거나, n차원 배열의 축(axis)과 형태(shape) 속성을 이해하고 싶다면 이 튜토리얼이 도움이 될 것입니다.

## Learning Objectives

After this tutorial, you should be able to:

- Understand the difference between one-, two- and n-dimensional arrays in NumPy;
- Understand how to apply some linear algebra operations to n-dimensional arrays without using for-loops;
- Understand axis and shape properties for n-dimensional arrays.

> ## 학습 목표
>
> 이 튜토리얼을 마친 후에는 다음을 할 수 있어야 합니다:
>
> - NumPy에서 1차원, 2차원 및 n차원 배열 간의 차이점 이해하기;
> - for 루프를 사용하지 않고 n차원 배열에 선형대수 연산을 적용하는 방법 이해하기;
> - n차원 배열의 축(axis)과 형태(shape) 속성 이해하기.

## Content

In this tutorial, we will use a [matrix decomposition](https://en.wikipedia.org/wiki/Matrix_decomposition) from linear algebra, the Singular Value Decomposition, to generate a compressed approximation of an image. We'll use the `face` image from the [scipy.datasets](https://docs.scipy.org/doc/scipy/reference/datasets.html) module:

> ## 내용
>
> 이 튜토리얼에서는 선형대수학의 [행렬 분해](https://en.wikipedia.org/wiki/Matrix_decomposition) 방법 중 하나인 특이값 분해(SVD)를 사용하여 이미지의 압축된 근사치를 생성합니다. [scipy.datasets](https://docs.scipy.org/doc/scipy/reference/datasets.html) 모듈에서 제공하는 `face` 이미지를 사용할 것입니다:

```{code-cell}
# TODO: Rm try-except with scipy 1.10 is the minimum supported version
try:
    from scipy.datasets import face
except ImportError:  # Data was in scipy.misc prior to scipy v1.10
    from scipy.misc import face

img = face()
```

**Note**: If you prefer, you can use your own image as you work through this tutorial. In order to transform your image into a NumPy array that can be manipulated, you can use the `imread` function from the [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot) submodule. Alternatively, you can use the [imageio.imread](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.v3.imread.html) function from the `imageio` library. Be aware that if you use your own image, you'll likely need to adapt the steps below. For more information on how images are treated when converted to NumPy arrays, see [A crash course on NumPy for images](https://scikit-image.org/docs/stable/user_guide/numpy_images.html) from the `scikit-image` documentation.

> **참고**: 원한다면 이 튜토리얼을 진행하면서 자신의 이미지를 사용할 수도 있습니다. 이미지를 조작할 수 있는 NumPy 배열로 변환하기 위해, [matplotlib.pyplot](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot) 서브모듈의 `imread` 함수를 사용할 수 있습니다. 또는 `imageio` 라이브러리의 [imageio.imread](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.v3.imread.html) 함수를 사용할 수도 있습니다. 자신의 이미지를 사용할 경우 아래 단계를 조정해야 할 수도 있습니다. 이미지가 NumPy 배열로 변환될 때 어떻게 처리되는지에 대한 자세한 정보는 `scikit-image` 문서의 [이미지를 위한 NumPy 속성 강좌](https://scikit-image.org/docs/stable/user_guide/numpy_images.html)를 참조하세요.

+++

Now, `img` is a NumPy array, as we can see when using the `type` function:

> 이제 `img`는 NumPy 배열입니다. `type` 함수를 사용하여 확인할 수 있습니다:

```{code-cell}
type(img)
```

We can see the image using the [matplotlib.pyplot.imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow) function & the special iPython command, `%matplotlib inline` to display plots inline:

> [matplotlib.pyplot.imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow) 함수와 특별한 iPython 명령어인 `%matplotlib inline`을 사용하여 이미지를 인라인으로 표시할 수 있습니다:

```{code-cell}
import matplotlib.pyplot as plt

%matplotlib inline
```

```{code-cell}
plt.imshow(img)
plt.show()
```

### Shape, axis and array properties

Note that, in linear algebra, the dimension of a vector refers to the number of entries in an array. In NumPy, it instead defines the number of axes. For example, a 1D array is a vector such as `[1, 2, 3]`, a 2D array is a matrix, and so forth.

> ### 형태, 축 및 배열 속성
>
> 선형대수학에서 벡터의 차원은 배열의 항목 수를 의미합니다. 그러나 NumPy에서는 축의 수를 정의합니다. 예를 들어, 1D 배열은 `[1, 2, 3]`과 같은 벡터이고, 2D 배열은 행렬이며, 이런 식으로 계속됩니다.

First, let's check for the shape of the data in our array. Since this image is two-dimensional (the pixels in the image form a rectangle), we might expect a two-dimensional array to represent it (a matrix). However, using the `shape` property of this NumPy array gives us a different result:

> 먼저, 배열의 형태를 확인해봅시다. 이 이미지는 2차원(이미지의, 픽셀이 직사각형을 형성함)이므로 2차원 배열(행렬)로 표현될 것이라고 예상할 수 있습니다. 그러나 이 NumPy 배열의 `shape` 속성을 사용하면 다른 결과를 얻게 됩니다:

```{code-cell}
img.shape
```

The output is a [tuple](https://docs.python.org/dev/tutorial/datastructures.html#tut-tuples) with three elements, which means that this is a three-dimensional array. In fact, since this is a color image, and we have used the `imread` function to read it, the data is organized in three 2D arrays, representing color channels (in this case, red, green and blue - RGB). You can see this by looking at the shape above: it indicates that we have an array of 3 matrices, each having shape 768x1024.

> 출력은 세 개의 요소를 가진 [튜플](https://docs.python.org/dev/tutorial/datastructures.html#tut-tuples)이며, 이는 3차원 배열임을 의미합니다. 실제로, 이것은 컬러 이미지이고 `imread` 함수를 사용해 읽었기 때문에, 데이터는 세 개의 2D 배열로 구성되어 있으며, 이는 색상 채널(이 경우, 빨강, 초록, 파랑 - RGB)을 나타냅니다. 위의 형태를 보면 알 수 있듯이, 우리는 각각 768x1024 형태를 가진 3개의 행렬 배열을 가지고 있습니다.

Furthermore, using the `ndim` property of this array, we can see that

> 또한, 이 배열의 `ndim` 속성을 사용하면 다음을 확인할 수 있습니다:

```{code-cell}
img.ndim
```

NumPy refers to each dimension as an *axis*. Because of how `imread` works, the *first index in the 3rd axis* is the red pixel data for our image. We can access this by using the syntax

> NumPy는 각 차원을 *축(axis)*이라고 부릅니다. `imread`가 작동하는 방식 때문에, *3번째 축의 첫 번째 인덱스*는 우리 이미지의 빨간색 픽셀 데이터입니다. 다음 구문을 사용하여 이에 접근할 수 있습니다:

```{code-cell}
img[:, :, 0]
```

From the output above, we can see that every value in `img[:, :, 0]` is an integer value between 0 and 255, representing the level of red in each corresponding image pixel (keep in mind that this might be different if you
use your own image instead of [scipy.datasets.face](https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.face.html)).

> 위의 출력에서 `img[:, :, 0]`의 모든 값이 0에서 255 사이의 정수 값임을 알 수 있으며, 이는 각 이미지 픽셀에서 빨간색의 레벨을 나타냅니다([scipy.datasets.face](https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.face.html) 대신 자신의 이미지를 사용하는 경우 이 값이 다를 수 있음을 명심하세요).

As expected, this is a 768x1024 matrix:

> 예상대로 이것은 768x1024 행렬입니다:

```{code-cell}
img[:, :, 0].shape
```

Since we are going to perform linear algebra operations on this data, it might be more interesting to have real numbers between 0 and 1 in each entry of the matrices to represent the RGB values. We can do that by setting

> 이 데이터에 대해 선형대수 연산을 수행할 것이므로, RGB 값을 나타내기 위해 행렬의 각 항목에 0과 1 사이의 실수를 갖는 것이 더 적합할 수 있습니다. 다음과 같이 설정하여 이를 수행할 수 있습니다:

```{code-cell}
img_array = img / 255
```

This operation, dividing an array by a scalar, works because of NumPy's [broadcasting rules](https://numpy.org/devdocs/user/theory.broadcasting.html#array-broadcasting-in-numpy). (Note that in real-world applications, it would be better to use, for example, the [img_as_float](https://scikit-image.org/docs/stable/api/skimage.html#skimage.img_as_float) utility function from `scikit-image`).

> 이 연산, 즉 배열을 스칼라로 나누는 것은 NumPy의 [브로드캐스팅 규칙](https://numpy.org/devdocs/user/theory.broadcasting.html#array-broadcasting-in-numpy) 덕분에 작동합니다. (실제 응용에서는 예를 들어 `scikit-image`의 [img_as_float](https://scikit-image.org/docs/stable/api/skimage.html#skimage.img_as_float) 유틸리티 함수를 사용하는 것이 더 좋습니다).

You can check that the above works by doing some tests; for example, inquiring
about maximum and minimum values for this array:

> 몇 가지 테스트를 수행하여 위의 작업이 제대로 작동하는지 확인할 수 있습니다. 예를 들어, 이 배열의 최대값과 최소값을 조회해 봅시다:

```{code-cell}
img_array.max(), img_array.min()
```

or checking the type of data in the array:

> 또는 배열의 데이터 유형을 확인해 봅시다:

```{code-cell}
img_array.dtype
```

Note that we can assign each color channel to a separate matrix using the slice syntax:

> 슬라이스 구문을 사용하여 각 색상 채널을 별도의 행렬에 할당할 수 있습니다:

```{code-cell}
red_array = img_array[:, :, 0]
green_array = img_array[:, :, 1]
blue_array = img_array[:, :, 2]
```

### Operations on an axis

It is possible to use methods from linear algebra to approximate an existing set of data. Here, we will use the [SVD (Singular Value Decomposition)](https://en.wikipedia.org/wiki/Singular_value_decomposition) to try to rebuild an image that uses less singular value information than the original one, while still retaining some of its features.

> ### 축에 대한 연산
>
> 선형대수학의 방법을 사용하여 기존 데이터 집합을 근사화하는 것이 가능합니다. 여기서는 [SVD(특이값 분해)](https://en.wikipedia.org/wiki/Singular_value_decomposition)를 사용하여 원본보다 적은 특이값 정보를 사용하면서도 일부 특징을 유지하는 이미지를 재구성해 볼 것입니다.

+++

**Note**: We will use NumPy's linear algebra module, [numpy.linalg](https://numpy.org/devdocs/reference/routines.linalg.html#module-numpy.linalg), to perform the operations in this tutorial. Most of the linear algebra functions in this module can also be found in [scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg), and users are encouraged to use the [scipy](https://docs.scipy.org/doc/scipy/reference/index.html#module-scipy) module for real-world applications. However, some functions in the [scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg) module, such as the SVD function, only support 2D arrays. For more information on this, check the [scipy.linalg page](https://docs.scipy.org/doc/scipy/tutorial/linalg.html).

> **참고**: 이 튜토리얼에서는 NumPy의 선형대수 모듈인 [numpy.linalg](https://numpy.org/devdocs/reference/routines.linalg.html#module-numpy.linalg)를 사용하여 연산을 수행할 것입니다. 이 모듈의 대부분의 선형대수 함수는 [scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg)에서도 찾을 수 있으며, 실제 응용에는 [scipy](https://docs.scipy.org/doc/scipy/reference/index.html#module-scipy) 모듈을 사용하는 것이 좋습니다. 그러나 [scipy.linalg](https://docs.scipy.org/doc/scipy/reference/linalg.html#module-scipy.linalg) 모듈의 일부 함수, 예를 들어 SVD 함수는 2D 배열만 지원합니다. 이에 대한 자세한 정보는 [scipy.linalg 페이지](https://docs.scipy.org/doc/scipy/tutorial/linalg.html)를 확인하세요.

+++

To proceed, import the linear algebra submodule from NumPy:

> 계속하기 위해 NumPy에서 선형대수 서브모듈을 가져옵니다:

```{code-cell}
from numpy import linalg
```

In order to extract information from a given matrix, we can use the SVD to obtain 3 arrays which can be multiplied to obtain the original matrix. From the theory of linear algebra, given a matrix $A$, the following product can be computed:

> 주어진 행렬에서 정보를 추출하기 위해, SVD를 사용하여 원래 행렬을 얻기 위해 곱할 수 있는 3개의 배열을 얻을 수 있습니다. 선형대수학 이론에 따르면, 행렬 $A$가 주어졌을 때, 다음과 같은 곱을 계산할 수 있습니다:

$$U \Sigma V^T = A$$

where $U$ and $V^T$ are square and $\Sigma$ is the same size as $A$. $\Sigma$ is a diagonal matrix and contains the [singular values](https://en.wikipedia.org/wiki/Singular_value) of $A$, organized from largest to smallest. These values are always non-negative and can be used as an indicator of the "importance" of some features represented by the matrix $A$.

> 여기서 $U$와 $V^T$는 정방행렬이고 $\Sigma$는 $A$와 같은 크기입니다. $\Sigma$는 대각행렬이며 $A$의 [특이값](https://en.wikipedia.org/wiki/Singular_value)을 가장 큰 값부터 가장 작은 값 순으로 정렬하여 포함합니다. 이 값들은 항상 음이 아니며 행렬 $A$가 나타내는 일부 특징의 "중요도"의 지표로 사용될 수 있습니다.

Let's see how this works in practice with just one matrix first. Note that according to [colorimetry](https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale),
it is possible to obtain a fairly reasonable grayscale version of our color image if we apply the formula

> 먼저 하나의 행렬로 이것이 어떻게 작동하는지 실제로 살펴봅시다. [색채측정학(colorimetry)](https://en.wikipedia.org/wiki/Grayscale#Colorimetric_(perceptual_luminance-preserving)_conversion_to_grayscale)에 따르면, 다음 공식을 적용하면 우리의 컬러 이미지를 상당히 합리적인 그레이스케일 버전으로 변환할 수 있습니다:

$$Y = 0.2126 R + 0.7152 G + 0.0722 B$$

where $Y$ is the array representing the grayscale image, and $R$, $G$ and $B$ are the red, green and blue channel arrays we had originally. Notice we can use the `@` operator (the matrix multiplication operator for NumPy arrays, see [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul)) for this:

> 여기서 $Y$는 그레이스케일 이미지를 나타내는 배열이고, $R$, $G$, $B$는 원래 우리가 가지고 있던 빨강, 초록, 파랑 채널 배열입니다. 이를 위해 `@` 연산자(NumPy 배열의 행렬 곱셈 연산자, [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul) 참조)를 사용할 수 있습니다:

```{code-cell}
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
```

Now, `img_gray` has shape

> 이제 `img_gray`의 형태는 다음과 같습니다:

```{code-cell}
img_gray.shape
```

To see if this makes sense in our image, we should use a colormap from `matplotlib` corresponding to the color we wish to see in out image (otherwise, `matplotlib` will default to a colormap that does not correspond to the real data).

> 이것이 우리 이미지에서 의미가 있는지 확인하기 위해, 우리가 이미지에서 보고 싶은 색상에 해당하는 `matplotlib`의 컬러맵을 사용해야 합니다(그렇지 않으면 `matplotlib`는 실제 데이터에 해당하지 않는 기본 컬러맵을 사용합니다).

In our case, we are approximating the grayscale portion of the image, so we will use the colormap `gray`:

> 우리의 경우, 이미지의 그레이스케일 부분을 근사화하고 있으므로 `gray` 컬러맵을 사용할 것입니다:

```{code-cell}
plt.imshow(img_gray, cmap="gray")
plt.show()
```

Now, applying the [linalg.svd](https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd) function to this matrix, we obtain the following decomposition:

> 이제 이 행렬에 [linalg.svd](https://numpy.org/devdocs/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd) 함수를 적용하면 다음과 같은 분해를 얻습니다:

```{code-cell}
U, s, Vt = linalg.svd(img_gray)
```

**Note** If you are using your own image, this command might take a while to run, depending on the size of your image and your hardware. Don't worry, this is normal! The SVD can be a pretty intensive computation.

> **참고** 자신의 이미지를 사용하는 경우, 이 명령은 이미지 크기와 하드웨어에 따라 실행하는 데 시간이 걸릴 수 있습니다. 걱정하지 마세요, 이는 정상입니다! SVD는 상당히 계산 집약적인 작업일 수 있습니다.

+++

Let's check that this is what we expected:

> 예상한 결과가 맞는지 확인해 봅시다:

```{code-cell}
U.shape, s.shape, Vt.shape
```

Note that `s` has a particular shape: it has only one dimension. This means that some linear algebra functions that expect 2d arrays might not work. For example, from the theory, one might expect `s` and `Vt` to be
compatible for multiplication. However, this is not true as `s` does not have a second axis. Executing

> `s`는 특별한 형태를 가지고 있습니다: 단 하나의 차원만 있습니다. 이는 2차원 배열을 기대하는 일부 선형대수 함수가 작동하지 않을 수 있음을 의미합니다. 예를 들어, 이론적으로는 `s`와 `Vt`가 곱셈에 호환될 것으로 예상할 수 있습니다. 그러나 `s`는 두 번째 축이 없기 때문에 이는 사실이 아닙니다. 다음을 실행하면:

```python
s @ Vt
```

results in a `ValueError`. This happens because having a one-dimensional array for `s`, in this case, is much more economic in practice than building a diagonal matrix with the same data. To reconstruct the original matrix, we can rebuild the diagonal matrix $\Sigma$ with the elements of `s` in its diagonal and with the appropriate dimensions for multiplying: in our case, $\Sigma$ should be 768x1024 since `U` is 768x768 and `Vt` is 1024x1024. In order to add the singular values to the diagonal of `Sigma`, we will use the [fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html) function from NumPy:

> `ValueError`가 발생합니다. 이는 이 경우에 `s`에 대해 1차원 배열을 가지는 것이 동일한 데이터로 대각행렬을 구성하는 것보다 실제로 훨씬 더 경제적이기 때문입니다. 원래 행렬을 재구성하기 위해 `s`의 요소를 대각선에 넣어 대각행렬 $\Sigma$를 다시 구성하고 곱셈에 적합한 차원을 가지도록 할 수 있습니다: 우리의 경우, `U`는 768x768이고 `Vt`는 1024x1024이므로 $\Sigma$는 768x1024여야 합니다. `Sigma`의 대각선에 특이값을 추가하기 위해 NumPy의 [fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html) 함수를 사용할 것입니다:

```{code-cell}
import numpy as np

Sigma = np.zeros((U.shape[1], Vt.shape[0]))
np.fill_diagonal(Sigma, s)
```

Now, we want to check if the reconstructed `U @ Sigma @ Vt` is close to the original `img_gray` matrix.

> 이제 재구성된 `U @ Sigma @ Vt`가 원래의 `img_gray` 행렬과 얼마나 가까운지 확인하고자 합니다.

+++

## Approximation

The [linalg](https://numpy.org/devdocs/reference/routines.linalg.html#module-numpy.linalg) module includes a `norm` function, which computes the norm of a vector or matrix represented in a NumPy array. For example, from the SVD explanation above, we would expect the norm of the difference between `img_gray` and the reconstructed SVD product to be small. As expected, you should see something like

> ## 근사화
>
> [linalg](https://numpy.org/devdocs/reference/routines.linalg.html#module-numpy.linalg) 모듈에는 NumPy 배열로 표현된 벡터나 행렬의 노름(norm)을 계산하는 `norm` 함수가 포함되어 있습니다. 예를 들어, 위의 SVD 설명에서 우리는 `img_gray`와 재구성된 SVD 곱 사이의 차이의 노름이 작을 것으로 예상합니다. 예상대로 다음과 같은 결과가 나타날 것입니다:

```{code-cell}
linalg.norm(img_gray - U @ Sigma @ Vt)
```

(The actual result of this operation might be different depending on your architecture and linear algebra setup. Regardless, you should see a small number.)

> (이 연산의 실제 결과는 사용자의 아키텍처와 선형대수 설정에 따라 다를 수 있습니다. 그럼에도 불구하고, 작은 숫자가 표시될 것입니다.)

We could also have used the [numpy.allclose](https://numpy.org/devdocs/reference/generated/numpy.allclose.html#numpy.allclose) function to make sure the reconstructed product is, in fact, *close* to our original matrix (the difference between the two arrays is small):

> 또한 [numpy.allclose](https://numpy.org/devdocs/reference/generated/numpy.allclose.html#numpy.allclose) 함수를 사용하여 재구성된 곱이 실제로 원래 행렬과 *가까운지* 확인할 수 있습니다(두 배열 간의 차이가 작습니다):

```{code-cell}
np.allclose(img_gray, U @ Sigma @ Vt)
```

To see if an approximation is reasonable, we can check the values in `s`:

> 근사치가 합리적인지 확인하기 위해 `s`의 값을 확인할 수 있습니다:

```{code-cell}
plt.plot(s)
plt.show()
```

In the graph, we can see that although we have 768 singular values in `s`, most of those (after the 150th entry or so) are pretty small. So it might make sense to use only the information related to the first (say, 50) *singular values* to build a more economical approximation to our image.

> 그래프에서 `s`에 768개의 특이값이 있지만, 대부분(약 150번째 항목 이후)은 상당히 작다는 것을 알 수 있습니다. 따라서 우리 이미지에 대한 더 경제적인 근사치를 구성하기 위해 첫 번째(예를 들어, 50개)의 *특이값*과 관련된 정보만 사용하는 것이 타당할 수 있습니다.

The idea is to consider all but the first `k` singular values in `Sigma` (which are the same as in `s`) as zeros, keeping `U` and `Vt` intact, and computing the product of these matrices as the approximation.

> 아이디어는 `Sigma`의 첫 번째 `k`개의 특이값(이는 `s`의 값과 동일함)을 제외한 모든 값을 0으로 간주하고, `U`와 `Vt`는 그대로 유지한 후, 이러한 행렬들의 곱을 근사치로 계산하는 것입니다.

For example, if we choose

> 예를 들어, 우리가 다음과 같이 선택한다면:

```{code-cell}
k = 10
```

we can build the approximation by doing

> 다음과 같이 근사치를 구성할 수 있습니다:

```{code-cell}
approx = U @ Sigma[:, :k] @ Vt[:k, :]
```

Note that we had to use only the first `k` rows of `Vt`, since all other rows would be multiplied by the zeros corresponding to the singular values we eliminated from this approximation.

> `Vt`의 첫 번째 `k`개의 행만 사용해야 했음을 주목하세요. 이는 다른 모든 행이 이 근사치에서 제거한 특이값에 해당하는 0과 곱해지기 때문입니다.

```{code-cell}
plt.imshow(approx, cmap="gray")
plt.show()
```

Now, you can go ahead and repeat this experiment with other values of `k`, and each of your experiments should give you a slightly better (or worse) image depending on the value you choose.

> 이제 다른 `k` 값으로 이 실험을 반복해볼 수 있으며, 선택한 값에 따라 각 실험에서 약간 더 좋거나(또는 나쁜) 이미지를 얻을 수 있습니다.

+++

### Applying to all colors

Now we want to do the same kind of operation, but to all three colors. Our first instinct might be to repeat the same operation we did above to each color matrix individually. However, NumPy's *broadcasting* takes care of this
for us.

> ### 모든 색상에 적용하기
>
> 이제 우리는 같은 종류의 연산을 세 가지 색상 모두에 적용하고자 합니다. 우리의 첫 번째 직관은 위에서 수행한 것과 동일한 연산을 각 색상 행렬에 개별적으로 반복하는 것일 수 있습니다. 그러나 NumPy의 *브로드캐스팅*이 이를 처리해 줍니다.

If our array has more than two dimensions, then the SVD can be applied to all axes at once. However, the linear algebra functions in NumPy expect to see an array of the form `(n, M, N)`, where the first axis `n` represents the number of `MxN` matrices in the stack.

> 배열이 두 개 이상의 차원을 가지고 있다면, SVD는 모든 축에 한 번에 적용될 수 있습니다. 그러나 NumPy의 선형대수 함수는 `(n, M, N)` 형태의 배열을 기대하며, 여기서 첫 번째 축 `n`은 스택에 있는 `MxN` 행렬의 수를 나타냅니다.

In our case,

> 우리의 경우,

```{code-cell}
img_array.shape
```

so we need to permutate the axis on this array to get a shape like `(3, 768, 1024)`. Fortunately, the [numpy.transpose](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose) function can do that for us:
```
np.transpose(x, axes=(i, j, k))
```
indicates that the axis will be reordered such that the final shape of the transposed array will be reordered according to the indices `(i, j, k)`.

> 따라서 이 배열의 축을 순열화하여 `(3, 768, 1024)`와 같은 형태를 얻어야 합니다. 다행히도, [numpy.transpose](https://numpy.org/devdocs/reference/generated/numpy.transpose.html#numpy.transpose) 함수가 이를 수행할 수 있습니다:
> ```
> np.transpose(x, axes=(i, j, k))
> ```
> 이는 전치된 배열의 최종 형태가 인덱스 `(i, j, k)`에 따라 재정렬될 것임을 나타냅니다.

Let's see how this goes for our array:

> 우리 배열에 대해 어떻게 진행되는지 살펴봅시다:

```{code-cell}
img_array_transposed = np.transpose(img_array, (2, 0, 1))
img_array_transposed.shape
```

Now we are ready to apply the SVD:

> 이제 SVD를 적용할 준비가 되었습니다:

```{code-cell}
U, s, Vt = linalg.svd(img_array_transposed)
```

Finally, to obtain the full approximated image, we need to reassemble these matrices into the approximation. Now, note that

> 마지막으로, 완전한 근사 이미지를 얻기 위해, 우리는 이러한 행렬들을 근사치로 재조립해야 합니다. 이제, 다음을 주목하세요:

```{code-cell}
U.shape, s.shape, Vt.shape
```

To build the final approximation matrix, we must understand how multiplication across different axes works.

> 최종 근사 행렬을 구성하기 위해, 우리는 서로 다른 축에 걸친 곱셈이 어떻게 작동하는지 이해해야 합니다.

+++

### Products with n-dimensional arrays

If you have worked before with only one- or two-dimensional arrays in NumPy, you might use [numpy.dot](https://numpy.org/devdocs/reference/generated/numpy.dot.html#numpy.dot) and [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul) (or the `@` operator) interchangeably. However, for n-dimensional arrays, they work in very different ways. For more details, check the documentation on [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul).

> ### n차원 배열과의 곱셈
>
> 이전에 NumPy에서 1차원 또는 2차원 배열만 다뤘다면, [numpy.dot](https://numpy.org/devdocs/reference/generated/numpy.dot.html#numpy.dot)과 [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul)(또는 `@` 연산자)을 서로 바꿔가며 사용했을 수 있습니다. 그러나 n차원 배열의 경우, 이들은 매우 다른 방식으로 작동합니다. 자세한 내용은 [numpy.matmul](https://numpy.org/devdocs/reference/generated/numpy.matmul.html#numpy.matmul)에 관한 문서를 확인하세요.

Now, to build our approximation, we first need to make sure that our singular values are ready for multiplication, so we build our `Sigma` matrix similarly to what we did before. The `Sigma` array must have dimensions `(3, 768, 1024)`. In order to add the singular values to the diagonal of `Sigma`, we will again use the [fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html) function, using each of the 3 rows in `s` as the diagonal for each of the 3 matrices in `Sigma`:

> 이제, 우리의 근사치를 구성하기 위해, 먼저, 우리의 특이값이 곱셈에 준비되었는지 확인해야 합니다. 그래서 이전에 했던 것과 유사하게 `Sigma` 행렬을 구성합니다. `Sigma` 배열은 `(3, 768, 1024)` 차원을 가져야 합니다. `Sigma`의 대각선에 특이값을 추가하기 위해, 다시 [fill_diagonal](https://numpy.org/devdocs/reference/generated/numpy.fill_diagonal.html) 함수를 사용할 것이며, `s`의 3개 행 각각을 `Sigma`의 3개 행렬 각각의 대각선으로 사용합니다:

```{code-cell}
Sigma = np.zeros((3, 768, 1024))
for j in range(3):
    np.fill_diagonal(Sigma[j, :, :], s[j, :])
```

Now, if we wish to rebuild the full SVD (with no approximation), we can do

> 이제, 전체 SVD(근사화 없이)를 재구성하고자 한다면, 다음과 같이 할 수 있습니다:

```{code-cell}
reconstructed = U @ Sigma @ Vt
```

Note that

> 다음을 주목하세요:

```{code-cell}
reconstructed.shape
```

The reconstructed image should be indistinguishable from the original one, except for differences due to floating point errors from the reconstruction. Recall that our original image consisted of floating point values in the range `[0., 1.]`. The accumulation of floating point error from the reconstruction can result in values slightly outside this original range:

> 재구성된 이미지는 재구성에서 발생하는 부동 소수점 오류로 인한 차이를 제외하고는 원본과 구분할 수 없어야 합니다. 우리의 원본 이미지가 `[0., 1.]` 범위의 부동 소수점 값으로 구성되어 있음을 상기하세요. 재구성으로 인한 부동 소수점 오류의 축적은 이 원래 범위를 약간 벗어난 값을 초래할 수 있습니다:

```{code-cell}
reconstructed.min(), reconstructed.max()
```

Since `imshow` expects values in the range, we can use `clip` to excise the floating point error:

> `imshow`는 범위 내의 값을 기대하므로, `clip`을 사용하여 부동 소수점 오류를 제거할 수 있습니다:

```{code-cell}
reconstructed = np.clip(reconstructed, 0, 1)
plt.imshow(np.transpose(reconstructed, (1, 2, 0)))
plt.show()
```

In fact, `imshow` peforms this clipping under-the-hood, so if you skip the first line in the previous code cell, you might see a warning message saying `"Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)."`

> 사실, `imshow`는 내부적으로 이 클리핑을 수행하므로, 이전 코드 셀의 첫 번째 줄을 건너뛰면 `"Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)."`라는 경고 메시지가 표시될 수 있습니다.

Now, to do the approximation, we must choose only the first `k` singular values for each color channel. This can be done using the following syntax:

> 이제 근사치를 구하기 위해, 각 색상 채널에 대해 첫 번째 `k`개의 특이값만 선택해야 합니다. 다음 구문을 사용하여 이를 수행할 수 있습니다:

```{code-cell}
approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]
```

You can see that we have selected only the first `k` components of the last axis for `Sigma` (this means that we have used only the first `k` columns of each of the three matrices in the stack), and that we have selected only the first `k` components in the second-to-last axis of `Vt` (this means we have selected only the first `k` rows from every matrix in the stack `Vt` and all columns). If you are unfamiliar with the ellipsis syntax, it is a
placeholder for other axes. For more details, see the documentation on [Indexing](https://numpy.org/devdocs/user/basics.indexing.html#basics-indexing).

> `Sigma`의 마지막 축에서 첫 번째 `k`개의 컴포넌트만 선택했음을 알 수 있습니다(이는 스택에 있는 세 개의 행렬 각각에서 첫 번째 `k`개의 열만 사용했음을 의미합니다), 그리고 `Vt`의 마지막에서 두 번째 축에서 첫 번째 `k`개의 컴포넌트만 선택했습니다(이는 스택 `Vt`의 모든 행렬에서 첫 번째 `k`개의 행과 모든 열만 선택했음을 의미합니다). 생략 부호(ellipsis) 구문에 익숙하지 않다면, 이는 다른 축들의 자리 표시자입니다. 자세한 내용은 [인덱싱](https://numpy.org/devdocs/user/basics.indexing.html#basics-indexing)에 관한 문서를 참조하세요.

Now,

> 이제,

```{code-cell}
approx_img.shape
```

which is not the right shape for showing the image. Finally, reordering the axes back to our original shape of `(768, 1024, 3)`, we can see our approximation:

> 이는 이미지를 표시하기에 적합한 형태가 아닙니다. 마지막으로, 축을 우리의 원래 형태인 `(768, 1024, 3)`로 다시 재정렬하면 우리의 근사치를 볼 수 있습니다:

```{code-cell}
plt.imshow(np.transpose(approx_img, (1, 2, 0)))
plt.show()
```

Even though the image is not as sharp, using a small number of `k` singular values (compared to the original set of 768 values), we can recover many of the distinguishing features from this image.

> 이미지가 그렇게 선명하지는 않지만, 적은 수의 `k` 특이값(원래의 768개 값에 비해)을 사용함으로써, 우리는 이 이미지에서 많은 특징적인 요소들을 복구할 수 있습니다.

+++

### Final words

Of course, this is not the best method to *approximate* an image. However, there is, in fact, a result in linear algebra that says that the approximation we built above is the best we can get to the original matrix in
terms of the norm of the difference. For more information, see *G. H. Golub and C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University Press, 1985*.

> ### 마무리 말
>
> 물론, 이것이 이미지를 *근사화*하는 최선의 방법은 아닙니다. 그러나 실제로 선형대수학에는 우리가 위에서 구축한 근사치가 차이의 노름 측면에서 원래 행렬에 가장 가깝게 얻을 수 있는 최선의 결과라는 것을 말하는 결과가 있습니다. 자세한 정보는 *G. H. Golub and C. F. Van Loan, Matrix Computations, Baltimore, MD, Johns Hopkins University Press, 1985*를 참조하세요.

## Further reading

-  [Python tutorial](https://docs.python.org/dev/tutorial/index.html)
-  [NumPy Reference](https://numpy.org/devdocs/reference/index.html#reference)
-  [SciPy Tutorial](https://docs.scipy.org/doc/scipy/tutorial/index.html)
-  [SciPy Lecture Notes](https://scipy-lectures.org)
-  [A matlab, R, IDL, NumPy/SciPy dictionary](http://mathesaurus.sf.net/)

> ## 더 읽을거리
>
> -  [Python 튜토리얼](https://docs.python.org/dev/tutorial/index.html)
> -  [NumPy 참조](https://numpy.org/devdocs/reference/index.html#reference)
> -  [SciPy 튜토리얼](https://docs.scipy.org/doc/scipy/tutorial/index.html)
> -  [SciPy 강의 노트](https://scipy-lectures.org)
> -  [Matlab, R, IDL, NumPy/SciPy 사전](http://mathesaurus.sf.net/)