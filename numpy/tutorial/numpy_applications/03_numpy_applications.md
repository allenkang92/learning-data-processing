# X-ray image processing
# X선 이미지 처리

+++

This tutorial demonstrates how to read and process X-ray images with NumPy,
imageio, Matplotlib and SciPy. You will learn how to load medical images, focus
on certain parts, and visually compare them using the
[Gaussian](https://en.wikipedia.org/wiki/Gaussian_filter),
[Laplacian-Gaussian](https://en.wikipedia.org/wiki/Laplace_distribution),
[Sobel](https://en.wikipedia.org/wiki/Sobel_operator), and
[Canny](https://en.wikipedia.org/wiki/Canny_edge_detector) filters for edge
detection.

이 튜토리얼은 NumPy, imageio, Matplotlib 및 SciPy를 사용하여 X선 이미지를 읽고 처리하는 방법을 보여줍니다. 의료 이미지를 로드하고, 특정 부분에 초점을 맞추며, [가우시안](https://en.wikipedia.org/wiki/Gaussian_filter), [라플라시안-가우시안](https://en.wikipedia.org/wiki/Laplace_distribution), [소벨](https://en.wikipedia.org/wiki/Sobel_operator), [캐니](https://en.wikipedia.org/wiki/Canny_edge_detector) 필터를 사용한 에지 검출을 통해 시각적으로 비교하는 방법을 배우게 됩니다.

X-ray image analysis can be part of your data analysis and
[machine learning workflow](https://www.sciencedirect.com/science/article/pii/S235291481930214X)
when, for example, you're building an algorithm that helps
[detect pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
as part of a [Kaggle](https://www.kaggle.com)
[competition](https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen).
In the healthcare industry, medical image processing and analysis is
particularly important when images are estimated to account for
[at least 90%](https://www-03.ibm.com/press/us/en/pressrelease/51146.wss) of all
medical data.

X선 이미지 분석은 예를 들어 [Kaggle](https://www.kaggle.com) [대회](https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen)의 일환으로 [폐렴 감지](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)를 돕는 알고리즘을 개발할 때 데이터 분석 및 [기계 학습 워크플로우](https://www.sciencedirect.com/science/article/pii/S235291481930214X)의 일부가 될 수 있습니다. 의료 산업에서 의료 이미지 처리 및 분석은 이미지가 모든 의료 데이터의 [최소 90%](https://www-03.ibm.com/press/us/en/pressrelease/51146.wss)를 차지하는 것으로 추정되는 만큼 특히 중요합니다.

You'll be working with radiology images from the
[ChestX-ray8](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
dataset provided by the [National Institutes of Health (NIH)](http://nih.gov).
ChestX-ray8 contains over 100,000 de-identified X-ray images in the PNG format
from more than 30,000 patients. You can find ChestX-ray8's files on NIH's public
Box [repository](https://nihcc.app.box.com/v/ChestXray-NIHCC) in the `/images`
folder. (For more details, refer to the research
[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)
published at CVPR (a computer vision conference) in 2017.)

[국립 보건원(NIH)](http://nih.gov)에서 제공하는 [ChestX-ray8](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) 데이터셋의 방사선 이미지를 사용하게 됩니다. ChestX-ray8에는 30,000명 이상의 환자로부터 얻은 100,000개 이상의 익명화된 PNG 형식의 X선 이미지가 포함되어 있습니다. NIH의 공개 Box [저장소](https://nihcc.app.box.com/v/ChestXray-NIHCC)의 `/images` 폴더에서 ChestX-ray8 파일을 찾을 수 있습니다. (자세한 내용은 2017년 CVPR(컴퓨터 비전 컨퍼런스)에서 발표된 연구 [논문](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)을 참조하세요.)

For your convenience, a small number of PNG images have been saved to this
tutorial's repository under `tutorial-x-ray-image-processing/`, since
ChestX-ray8 contains gigabytes of data and you may find it challenging to
download it in batches.

ChestX-ray8는 기가바이트 단위의 데이터를 포함하고 있어 배치로 다운로드하기 어려울 수 있으므로, 편의를 위해 소수의 PNG 이미지가 이 튜토리얼의 저장소 `tutorial-x-ray-image-processing/` 아래에 저장되어 있습니다.

![A series of 9 x-ray images of the same region of a patient's chest is shown with different types of image processing filters applied to each image. Each x-ray shows different types of biological detail.](_static/tutorial-x-ray-image-processing.png)

![환자 흉부의 동일한 부위에 대한 9개의 X선 이미지 시리즈가 각 이미지에 적용된 다양한 유형의 이미지 처리 필터와 함께 표시됩니다. 각 X선은 다양한 유형의 생물학적 세부 사항을 보여줍니다.](_static/tutorial-x-ray-image-processing.png)

+++

## Prerequisites
## 사전 요구사항

+++

The reader should have some knowledge of Python, NumPy arrays, and Matplotlib.
To refresh the memory, you can take the
[Python](https://docs.python.org/dev/tutorial/index.html) and Matplotlib
[PyPlot](https://matplotlib.org/tutorials/introductory/pyplot.html) tutorials,
and the NumPy [quickstart](https://numpy.org/devdocs/user/quickstart.html).

독자는 Python, NumPy 배열 및 Matplotlib에 대한 기본 지식이 있어야 합니다. 기억을 되살리기 위해 [Python](https://docs.python.org/dev/tutorial/index.html) 및 Matplotlib [PyPlot](https://matplotlib.org/tutorials/introductory/pyplot.html) 튜토리얼, 그리고 NumPy [퀵스타트](https://numpy.org/devdocs/user/quickstart.html)를 참고할 수 있습니다.

The following packages are used in this tutorial:

이 튜토리얼에서는 다음 패키지들이 사용됩니다:

- [imageio](https://imageio.github.io) for reading and writing image data. The
healthcare industry usually works with the
[DICOM](https://en.wikipedia.org/wiki/DICOM) format for medical imaging and
[imageio](https://imageio.readthedocs.io/en/stable/format_dicom.html) should be
well-suited for reading that format. For simplicity, in this tutorial you'll be
working with PNG files.
- [Matplotlib](https://matplotlib.org/) for data visualization.
- [SciPy](https://www.scipy.org) for multi-dimensional image processing via
[`ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html).

- [imageio](https://imageio.github.io): 이미지 데이터 읽기 및 쓰기용. 의료 산업에서는 일반적으로 의료 영상을 위한 [DICOM](https://en.wikipedia.org/wiki/DICOM) 형식을 사용하며, [imageio](https://imageio.readthedocs.io/en/stable/format_dicom.html)는 해당 형식을 읽는 데 적합합니다. 단순화를 위해 이 튜토리얼에서는 PNG 파일을 사용합니다.
- [Matplotlib](https://matplotlib.org/): 데이터 시각화용.
- [SciPy](https://www.scipy.org): [`ndimage`](https://docs.scipy.org/doc/scipy/reference/ndimage.html)를 통한 다차원 이미지 처리용.

This tutorial can be run locally in an isolated environment, such as
[Virtualenv](https://virtualenv.pypa.io/en/stable/) or
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
You can use [Jupyter Notebook or JupyterLab](https://jupyter.org/install) to run
each notebook cell.

이 튜토리얼은 [Virtualenv](https://virtualenv.pypa.io/en/stable/) 또는 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)와 같은 격리된 환경에서 로컬로 실행할 수 있습니다. [Jupyter Notebook 또는 JupyterLab](https://jupyter.org/install)을 사용하여 각 노트북 셀을 실행할 수 있습니다.

+++

## Table of contents
## 목차

+++

1. Examine an X-ray with `imageio`
2. Combine images into a multi-dimensional array to demonstrate progression
3. Edge detection using the Laplacian-Gaussian, Gaussian gradient, Sobel, and
   Canny filters
4. Apply masks to X-rays with `np.where()`
5. Compare the results

1. `imageio`로 X선 검사하기
2. 진행 상황을 보여주기 위해 이미지를 다차원 배열로 결합하기
3. 라플라시안-가우시안, 가우시안 그래디언트, 소벨, 캐니 필터를 사용한 에지 검출
4. `np.where()`를 사용하여 X선에 마스크 적용하기
5. 결과 비교하기

---

+++

## Examine an X-ray with `imageio`
## `imageio`로 X선 검사하기

+++

Let's begin with a simple example using just one X-ray image from the
ChestX-ray8 dataset.

ChestX-ray8 데이터셋에서 하나의 X선 이미지만 사용하는 간단한 예제로 시작해 보겠습니다.

The file — `00000011_001.png` — has been downloaded for you and saved in the
`/tutorial-x-ray-image-processing` folder.

파일 — `00000011_001.png` — 이 다운로드되어 `/tutorial-x-ray-image-processing` 폴더에 저장되어 있습니다.

+++

**1.** Load the image with `imageio`:

**1.** `imageio`로 이미지 로드하기:

```{code-cell}
import os
import imageio

DIR = "tutorial-x-ray-image-processing"

xray_image = imageio.v3.imread(os.path.join(DIR, "00000011_001.png"))
```

**2.** Check that its shape is 1024x1024 pixels and that the array is made up of
8-bit integers:

**2.** 모양이 1024x1024 픽셀이고 배열이 8비트 정수로 구성되어 있는지 확인하기:

```{code-cell}
print(xray_image.shape)
print(xray_image.dtype)
```

**3.** Import `matplotlib` and display the image in a grayscale colormap:

**3.** `matplotlib`를 가져와서 그레이스케일 컬러맵으로 이미지 표시하기:

```{code-cell}
import matplotlib.pyplot as plt

plt.imshow(xray_image, cmap="gray")
plt.axis("off")
plt.show()
```

## Combine images into a multidimensional array to demonstrate progression
## 진행 상황을 보여주기 위해 이미지를 다차원 배열로 결합하기

+++

In the next example, instead of 1 image you'll use 9 X-ray 1024x1024-pixel
images from the ChestX-ray8 dataset that have been downloaded and extracted
from one of the dataset files. They are numbered from `...000.png` to
`...008.png` and let's assume they belong to the same patient.

다음 예제에서는 1개의 이미지 대신 ChestX-ray8 데이터셋에서 다운로드하고 추출한 9개의 X선 1024x1024 픽셀 이미지를 사용합니다. 이것들은 `...000.png`부터 `...008.png`까지 번호가 매겨져 있으며 같은 환자의 것이라고 가정해 보겠습니다.

**1.** Import NumPy, read in each of the X-rays, and create a three-dimensional
array where the first dimension corresponds to image number:

**1.** NumPy를 가져와 각 X선을 읽고, 첫 번째 차원이 이미지 번호에 해당하는 3차원 배열 생성하기:

```{code-cell}
import numpy as np
num_imgs = 9

combined_xray_images_1 = np.array(
    [imageio.v3.imread(os.path.join(DIR, f"00000011_00{i}.png")) for i in range(num_imgs)]
)
```

**2.** Check the shape of the new X-ray image array containing 9 stacked images:

**2.** 9개의 적층된 이미지가 포함된 새 X선 이미지 배열의 모양 확인하기:

```{code-cell}
combined_xray_images_1.shape
```

Note that the shape in the first dimension matches `num_imgs`, so the
`combined_xray_images_1` array can be interpreted as a stack of 2D images.

첫 번째 차원의 모양이 `num_imgs`와 일치하므로, `combined_xray_images_1` 배열은 2D 이미지의 스택으로 해석될 수 있습니다.

**3.** You can now display the "health progress" by plotting each of frames next
to each other using Matplotlib:

**3.** Matplotlib을 사용하여 각 프레임을 나란히 표시하여 "건강 진행 상황"을 보여줄 수 있습니다:

```{code-cell} ipython3
fig, axes = plt.subplots(nrows=1, ncols=num_imgs, figsize=(30, 30))

for img, ax in zip(combined_xray_images_1, axes):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
```

**4.** In addition, it can be helpful to show the progress as an animation.
Let's create a GIF file with `imageio.mimwrite()` and display the result in the
notebook:

**4.** 또한, 진행 상황을 애니메이션으로 보여주는 것이 도움이 될 수 있습니다. `imageio.mimwrite()`로 GIF 파일을 만들어 노트북에 결과를 표시해 보겠습니다:

```{code-cell} ipython3
GIF_PATH = os.path.join(DIR, "xray_image.gif")
imageio.mimwrite(GIF_PATH, combined_xray_images_1, format= ".gif", duration=1000)
```

Which gives us:
![An animated gif repeatedly cycles through a series of 8 x-rays, showing the same viewpoint of the patient's chest at different points in time. The patient's bones and internal organs can be visually compared from frame to frame.](tutorial-x-ray-image-processing/xray_image.gif)

결과는 다음과 같습니다:
![애니메이션 GIF가 8개의 X선 시리즈를 반복적으로 순환하면서 서로 다른 시점에서 환자의 흉부를 동일한 관점에서 보여줍니다. 환자의 뼈와 내부 장기를 프레임별로 시각적으로 비교할 수 있습니다.](tutorial-x-ray-image-processing/xray_image.gif)

## Edge detection using the Laplacian-Gaussian, Gaussian gradient, Sobel, and Canny filters
## 라플라시안-가우시안, 가우시안 그래디언트, 소벨, 캐니 필터를 사용한 에지 검출

+++

When processing biomedical data, it can be useful to emphasize the 2D
["edges"](https://en.wikipedia.org/wiki/Edge_detection) to focus on particular
features in an image. To do that, using
[image gradients](https://en.wikipedia.org/wiki/Image_gradient) can be
particularly helpful when detecting the change of color pixel intensity.

생체 의학 데이터를 처리할 때 2D ["에지"](https://en.wikipedia.org/wiki/Edge_detection)를 강조하여 이미지의 특정 특징에 초점을 맞추는 것이 유용할 수 있습니다. 이를 위해 색상 픽셀 강도의 변화를 감지할 때 [이미지 그래디언트](https://en.wikipedia.org/wiki/Image_gradient)를 사용하는 것이 특히 도움이 됩니다.

+++

### The Laplace filter with Gaussian second derivatives
### 가우시안 2차 미분을 이용한 라플라스 필터

Let's start with an n-dimensional
[Laplace](https://en.wikipedia.org/wiki/Laplace_distribution) filter
("Laplacian-Gaussian") that uses
[Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) second
derivatives. This Laplacian method focuses on pixels with rapid intensity change
in values and is combined with Gaussian smoothing to
[remove noise](https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm). Let's examine
how it can be useful in analyzing 2D X-ray images.

[가우시안](https://en.wikipedia.org/wiki/Normal_distribution) 2차 미분을 사용하는 n차원 [라플라스](https://en.wikipedia.org/wiki/Laplace_distribution) 필터("라플라시안-가우시안")부터 시작해 보겠습니다. 이 라플라시안 방법은 값의 급격한 강도 변화가 있는 픽셀에 초점을 맞추고 가우시안 스무딩과 결합하여 [노이즈를 제거](https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm)합니다. 2D X선 이미지 분석에 어떻게 유용할 수 있는지 살펴보겠습니다.

+++

- The implementation of the Laplacian-Gaussian filter is relatively
straightforward: 1) import the `ndimage` module from SciPy; and 2) call
[`scipy.ndimage.gaussian_laplace()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html)
with a sigma (scalar) parameter, which affects the standard deviations of the
Gaussian filter (you'll use `1` in the example below):

- 라플라시안-가우시안 필터의 구현은 비교적 간단합니다: 1) SciPy에서 `ndimage` 모듈을 가져오고, 2) 가우시안 필터의 표준 편차에 영향을 미치는 시그마(스칼라) 매개변수와 함께 [`scipy.ndimage.gaussian_laplace()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_laplace.html)를 호출합니다(아래 예제에서는 `1`을 사용합니다):

```{code-cell}
from scipy import ndimage

xray_image_laplace_gaussian = ndimage.gaussian_laplace(xray_image, sigma=1)
```

Display the original X-ray and the one with the Laplacian-Gaussian filter:

원본 X선과 라플라시안-가우시안 필터를 적용한 X선 표시하기:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Laplacian-Gaussian (edges)")
axes[1].imshow(xray_image_laplace_gaussian, cmap="gray")
for i in axes:
    i.axis("off")
plt.show()
```

### The Gaussian gradient magnitude method
### 가우시안 그래디언트 크기 방법

Another method for edge detection that can be useful is the
[Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) (gradient) filter.
It computes the multidimensional gradient magnitude with Gaussian derivatives
and helps by remove
[high-frequency](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf)
image components.

에지 검출에 유용할 수 있는 또 다른 방법은 [가우시안](https://en.wikipedia.org/wiki/Normal_distribution)(그래디언트) 필터입니다. 이는 가우시안 미분을 사용하여 다차원 그래디언트 크기를 계산하고 [고주파](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf) 이미지 구성 요소를 제거하는 데 도움이 됩니다.

+++

**1.** Call [`scipy.ndimage.gaussian_gradient_magnitude()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html)
with a sigma (scalar) parameter (for standard deviations; you'll use `2` in the
example below):

**1.** 시그마(스칼라) 매개변수(표준 편차용; 아래 예제에서는 `2` 사용)와 함께 [`scipy.ndimage.gaussian_gradient_magnitude()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_gradient_magnitude.html) 호출하기:

```{code-cell}
x_ray_image_gaussian_gradient = ndimage.gaussian_gradient_magnitude(xray_image, sigma=2)
```

**2.** Display the original X-ray and the one with the Gaussian gradient filter:

**2.** 원본 X선과 가우시안 그래디언트 필터를 적용한 X선 표시하기:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Gaussian gradient (edges)")
axes[1].imshow(x_ray_image_gaussian_gradient, cmap="gray")
for i in axes:
    i.axis("off")
plt.show()
```

### The Sobel-Feldman operator (the Sobel filter)
### 소벨-펠드만 연산자(소벨 필터)

To find regions of high spatial frequency (the edges or the edge maps) along the
horizontal and vertical axes of a 2D X-ray image, you can use the
[Sobel-Feldman operator (Sobel filter)](https://en.wikipedia.org/wiki/Sobel_operator)
technique. The Sobel filter applies two 3x3 kernel matrices — one for each axis
— onto the X-ray through a [convolution](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution).
Then, these two points (gradients) are combined using the
[Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem) to
produce a gradient magnitude.

2D X선 이미지의 수평 및 수직 축을 따라 높은 공간 주파수 영역(에지 또는 에지 맵)을 찾기 위해 [소벨-펠드만 연산자(소벨 필터)](https://en.wikipedia.org/wiki/Sobel_operator) 기법을 사용할 수 있습니다. 소벨 필터는 두 개의 3x3 커널 행렬 — 각 축당 하나씩 — 을 [컨볼루션](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution)을 통해 X선에 적용합니다. 그런 다음 이 두 지점(그래디언트)은 [피타고라스 정리](https://en.wikipedia.org/wiki/Pythagorean_theorem)를 사용하여 결합되어 그래디언트 크기를 생성합니다.

+++

**1.** Use the Sobel filters — ([`scipy.ndimage.sobel()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html))
— on x- and y-axes of the X-ray. Then, calculate the distance between `x` and
`y` (with the Sobel filters applied to them) using the
[Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem) and
NumPy's [`np.hypot()`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html)
to obtain the magnitude. Finally, normalize the rescaled image for the pixel
values to be between 0 and 255.

**1.** X선의 x축과 y축에 소벨 필터 — ([`scipy.ndimage.sobel()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html)) — 를 사용합니다. 그런 다음, [피타고라스 정리](https://en.wikipedia.org/wiki/Pythagorean_theorem)와 NumPy의 [`np.hypot()`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html)를 사용하여 `x`와 `y`(소벨 필터가 적용된) 사이의 거리를 계산하여 크기를 얻습니다. 마지막으로, 픽셀 값이 0과 255 사이가 되도록 재조정된 이미지를 정규화합니다.

[Image normalization](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)
follows the `output_channel = 255.0 * (input_channel - min_value) / (max_value - min_value)`
[formula](http://dev.ipol.im/~nmonzon/Normalization.pdf). Because you're
using a grayscale image, you need to normalize just one channel.

[이미지 정규화](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)는 `output_channel = 255.0 * (input_channel - min_value) / (max_value - min_value)` [공식](http://dev.ipol.im/~nmonzon/Normalization.pdf)을 따릅니다. 그레이스케일 이미지를 사용하고 있으므로 한 채널만 정규화하면 됩니다.

```{code-cell}
x_sobel = ndimage.sobel(xray_image, axis=0)
y_sobel = ndimage.sobel(xray_image, axis=1)

xray_image_sobel = np.hypot(x_sobel, y_sobel)

xray_image_sobel *= 255.0 / np.max(xray_image_sobel)
```

**2.** Change the new image array data type to the 32-bit floating-point format
from `float16` to [make it compatible](https://github.com/matplotlib/matplotlib/issues/15432)
with Matplotlib:

**2.** 새 이미지 배열 데이터 유형을 `float16`에서 32비트 부동소수점 형식으로 변경하여 Matplotlib과 [호환되도록 만들기](https://github.com/matplotlib/matplotlib/issues/15432):

```{code-cell}
print("The data type - before: ", xray_image_sobel.dtype)

xray_image_sobel = xray_image_sobel.astype("float32")

print("The data type - after: ", xray_image_sobel.dtype)
```

**3.** Display the original X-ray and the one with the Sobel "edge" filter
applied. Note that both the grayscale and `CMRmap` colormaps are used to help
emphasize the edges:

**3.** 원본 X선과 소벨 "에지" 필터가 적용된 X선 표시하기. 에지를 강조하기 위해 그레이스케일과 `CMRmap` 컬러맵이 모두 사용된다는 점에 유의하세요:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Sobel (edges) - grayscale")
axes[1].imshow(xray_image_sobel, cmap="gray")
axes[2].set_title("Sobel (edges) - CMRmap")
axes[2].imshow(xray_image_sobel, cmap="CMRmap")
for i in axes:
    i.axis("off")
plt.show()
```

### The Canny filter
### 캐니 필터

You can also consider using another well-known filter for edge detection called
the [Canny filter](https://en.wikipedia.org/wiki/Canny_edge_detector).

에지 검출을 위한 또 다른 잘 알려진 필터인 [캐니 필터](https://en.wikipedia.org/wiki/Canny_edge_detector)를 사용하는 것도 고려할 수 있습니다.

First, you apply a [Gaussian](https://en.wikipedia.org/wiki/Gaussian_filter)
filter to remove the noise in an image. In this example, you're using using the
[Fourier](https://en.wikipedia.org/wiki/Fourier_transform) filter which
smoothens the X-ray through a [convolution](https://en.wikipedia.org/wiki/Convolution)
process. Next, you apply the [Prewitt filter](https://en.wikipedia.org/wiki/Prewitt_operator)
on each of the 2 axes of the image to help detect some of the edges — this will
result in 2 gradient values. Similar to the Sobel filter, the Prewitt operator
also applies two 3x3 kernel matrices — one for each axis — onto the X-ray
through a [convolution](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution).
In the end, you compute the magnitude between the two gradients using the
[Pythagorean theorem](https://en.wikipedia.org/wiki/Pythagorean_theorem) and
[normalize](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)
the images, as before.

먼저, [가우시안](https://en.wikipedia.org/wiki/Gaussian_filter) 필터를 적용하여 이미지의 노이즈를 제거합니다. 이 예제에서는 [푸리에](https://en.wikipedia.org/wiki/Fourier_transform) 필터를 사용하여 [컨볼루션](https://en.wikipedia.org/wiki/Convolution) 과정을 통해 X선을 부드럽게 합니다. 다음으로, 이미지의 2개 축 각각에 [프리윗 필터](https://en.wikipedia.org/wiki/Prewitt_operator)를 적용하여 일부 에지 검출을 돕습니다 — 이는 2개의 그래디언트 값을 생성합니다. 소벨 필터와 유사하게, 프리윗 연산자도 두 개의 3x3 커널 행렬 — 각 축당 하나씩 — 을 [컨볼루션](https://en.wikipedia.org/wiki/Kernel_(image_processing)#Convolution)을 통해 X선에 적용합니다. 마지막으로, [피타고라스 정리](https://en.wikipedia.org/wiki/Pythagorean_theorem)를 사용하여 두 그래디언트 사이의 크기를 계산하고 이전과 같이 이미지를 [정규화](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)합니다.

+++

**1.** Use SciPy's Fourier filters — [`scipy.ndimage.fourier_gaussian()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_gaussian.html)
— with a small `sigma` value to remove some of the noise from the X-ray. Then,
calculate two gradients using [`scipy.ndimage.prewitt()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.prewitt.html).
Next, measure the distance between the gradients using NumPy's `np.hypot()`.
Finally, [normalize](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)
the rescaled image, as before.

**1.** 작은 `sigma` 값과 함께 SciPy의 푸리에 필터 — [`scipy.ndimage.fourier_gaussian()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.fourier_gaussian.html) — 를 사용하여 X선에서 일부 노이즈를 제거합니다. 그런 다음, [`scipy.ndimage.prewitt()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.prewitt.html)를 사용하여 두 개의 그래디언트를 계산합니다. 다음으로, NumPy의 `np.hypot()`를 사용하여 그래디언트 간의 거리를 측정합니다. 마지막으로, 이전과 같이 재조정된 이미지를 [정규화](https://en.wikipedia.org/wiki/Normalization_%28image_processing%29)합니다.

```{code-cell}
fourier_gaussian = ndimage.fourier_gaussian(xray_image, sigma=0.05)

x_prewitt = ndimage.prewitt(fourier_gaussian, axis=0)
y_prewitt = ndimage.prewitt(fourier_gaussian, axis=1)

xray_image_canny = np.hypot(x_prewitt, y_prewitt)

xray_image_canny *= 255.0 / np.max(xray_image_canny)

print("The data type - ", xray_image_canny.dtype)
```

**2.** Plot the original X-ray image and the ones with the edges detected with
the help of the Canny filter technique. The edges can be emphasized using the
`prism`, `nipy_spectral`, and `terrain` Matplotlib colormaps.

**2.** 원본 X선 이미지와 캐니 필터 기법의 도움으로 에지가 검출된 이미지들을 표시합니다. 에지는 `prism`, `nipy_spectral`, `terrain` Matplotlib 컬러맵을 사용하여 강조할 수 있습니다.

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 15))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Canny (edges) - prism")
axes[1].imshow(xray_image_canny, cmap="prism")
axes[2].set_title("Canny (edges) - nipy_spectral")
axes[2].imshow(xray_image_canny, cmap="nipy_spectral")
axes[3].set_title("Canny (edges) - terrain")
axes[3].imshow(xray_image_canny, cmap="terrain")
for i in axes:
    i.axis("off")
plt.show()
```

## Apply masks to X-rays with `np.where()`
## `np.where()`를 사용하여 X선에 마스크 적용하기

+++

To screen out only certain pixels in X-ray images to help detect particular
features, you can apply masks with NumPy's
[`np.where(condition: array_like (bool), x: array_like, y: ndarray)`](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
that returns `x` when `True` and `y` when `False`.

X선 이미지에서 특정 특징을 감지하는 데 도움이 되도록 특정 픽셀만 선별하기 위해 NumPy의 [`np.where(condition: array_like (bool), x: array_like, y: ndarray)`](https://numpy.org/doc/stable/reference/generated/numpy.where.html)를 사용하여 마스크를 적용할 수 있습니다. 이 함수는 `조건`이 `True`일 때 `x`를 반환하고, `False`일 때 `y`를 반환합니다.

Identifying regions of interest — certain sets of pixels in an image — can be
useful and masks serve as boolean arrays of the same shape as the original
image.

관심 영역 — 이미지의 특정 픽셀 집합 — 을 식별하는 것은 유용할 수 있으며, 마스크는 원본 이미지와 동일한 모양의 부울 배열 역할을 합니다.

+++

**1.** Retrieve some basics statistics about the pixel values in the original
X-ray image you've been working with:

**1.** 지금까지 작업한 원본 X선 이미지의 픽셀 값에 대한 기본 통계 정보 검색하기:

```{code-cell}
print("The data type of the X-ray image is: ", xray_image.dtype)
print("The minimum pixel value is: ", np.min(xray_image))
print("The maximum pixel value is: ", np.max(xray_image))
print("The average pixel value is: ", np.mean(xray_image))
print("The median pixel value is: ", np.median(xray_image))
```

**2.** The array data type is `uint8` and the minimum/maximum value results
suggest that all 256 colors (from `0` to `255`) are used in the X-ray. Let's
visualize the _pixel intensity distribution_ of the original raw X-ray image
with `ndimage.histogram()` and Matplotlib:

**2.** 배열 데이터 유형은 `uint8`이며 최소/최대 값 결과는 X선에서 256색(`0`부터 `255`까지) 모두 사용됨을 시사합니다. `ndimage.histogram()`과 Matplotlib을 사용하여 원본 X선 이미지의 _픽셀 강도 분포_를 시각화해 보겠습니다:

```{code-cell}
pixel_intensity_distribution = ndimage.histogram(
    xray_image, min=np.min(xray_image), max=np.max(xray_image), bins=256
)

plt.plot(pixel_intensity_distribution)
plt.title("Pixel intensity distribution")
plt.show()
```

As the pixel intensity distribution suggests, there are many low (between around
0 and 20) and very high (between around 200 and 240) pixel values.

픽셀 강도 분포가 시사하는 바와 같이, 많은 낮은(약 0과 20 사이) 및 매우 높은(약 200과 240 사이) 픽셀 값이 있습니다.

**3.** You can create different conditional masks with NumPy's `np.where()` —
for example, let's have only those values of the image with the pixels exceeding
a certain threshold:

**3.** NumPy의 `np.where()`로 다양한 조건부 마스크를 만들 수 있습니다 — 예를 들어, 특정 임계값을 초과하는 픽셀이 있는 이미지 값만 가져보겠습니다:

```{code-cell}
# The threshold is "greater than 150"
# Return the original image if true, `0` otherwise
xray_image_mask_noisy = np.where(xray_image > 150, xray_image, 0)

plt.imshow(xray_image_mask_noisy, cmap="gray")
plt.axis("off")
plt.show()
```

```{code-cell}
# The threshold is "greater than 150"
# Return `1` if true, `0` otherwise
xray_image_mask_less_noisy = np.where(xray_image > 150, 1, 0)

plt.imshow(xray_image_mask_less_noisy, cmap="gray")
plt.axis("off")
plt.show()
```

## Compare the results
## 결과 비교하기

+++

Let's display some of the results of processed X-ray images you've worked with
so far:

지금까지 작업한 처리된 X선 이미지의 일부 결과를 표시해 보겠습니다:

```{code-cell}
fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(30, 30))

axes[0].set_title("Original")
axes[0].imshow(xray_image, cmap="gray")
axes[1].set_title("Laplace-Gaussian (edges)")
axes[1].imshow(xray_image_laplace_gaussian, cmap="gray")
axes[2].set_title("Gaussian gradient (edges)")
axes[2].imshow(x_ray_image_gaussian_gradient, cmap="gray")
axes[3].set_title("Sobel (edges) - grayscale")
axes[3].imshow(xray_image_sobel, cmap="gray")
axes[4].set_title("Sobel (edges) - hot")
axes[4].imshow(xray_image_sobel, cmap="hot")
axes[5].set_title("Canny (edges) - prism)")
axes[5].imshow(xray_image_canny, cmap="prism")
axes[6].set_title("Canny (edges) - nipy_spectral)")
axes[6].imshow(xray_image_canny, cmap="nipy_spectral")
axes[7].set_title("Mask (> 150, noisy)")
axes[7].imshow(xray_image_mask_noisy, cmap="gray")
axes[8].set_title("Mask (> 150, less noisy)")
axes[8].imshow(xray_image_mask_less_noisy, cmap="gray")
for i in axes:
    i.axis("off")
plt.show()
```

## Next steps
## 다음 단계

+++

If you want to use your own samples, you can use
[this image](https://openi.nlm.nih.gov/detailedresult?img=CXR3666_IM-1824-1001&query=chest%20infection&it=xg&req=4&npos=32)
or search for various other ones on the [_Openi_](https://openi.nlm.nih.gov)
database. Openi contains many biomedical images and it can be especially helpful
if you have low bandwidth and/or are restricted by the amount of data you can
download.

자신만의 샘플을 사용하고 싶다면 [이 이미지](https://openi.nlm.nih.gov/detailedresult?img=CXR3666_IM-1824-1001&query=chest%20infection&it=xg&req=4&npos=32)를 사용하거나 [_Openi_](https://openi.nlm.nih.gov) 데이터베이스에서 다양한 다른 이미지를 검색할 수 있습니다. Openi에는 많은 생체 의학 이미지가 포함되어 있으며, 대역폭이 낮거나 다운로드할 수 있는 데이터 양에 제한이 있는 경우 특히 유용할 수 있습니다.

To learn more about image processing in the context of biomedical image data or
simply edge detection, you may find the following material useful:

생체 의학 이미지 데이터 맥락에서의 이미지 처리 또는 단순히 에지 검출에 대해 더 자세히 알아보려면 다음 자료가 유용할 수 있습니다:

- [DICOM processing and segmentation in Python](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/) with Scikit-Image and pydicom (Radiology Data Quest)
- [Image manipulation and processing using Numpy and Scipy](https://scipy-lectures.org/advanced/image_processing/index.html) (Scipy Lecture Notes)
- [Intensity values](https://s3.amazonaws.com/assets.datacamp.com/production/course_7032/slides/chapter2.pdf) (presentation, DataCamp)
- [Object detection with Raspberry Pi and Python](https://makersportal.com/blog/2019/4/23/image-processing-with-raspberry-pi-and-python-part-ii-spatial-statistics-and-correlations) (Maker Portal)
- [X-ray data preparation and segmentation](https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen) with deep learning (a Kaggle-hosted Jupyter notebook)
- [Image filtering](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf) (lecture slides, CS6670: Computer Vision, Cornell University)
- [Edge detection in Python](https://towardsdatascience.com/edge-detection-in-python-a3c263a13e03) and NumPy (Towards Data Science)
- [Edge detection](https://datacarpentry.org/image-processing/08-edge-detection/) with Scikit-Image (Data Carpentry)
- [Image gradients and gradient filtering](https://www.cs.cmu.edu/~16385/s17/Slides/4.0_Image_Gradients_and_Gradient_Filtering.pdf) (lecture slides, 16-385 Computer Vision, Carnegie Mellon University)

- Scikit-Image와 pydicom을 사용한 [Python에서 DICOM 처리 및 분할](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/) (Radiology Data Quest)
- [Numpy와 Scipy를 사용한 이미지 조작 및 처리](https://scipy-lectures.org/advanced/image_processing/index.html) (Scipy Lecture Notes)
- [강도 값](https://s3.amazonaws.com/assets.datacamp.com/production/course_7032/slides/chapter2.pdf) (프레젠테이션, DataCamp)
- [Raspberry Pi와 Python을 사용한 객체 검출](https://makersportal.com/blog/2019/4/23/image-processing-with-raspberry-pi-and-python-part-ii-spatial-statistics-and-correlations) (Maker Portal)
- 딥 러닝을 통한 [X선 데이터 준비 및 분할](https://www.kaggle.com/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen) (Kaggle에서 호스팅하는 Jupyter 노트북)
- [이미지 필터링](https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf) (강의 슬라이드, CS6670: Computer Vision, Cornell University)
- Python과 NumPy를 사용한 [에지 검출](https://towardsdatascience.com/edge-detection-in-python-a3c263a13e03) (Towards Data Science)
- Scikit-Image를 사용한 [에지 검출](https://datacarpentry.org/image-processing/08-edge-detection/) (Data Carpentry)
- [이미지 그래디언트 및 그래디언트 필터링](https://www.cs.cmu.edu/~16385/s17/Slides/4.0_Image_Gradients_and_Gradient_Filtering.pdf) (강의 슬라이드, 16-385 Computer Vision, Carnegie Mellon University)