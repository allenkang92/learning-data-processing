# Deep learning on MNIST
# MNIST에서의 딥러닝

## Table of contents
## 목차

1. Load the MNIST dataset  
   1. MNIST 데이터셋 로드하기

2. Preprocess the dataset  
   2. 데이터셋 전처리하기

3. Build and train a small neural network from scratch  
   3. 작은 신경망을 처음부터 구축하고 훈련하기

4. Next steps  
   4. 다음 단계

---

## 1. Load the MNIST dataset
## 1. MNIST 데이터셋 로드하기

In this section, you will download the zipped MNIST dataset files originally developed by Yann LeCun's research team. (More details of the MNIST dataset are available on [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).) Then, you will transform them into 4 files of NumPy array type using built-in Python modules. Finally, you will split the arrays into training and test sets.
  
이 섹션에서는 Yann LeCun의 연구팀이 원래 개발한 압축된 MNIST 데이터셋 파일을 다운로드한 후, 내장 Python 모듈을 활용하여 이를 4개의 NumPy 배열 형식 파일로 변환하고, 마지막으로 배열을 훈련 세트와 테스트 세트로 분할합니다.

**1.** Define a variable to store the training/test image/label names of the MNIST dataset in a list:  
**1.** MNIST 데이터셋의 훈련/테스트 이미지와 레이블 이름을 리스트에 저장할 변수를 정의합니다:

```{code-cell}
data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}
```

**2.** Load the data. First check if the data is stored locally; if not, then download it.  
**2.** 데이터를 로드합니다. 우선 로컬에 데이터가 저장되어 있는지 확인한 후, 없으면 다운로드합니다.

```{code-cell}
:tags: [remove-cell]

# Use responsibly! When running notebooks locally, be sure to keep local
# copies of the datasets to prevent unnecessary server requests
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0"
}
request_opts = {
    "headers": headers,
    "params": {"raw": "true"},
}
```

```{code-cell}
import requests
import os

data_dir = "../_data"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://github.com/rossbar/numpy-tutorial-data-mirror/blob/main/"

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True, **request_opts)
        resp.raise_for_status()  # Ensure download was succesful
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)
```

**3.** Decompress the 4 files and create 4 [`ndarrays`](https://numpy.org/doc/stable/reference/arrays.ndarray.html), saving them into a dictionary. Each original image is of size 28x28 and neural networks normally expect a 1D vector input; therefore, you also need to reshape the images by multiplying 28 by 28 (784).  
**3.** 4개의 파일 압축을 해제하여 4개의 [`ndarrays`](https://numpy.org/doc/stable/reference/arrays.ndarray.html)를 생성한 후, 딕셔너리에 저장합니다. 각 원본 이미지는 28x28 크기이며 신경망은 일반적으로 1차원 벡터 입력을 기대하므로, 이미지를 28×28 (즉, 784)로 재구성해야 합니다.

```{code-cell}
import gzip
import numpy as np

mnist_dataset = {}

# Images
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)
# Labels
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)
```

**4.** Split the data into training and test sets using the standard notation of `x` for data and `y` for labels, calling the training and test set images `x_train` and `x_test`, and the labels `y_train` and `y_test`:  
**4.** 데이터를 `x`(데이터)와 `y`(레이블)의 표준 표기법을 사용하여 훈련 세트와 테스트 세트로 분할하고, 훈련 이미지는 `x_train`, 테스트 이미지는 `x_test`, 레이블은 각각 `y_train`과 `y_test`로 지정합니다.

```{code-cell}
x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)
```

**5.** You can confirm that the shape of the image arrays is `(60000, 784)` and `(10000, 784)` for training and test sets, respectively, and the labels — `(60000,)` and `(10000,)`:  
**5.** 훈련 세트의 이미지 배열 형태가 `(60000, 784)`, 테스트 세트의 이미지 배열 형태가 `(10000, 784)`, 레이블은 각각 `(60000,)`와 `(10000,)`임을 확인할 수 있습니다.

```{code-cell}
print(
    "The shape of training images: {} and training labels: {}".format(
        x_train.shape, y_train.shape
    )
)
print(
    "The shape of test images: {} and test labels: {}".format(
        x_test.shape, y_test.shape
    )
)
```

**6.** And you can inspect some images using Matplotlib:  
**6.** 또한 Matplotlib을 사용하여 일부 이미지를 확인할 수 있습니다.

```{code-cell}
import matplotlib.pyplot as plt

# Take the 60,000th image (indexed at 59,999) from the training set,
# reshape from (784, ) to (28, 28) to have a valid shape for displaying purposes.
mnist_image = x_train[59999, :].reshape(28, 28)
# Set the color mapping to grayscale to have a black background.
plt.imshow(mnist_image, cmap="gray")
# Display the image.
plt.show()
```

```{code-cell}
# Display 5 random images from the training set.
num_examples = 5
seed = 147197952744
rng = np.random.default_rng(seed)

fig, axes = plt.subplots(1, num_examples)
for sample, ax in zip(rng.choice(x_train, size=num_examples, replace=False), axes):
    ax.imshow(sample.reshape(28, 28), cmap="gray")
```

_Above are five images taken from the MNIST training set. Various hand-drawn
Arabic numerals are shown, with exact values chosen randomly with each run of the code._

> **Note:** You can also visualize a sample image as an array by printing `x_train[59999]`. Here, `59999` is your 60,000th training image sample (`0` would be your first). Your output will be quite long and should contain an array of 8-bit integers:
>
>
> ```
> ...
>          0,   0,  38,  48,  48,  22,   0,   0,   0,   0,   0,   0,   0,
>          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
>          0,  62,  97, 198, 243, 254, 254, 212,  27,   0,   0,   0,   0,
> ...
> ```

```{code-cell}
# Display the label of the 60,000th image (indexed at 59,999) from the training set.
y_train[59999]
```

## 2. Preprocess the data
## 2. 데이터 전처리

Neural networks can work with inputs that are in a form of tensors (multidimensional arrays) of floating-point type. When preprocessing the data, you should consider the following processes: [vectorization](https://en.wikipedia.org/wiki/Vectorization_%28mathematics%29) and [conversion to a floating-point format](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Floating-point_numbers).

신경망은 부동 소수점 형식의 텐서(다차원 배열) 형태의 입력을 처리할 수 있습니다. 데이터를 전처리할 때는 [벡터화](https://en.wikipedia.org/wiki/Vectorization_%28mathematics%29) 및 [부동 소수점 형식으로 변환](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Floating-point_numbers)하는 과정을 고려해야 합니다.

Since the MNIST data is already vectorized and the arrays are of `dtype` `uint8`, your next challenge is to convert them to a floating-point format, such as `float64` ([double-precision](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)):

MNIST 데이터는 이미 벡터화되어 있으며 배열의 `dtype`이 `uint8`이므로, 다음 과제는 이를 `float64`([배정밀도](https://en.wikipedia.org/wiki/Double-precision_floating-point_format))와 같은 부동 소수점 형식으로 변환하는 것입니다:

- _Normalizing_ the image data: a [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling#Application) procedure that can speed up the neural network training process by standardizing the [distribution of your input data](https://arxiv.org/pdf/1502.03167.pdf).
- 이미지 데이터 _정규화_: 입력 데이터의 [분포를 표준화](https://arxiv.org/pdf/1502.03167.pdf)하여 신경망 훈련 과정을 가속화할 수 있는 [특징 스케일링](https://en.wikipedia.org/wiki/Feature_scaling#Application) 절차입니다.
- _[One-hot/categorical encoding](https://en.wikipedia.org/wiki/One-hot)_ of the image labels.
- 이미지 레이블의 _[원-핫/범주형 인코딩](https://en.wikipedia.org/wiki/One-hot)_.

In practice, you can use different types of floating-point precision depending on your goals and you can find more information about that in the [Nvidia](https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/) and [Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) blog posts.

실제로는 목표에 따라 다양한 유형의 부동 소수점 정밀도를 사용할 수 있으며, 이에 대한 자세한 내용은 [Nvidia](https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/) 및 [Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) 블로그 게시물에서 확인할 수 있습니다.

### Convert the image data to the floating-point format
### 이미지 데이터를 부동 소수점 형식으로 변환

The images data contain 8-bit integers encoded in the [0, 255] interval with color values between 0 and 255.

이미지 데이터는 0에서 255 사이의 색상 값으로 [0, 255] 범위에 인코딩된 8비트 정수를 포함합니다.

You will normalize them into floating-point arrays in the [0, 1] interval by dividing them by 255.

이를 255로 나누어 [0, 1] 범위의 부동 소수점 배열로 정규화합니다.

**1.** Check that the vectorized image data has type `uint8`:  
**1.** 벡터화된 이미지 데이터의 유형이 `uint8`인지 확인합니다:

```{code-cell}
print("The data type of training images: {}".format(x_train.dtype))
print("The data type of test images: {}".format(x_test.dtype))
```

**2.** Normalize the arrays by dividing them by 255 (and thus promoting the data type from `uint8` to `float64`) and then assign the train and test image data variables — `x_train` and `x_test` — to `training_images` and `train_labels`, respectively.
To reduce the model training and evaluation time in this example, only a subset
of the training and test images will be used.
Both `training_images` and `test_images` will contain only 1,000 samples each out
of the complete datasets of 60,000 and 10,000 images, respectively.
These values can be controlled by changing the  `training_sample` and
`test_sample` below, up to their maximum values of 60,000 and 10,000.

**2.** 배열을 255로 나누어 정규화하고(따라서 데이터 유형을 `uint8`에서 `float64`로 승격) 훈련 및 테스트 이미지 데이터 변수인 `x_train`과 `x_test`를 각각 `training_images`와 `train_labels`에 할당합니다.
이 예제에서는 모델 훈련 및 평가 시간을 줄이기 위해 훈련 및 테스트 이미지의 일부만 사용됩니다.
`training_images`와 `test_images`는 각각 전체 데이터셋의 60,000개와 10,000개 이미지 중 각각 1,000개의 샘플만 포함합니다.
이 값은 아래의 `training_sample` 및 `test_sample`을 최대 값인 60,000 및 10,000으로 변경하여 제어할 수 있습니다.

```{code-cell}
training_sample, test_sample = 1000, 1000
training_images = x_train[0:training_sample] / 255
test_images = x_test[0:test_sample] / 255
```

**3.** Confirm that the image data has changed to the floating-point format:  
**3.** 이미지 데이터가 부동 소수점 형식으로 변경되었는지 확인합니다:

```{code-cell}
print("The data type of training images: {}".format(training_images.dtype))
print("The data type of test images: {}".format(test_images.dtype))
```

> **Note:** You can also check that normalization was successful by printing `training_images[0]` in a notebook cell. Your long output should contain an array of floating-point numbers:
>
> **참고:** 노트북 셀에서 `training_images[0]`을 출력하여 정규화가 성공했는지 확인할 수도 있습니다. 긴 출력에는 부동 소수점 숫자의 배열이 포함되어야 합니다:
>
> ```
> ...
>        0.        , 0.        , 0.01176471, 0.07058824, 0.07058824,
>        0.07058824, 0.49411765, 0.53333333, 0.68627451, 0.10196078,
>        0.65098039, 1.        , 0.96862745, 0.49803922, 0.        ,
> ...
> ```

### Convert the labels to floating point through categorical/one-hot encoding
### 범주형/원-핫 인코딩을 통해 레이블을 부동 소수점으로 변환

You will use one-hot encoding to embed each digit label as an all-zero vector with `np.zeros()` and place `1` for a label index. As a result, your label data will be arrays with `1.0` (or `1.`) in the position of each image label.

`np.zeros()`를 사용하여 각 숫자 레이블을 모두 0인 벡터로 임베딩하고 레이블 인덱스에 `1`을 배치합니다. 결과적으로 레이블 데이터는 각 이미지 레이블 위치에 `1.0`(또는 `1.`)이 있는 배열이 됩니다.

Since there are 10 labels (from 0 to 9) in total, your arrays will look similar to this:

총 10개의 레이블(0에서 9까지)이 있으므로 배열은 다음과 유사하게 보일 것입니다:

```
array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
```

**1.** Confirm that the image label data are integers with `dtype` `uint8`:  
**1.** 이미지 레이블 데이터가 `dtype`이 `uint8`인 정수인지 확인합니다:

```{code-cell}
print("The data type of training labels: {}".format(y_train.dtype))
print("The data type of test labels: {}".format(y_test.dtype))
```

**2.** Define a function that performs one-hot encoding on arrays:  
**2.** 배열에 대해 원-핫 인코딩을 수행하는 함수를 정의합니다:

```{code-cell}
def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zero vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float64)
```

**3.** Encode the labels and assign the values to new variables:  
**3.** 레이블을 인코딩하고 값을 새 변수에 할당합니다:

```{code-cell}
training_labels = one_hot_encoding(y_train[:training_sample])
test_labels = one_hot_encoding(y_test[:test_sample])
```

**4.** Check that the data type has changed to floating point:  
**4.** 데이터 유형이 부동 소수점으로 변경되었는지 확인합니다:

```{code-cell}
print("The data type of training labels: {}".format(training_labels.dtype))
print("The data type of test labels: {}".format(test_labels.dtype))
```

**5.** Examine a few encoded labels:  
**5.** 몇 가지 인코딩된 레이블을 살펴봅니다:

```{code-cell}
print(training_labels[0])
print(training_labels[1])
print(training_labels[2])
```

...and compare to the originals:  
...그리고 원본과 비교합니다:

```{code-cell}
print(y_train[0])
print(y_train[1])
print(y_train[2])
```

You have finished preparing the dataset.
데이터셋 준비를 마쳤습니다.

## 3. Build and train a small neural network from scratch
## 3. 작은 신경망을 처음부터 구축하고 훈련하기

In this section you will familiarize yourself with some high-level concepts of the basic building blocks of a deep learning model. You can refer to the original [Deep learning](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) research publication for more information.

이 섹션에서는 딥러닝 모델의 기본 구성 요소에 대한 몇 가지 고급 개념을 익히게 됩니다. 자세한 내용은 원본 [딥러닝](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) 연구 논문을 참조할 수 있습니다.

Afterwards, you will construct the building blocks of a simple deep learning model in Python and NumPy and train it to learn to identify handwritten digits from the MNIST dataset with a certain level of accuracy.

그 후, Python과 NumPy에서 간단한 딥러닝 모델의 구성 요소를 구축하고 이를 훈련하여 MNIST 데이터셋에서 손으로 쓴 숫자를 일정 수준의 정확도로 식별하는 방법을 학습합니다.

### Neural network building blocks with NumPy
### NumPy를 사용한 신경망 구성 요소

- _Layers_: These building blocks work as data filters — they process data and learn representations from inputs to better predict the target outputs.

    - _레이어_: 이러한 구성 요소는 데이터 필터로 작동하여 데이터를 처리하고 입력에서 표현을 학습하여 대상 출력을 더 잘 예측합니다.

    You will use 1 hidden layer in your model to pass the inputs forward (_forward propagation_) and propagate the gradients/error derivatives of a loss function backward (_backpropagation_). These are input, hidden and output layers.

    모델에서 1개의 은닉 레이어를 사용하여 입력을 앞으로 전달(_순방향 전파_)하고 손실 함수의 그래디언트/오차 도함수를 뒤로 전파(_역전파_)합니다. 이는 입력, 은닉 및 출력 레이어입니다.

    In the hidden (middle) and output (last) layers, the neural network model will compute the weighted sum of inputs. To compute this process, you will use NumPy's matrix multiplication function (the "dot multiply" or `np.dot(layer, weights)`).

    은닉(중간) 및 출력(마지막) 레이어에서 신경망 모델은 입력의 가중 합을 계산합니다. 이 과정을 계산하기 위해 NumPy의 행렬 곱셈 함수("점 곱셈" 또는 `np.dot(layer, weights)`)를 사용합니다.

    > **Note:** For simplicity, the bias term is omitted in this example (there is no `np.dot(layer, weights) + bias`).

    > **참고:** 간단히 하기 위해 이 예제에서는 바이어스 항이 생략되었습니다(`np.dot(layer, weights) + bias`가 없습니다).

- _Weights_: These are important adjustable parameters that the neural network fine-tunes by forward and backward propagating the data. They are optimized through a process called [gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). Before the model training starts, the weights are randomly initialized with NumPy's [`Generator.random()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html).

    - _가중치_: 이는 신경망이 데이터를 순방향 및 역방향으로 전파하여 미세 조정하는 중요한 조정 가능한 매개변수입니다. [경사 하강법](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)이라는 과정을 통해 최적화됩니다. 모델 훈련이 시작되기 전에 가중치는 NumPy의 [`Generator.random()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html)으로 무작위 초기화됩니다.

    The optimal weights should produce the highest prediction accuracy and the lowest error on the training and test sets.

    최적의 가중치는 훈련 및 테스트 세트에서 가장 높은 예측 정확도와 가장 낮은 오류를 생성해야 합니다.

- _Activation function_: Deep learning models are capable of determining non-linear relationships between inputs and outputs and these [non-linear functions](https://en.wikipedia.org/wiki/Activation_function) are usually applied to the output of each layer.

    - _활성화 함수_: 딥러닝 모델은 입력과 출력 간의 비선형 관계를 결정할 수 있으며 이러한 [비선형 함수](https://en.wikipedia.org/wiki/Activation_function)는 일반적으로 각 레이어의 출력에 적용됩니다.

    You will use a [rectified linear unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) to the hidden layer's output (for example, `relu(np.dot(layer, weights))`.

    은닉 레이어의 출력에 [정류 선형 유닛(ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))을 사용합니다(예: `relu(np.dot(layer, weights))`).

- _Regularization_: This [technique](https://en.wikipedia.org/wiki/Regularization_(mathematics)) helps prevent the neural network model from [overfitting](https://en.wikipedia.org/wiki/Overfitting).

    - _정규화_: 이 [기법](https://en.wikipedia.org/wiki/Regularization_(mathematics))은 신경망 모델이 [과적합](https://en.wikipedia.org/wiki/Overfitting)되는 것을 방지하는 데 도움이 됩니다.

    In this example, you will use a method called dropout — [dilution](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) — that randomly sets a number of features in a layer to 0s. You will define it with NumPy's [`Generator.integers()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html) method and apply it to the hidden layer of the network.

    이 예제에서는 드롭아웃([희석](https://en.wikipedia.org/wiki/Dilution_(neural_networks)))이라는 방법을 사용하여 레이어의 여러 기능을 무작위로 0으로 설정합니다. 이를 NumPy의 [`Generator.integers()`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html) 메서드로 정의하고 네트워크의 은닉 레이어에 적용합니다.

- _Loss function_: The computation determines the quality of predictions by comparing the image labels (the truth) with the predicted values in the final layer's output.

    - _손실 함수_: 계산은 이미지 레이블(진실)과 최종 레이어의 출력에서 예측된 값을 비교하여 예측의 품질을 결정합니다.

    For simplicity, you will use a basic total squared error using NumPy's `np.sum()` function (for example, `np.sum((final_layer_output - image_labels) ** 2)`).

    간단히 하기 위해 NumPy의 `np.sum()` 함수를 사용하여 기본 총 제곱 오차를 사용합니다(예: `np.sum((final_layer_output - image_labels) ** 2)`).

- _Accuracy_: This metric measures the accuracy of the network's ability to predict on the data it hasn't seen.

    - _정확도_: 이 메트릭은 네트워크가 보지 못한 데이터에 대한 예측 능력의 정확도를 측정합니다.

### Model architecture and training summary
### 모델 아키텍처 및 훈련 요약

Here is a summary of the neural network model architecture and the training process:

다음은 신경망 모델 아키텍처 및 훈련 과정의 요약입니다:

![Diagram showing operations detailed in this tutorial (The input image
is passed into a Hidden layer that creates a weighted sum of outputs.
The weighted sum is passed to the Non-linearity, then regularization and
into the output layer. The output layer creates a prediction which can
then be compared to existing data. The errors are used to calculate the
loss function and update weights in the hidden layer and output
layer.)](_static/tutorial-deep-learning-on-mnist.png)

- _The input layer_:

    - _입력 레이어_:

    It is the input for the network — the previously preprocessed data that is loaded from `training_images` into `layer_0`.

    이는 네트워크의 입력으로, `training_images`에서 `layer_0`으로 로드된 이전에 전처리된 데이터입니다.

- _The hidden (middle) layer_:

    - _은닉(중간) 레이어_:

    `layer_1` takes the output from the previous layer and performs matrix-multiplication of the input by weights (`weights_1`) with NumPy's `np.dot()`).

    `layer_1`은 이전 레이어의 출력을 받아 입력을 가중치(`weights_1`)로 행렬 곱셈을 수행합니다(NumPy의 `np.dot()` 사용).

    Then, this output is passed through the ReLU activation function for non-linearity and then dropout is applied to help with overfitting.

    그런 다음 이 출력은 비선형성을 위해 ReLU 활성화 함수를 통과한 후 과적합을 방지하기 위해 드롭아웃이 적용됩니다.

- _The output (last) layer_:

    - _출력(마지막) 레이어_:

    `layer_2` ingests the output from `layer_1` and repeats the same "dot multiply" process with `weights_2`.

    `layer_2`는 `layer_1`의 출력을 받아 `weights_2`와 동일한 "점 곱셈" 과정을 반복합니다.

    The final output returns 10 scores for each of the 0-9 digit labels. The network model ends with a size 10 layer — a 10-dimensional vector.

    최종 출력은 0-9 숫자 레이블 각각에 대해 10개의 점수를 반환합니다. 네트워크 모델은 크기 10 레이어(10차원 벡터)로 끝납니다.

- _Forward propagation, backpropagation, training loop_:

    - _순방향 전파, 역전파, 훈련 루프_:

    In the beginning of model training, your network randomly initializes the weights and feeds the input data forward through the hidden and output layers. This process is the forward pass or forward propagation.

    모델 훈련 초기에 네트워크는 가중치를 무작위로 초기화하고 입력 데이터를 은닉 및 출력 레이어를 통해 순방향으로 전달합니다. 이 과정은 순방향 패스 또는 순방향 전파입니다.

    Then, the network propagates the "signal" from the loss function back through the hidden layer and adjusts the weights values with the help of the learning rate parameter (more on that later).

    그런 다음 네트워크는 손실 함수에서 "신호"를 은닉 레이어를 통해 다시 전파하고 학습률 매개변수의 도움으로 가중치 값을 조정합니다(자세한 내용은 나중에 설명).

> **Note:** In more technical terms, you:
>
>    > **참고:** 더 기술적인 용어로 설명하면 다음과 같습니다:
>
>    1. Measure the error by comparing the real label of an image (the truth) with the prediction of the model.
>    1. 이미지의 실제 레이블(진실)과 모델의 예측을 비교하여 오류를 측정합니다.
>    2. Differentiate the loss function.
>    2. 손실 함수를 미분합니다.
>    3. Ingest the [gradients](https://en.wikipedia.org/wiki/Gradient) with the respect to the output, and backpropagate them with the respect to the inputs through the layer(s).
>    3. 출력에 대한 [그래디언트](https://en.wikipedia.org/wiki/Gradient)를 받아들이고, 입력에 대한 그래디언트를 레이어를 통해 역전파합니다.
>
>    Since the network contains tensor operations and weight matrices, backpropagation uses the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).
>
>    네트워크에는 텐서 연산과 가중치 행렬이 포함되어 있으므로 역전파는 [연쇄 법칙](https://en.wikipedia.org/wiki/Chain_rule)을 사용합니다.
>
>    With each iteration (epoch) of the neural network training, this forward and backward propagation cycle adjusts the weights, which is reflected in the accuracy and error metrics. As you train the model, your goal is to minimize the error and maximize the accuracy on the training data, where the model learns from, as well as the test data, where you evaluate the model.
>
>    신경망 훈련의 각 반복(에포크)마다 이 순방향 및 역방향 전파 주기는 가중치를 조정하며, 이는 정확도 및 오류 메트릭에 반영됩니다. 모델을 훈련할 때의 목표는 모델이 학습하는 훈련 데이터와 모델을 평가하는 테스트 데이터에서 오류를 최소화하고 정확도를 최대화하는 것입니다.

### Compose the model and begin training and testing it
### 모델을 구성하고 훈련 및 테스트를 시작합니다

Having covered the main deep learning concepts and the neural network architecture, let's write the code.

주요 딥러닝 개념과 신경망 아키텍처를 다루었으므로 이제 코드를 작성해 보겠습니다.

**1.** We'll start by creating a new random number generator, providing a seed for reproducibility:  
**1.** 재현성을 위해 시드를 제공하여 새로운 난수 생성기를 생성하는 것으로 시작합니다:

```{code-cell}
seed = 884736743
rng = np.random.default_rng(seed)
```

**2.** For the hidden layer, define the ReLU activation function for forward propagation and ReLU's derivative that will be used during backpropagation:  
**2.** 은닉 레이어의 경우 순방향 전파를 위한 ReLU 활성화 함수를 정의하고 역전파 시 사용할 ReLU의 도함수를 정의합니다:

```{code-cell}
# Define ReLU that returns the input if it's positive and 0 otherwise.
def relu(x):
    return (x >= 0) * x


# Set up a derivative of the ReLU function that returns 1 for a positive input
# and 0 otherwise.
def relu2deriv(output):
    return output >= 0
```

**3.** Set certain default values of [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)), such as:  
**3.** 다음과 같은 [하이퍼파라미터](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))의 기본값을 설정합니다:

- [_Learning rate_](https://en.wikipedia.org/wiki/Learning_rate): `learning_rate` — helps limit the magnitude of weight updates to prevent them from overcorrecting.
- [_학습률_](https://en.wikipedia.org/wiki/Learning_rate): `learning_rate` — 가중치 업데이트의 크기를 제한하여 과도한 수정이 발생하지 않도록 합니다.
- _Epochs (iterations)_: `epochs` — the number of complete passes — forward and backward propagations — of the data through the network. This parameter can positively or negatively affect the results. The higher the iterations, the longer the learning process may take. Because this is a computationally intensive task, we have chosen a very low number of epochs (20). To get meaningful results, you should choose a much larger number.
- _에포크(반복 횟수)_: `epochs` — 데이터가 네트워크를 통해 순방향 및 역방향 전파되는 전체 패스 수입니다. 이 매개변수는 결과에 긍정적 또는 부정적 영향을 미칠 수 있습니다. 반복 횟수가 많을수록 학습 과정이 더 오래 걸릴 수 있습니다. 이는 계산 집약적인 작업이기 때문에 매우 적은 수의 에포크(20)를 선택했습니다. 의미 있는 결과를 얻으려면 훨씬 더 많은 수를 선택해야 합니다.
- _Size of the hidden (middle) layer in a network_: `hidden_size` — different sizes of the hidden layer can affect the results during training and testing.
- _네트워크의 은닉(중간) 레이어 크기_: `hidden_size` — 은닉 레이어의 크기가 다르면 훈련 및 테스트 중 결과에 영향을 미칠 수 있습니다.
- _Size of the input:_ `pixels_per_image` — you have established that the image input is 784 (28x28) (in pixels).
- _입력 크기:_ `pixels_per_image` — 이미지 입력이 784(28x28)임을 확인했습니다(픽셀 단위).
- _Number of labels_: `num_labels` — indicates the output number for the output layer where the predictions occur for 10 (0 to 9) handwritten digit labels.
- _레이블 수_: `num_labels` — 10개의 손으로 쓴 숫자 레이블(0에서 9)에 대한 예측이 발생하는 출력 레이어의 출력 수를 나타냅니다.

```{code-cell}
learning_rate = 0.005
epochs = 20
hidden_size = 100
pixels_per_image = 784
num_labels = 10
```

**4.** Initialize the weight vectors that will be used in the hidden and output layers with random values:  
**4.** 은닉 및 출력 레이어에서 사용할 가중치 벡터를 무작위 값으로 초기화합니다:

```{code-cell}
weights_1 = 0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1
weights_2 = 0.2 * rng.random((hidden_size, num_labels)) - 0.1
```

**5.** Set up the neural network's learning experiment with a training loop and start the training process.
Note that the model is evaluated against the test set at each epoch to track
its performance over the training epochs.

**5.** 훈련 루프를 사용하여 신경망의 학습 실험을 설정하고 훈련 과정을 시작합니다.
모델은 각 에포크에서 테스트 세트를 기준으로 평가되어 훈련 에포크 동안의 성능을 추적합니다.

Start the training process:
훈련 과정을 시작합니다:

```{code-cell}
# To store training and test set losses and accurate predictions
# for visualization.
store_training_loss = []
store_training_accurate_pred = []
store_test_loss = []
store_test_accurate_pred = []

# This is a training loop.
# Run the learning experiment for a defined number of epochs (iterations).
for j in range(epochs):

    #################
    # Training step #
    #################

    # Set the initial loss/error and the number of accurate predictions to zero.
    training_loss = 0.0
    training_accurate_predictions = 0

    # For all images in the training set, perform a forward pass
    # and backpropagation and adjust the weights accordingly.
    for i in range(len(training_images)):
        # Forward propagation/forward pass:
        # 1. The input layer:
        #    Initialize the training image data as inputs.
        layer_0 = training_images[i]
        # 2. The hidden layer:
        #    Take in the training image data into the middle layer by
        #    matrix-multiplying it by randomly initialized weights.
        layer_1 = np.dot(layer_0, weights_1)
        # 3. Pass the hidden layer's output through the ReLU activation function.
        layer_1 = relu(layer_1)
        # 4. Define the dropout function for regularization.
        dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
        # 5. Apply dropout to the hidden layer's output.
        layer_1 *= dropout_mask * 2
        # 6. The output layer:
        #    Ingest the output of the middle layer into the the final layer
        #    by matrix-multiplying it by randomly initialized weights.
        #    Produce a 10-dimension vector with 10 scores.
        layer_2 = np.dot(layer_1, weights_2)

        # Backpropagation/backward pass:
        # 1. Measure the training error (loss function) between the actual
        #    image labels (the truth) and the prediction by the model.
        training_loss += np.sum((training_labels[i] - layer_2) ** 2)
        # 2. Increment the accurate prediction count.
        training_accurate_predictions += int(
            np.argmax(layer_2) == np.argmax(training_labels[i])
        )
        # 3. Differentiate the loss function/error.
        layer_2_delta = training_labels[i] - layer_2
        # 4. Propagate the gradients of the loss function back through the hidden layer.
        layer_1_delta = np.dot(weights_2, layer_2_delta) * relu2deriv(layer_1)
        # 5. Apply the dropout to the gradients.
        layer_1_delta *= dropout_mask
        # 6. Update the weights for the middle and input layers
        #    by multiplying them by the learning rate and the gradients.
        weights_1 += learning_rate * np.outer(layer_0, layer_1_delta)
        weights_2 += learning_rate * np.outer(layer_1, layer_2_delta)

    # Store training set losses and accurate predictions.
    store_training_loss.append(training_loss)
    store_training_accurate_pred.append(training_accurate_predictions)

    ###################
    # Evaluation step #
    ###################

    # Evaluate model performance on the test set at each epoch.

    # Unlike the training step, the weights are not modified for each image
    # (or batch). Therefore the model can be applied to the test images in a
    # vectorized manner, eliminating the need to loop over each image
    # individually:

    results = relu(test_images @ weights_1) @ weights_2

    # Measure the error between the actual label (truth) and prediction values.
    test_loss = np.sum((test_labels - results) ** 2)

    # Measure prediction accuracy on test set
    test_accurate_predictions = np.sum(
        np.argmax(results, axis=1) == np.argmax(test_labels, axis=1)
    )

    # Store test set losses and accurate predictions.
    store_test_loss.append(test_loss)
    store_test_accurate_pred.append(test_accurate_predictions)

    # Summarize error and accuracy metrics at each epoch
    print(
        (
            f"Epoch: {j}\n"
            f"  Training set error: {training_loss / len(training_images):.3f}\n"
            f"  Training set accuracy: {training_accurate_predictions / len(training_images)}\n"
            f"  Test set error: {test_loss / len(test_images):.3f}\n"
            f"  Test set accuracy: {test_accurate_predictions / len(test_images)}"
        )
    )
```

The training process may take many minutes, depending on a number of factors, such as the processing power of the machine you are running the experiment on and the number of epochs. To reduce the waiting time, you can change the epoch (iteration) variable from 100 to a lower number, reset the runtime (which will reset the weights), and run the notebook cells again.

훈련 과정은 실험을 실행하는 머신의 처리 능력 및 에포크 수와 같은 여러 요인에 따라 몇 분이 걸릴 수 있습니다. 대기 시간을 줄이기 위해 에포크(반복) 변수를 100에서 더 낮은 숫자로 변경하고 런타임을 재설정(가중치가 재설정됨)한 후 노트북 셀을 다시 실행할 수 있습니다.

+++

After executing the cell above, you can visualize the training and test set errors and accuracy for an instance of this training process.

위의 셀을 실행한 후 이 훈련 과정의 인스턴스에 대한 훈련 및 테스트 세트 오류와 정확도를 시각화할 수 있습니다.

```{code-cell}
epoch_range = np.arange(epochs) + 1  # Starting from 1

# The training set metrics.
training_metrics = {
    "accuracy": np.asarray(store_training_accurate_pred) / len(training_images),
    "error": np.asarray(store_training_loss) / len(training_images),
}

# The test set metrics.
test_metrics = {
    "accuracy": np.asarray(store_test_accurate_pred) / len(test_images),
    "error": np.asarray(store_test_loss) / len(test_images),
}

# Display the plots.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
for ax, metrics, title in zip(
    axes, (training_metrics, test_metrics), ("Training set", "Test set")
):
    # Plot the metrics
    for metric, values in metrics.items():
        ax.plot(epoch_range, values, label=metric.capitalize())
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.legend()
plt.show()
```

_The training and testing error is shown above in the left and right
plots, respectively. As the number of Epochs increases, the total error
decreases and the accuracy increases._

_훈련 및 테스트 오류는 각각 왼쪽 및 오른쪽 플롯에 표시됩니다. 에포크 수가 증가함에 따라 총 오류는 감소하고 정확도는 증가합니다._

The accuracy rates that your model reaches during training and testing may be somewhat plausible but you may also find the error rates to be quite high.
훈련 및 테스트 중 모델이 도달하는 정확도는 다소 그럴듯할 수 있지만 오류율이 상당히 높을 수도 있습니다.
