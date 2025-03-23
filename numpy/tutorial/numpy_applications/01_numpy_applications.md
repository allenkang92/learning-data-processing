# Determining Moore's Law with real data in NumPy
# NumPy에서 실제 데이터로 무어의 법칙 확인하기
![Scatter plot of MOS transistor count per microprocessor every two years as a demonstration of Moore's Law.](_static/01-mooreslaw-tutorial-intro.png)

_The number of transistors reported per a given chip plotted on a log scale in the y axis with the date of introduction on the linear scale x-axis. The blue data points are from a [transistor count table](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors). The red line is an ordinary least squares prediction and the orange line is Moore's law._

_특정 칩당 보고된 트랜지스터 수를 y축의 로그 스케일에 표시하고 도입 날짜를 x축의 선형 스케일에 표시한 그래프입니다. 파란색 데이터 포인트는 [트랜지스터 수 표](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors)에서 가져온 것입니다. 빨간색 선은 일반 최소 제곱 예측이고 주황색 선은 무어의 법칙입니다._

## What you'll do
## 무엇을 할 것인가

In 1965, engineer Gordon Moore
[predicted](https://en.wikipedia.org/wiki/Moore%27s_law) that
transistors on a chip would double every two years in the coming decade
[[1](https://en.wikipedia.org/wiki/Moore%27s_law)].
You'll compare Moore's prediction against actual transistor counts in
the 53 years following his prediction. You will determine the best-fit constants to describe the exponential growth of transistors on semiconductors compared to Moore's Law.

1965년, 엔지니어 고든 무어는 다가오는 10년 동안 칩의 트랜지스터가 2년마다 두 배가 될 것이라고 [예측](https://en.wikipedia.org/wiki/Moore%27s_law)했습니다[[1](https://en.wikipedia.org/wiki/Moore%27s_law)]. 무어의 예측을 그의 예측 이후 53년 동안의 실제 트랜지스터 수와 비교할 것입니다. 무어의 법칙과 비교하여 반도체의 트랜지스터 지수 성장을 설명하는 최적의 상수를 결정할 것입니다.


## Skills you'll learn
## 배우게 될 기술


- Load data from a [\*.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file
- Perform linear regression and predict exponential growth using ordinary least squares
- You'll compare exponential growth constants between models
- Share your analysis in a file:
    - as NumPy zipped files `*.npz`
    - as a `*.csv` file
- Assess the amazing progress semiconductor manufacturers have made in the last five decades

- [\*.csv](https://en.wikipedia.org/wiki/Comma-separated_values) 파일에서 데이터 로드하기
- 일반 최소 제곱법을 사용하여 선형 회귀를 수행하고 지수 성장을 예측하기
- 모델 간의 지수 성장 상수 비교하기
- 분석 결과를 파일로 공유하기:
    - NumPy 압축 파일 `*.npz`로
    - `*.csv` 파일로
- 지난 50년 동안 반도체 제조업체가 이룬 놀라운 발전 평가하기

## What you'll need
## 필요한 것

**1.** These packages:
**1.** 다음 패키지들:

* NumPy
* [Matplotlib](https://matplotlib.org/)

imported with the following commands
다음 명령으로 가져옵니다

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
```

**2.** Since this is an exponential growth law you need a little background in doing math with [natural logs](https://en.wikipedia.org/wiki/Natural_logarithm) and [exponentials](https://en.wikipedia.org/wiki/Exponential_function).

**2.** 이것은 지수 성장 법칙이므로 [자연 로그](https://en.wikipedia.org/wiki/Natural_logarithm)와 [지수 함수](https://en.wikipedia.org/wiki/Exponential_function)로 수학을 하는 약간의 배경 지식이 필요합니다.

You'll use these NumPy and Matplotlib functions:
다음 NumPy 및 Matplotlib 함수를 사용할 것입니다:

* [`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html): this function loads text into a NumPy array
* [`np.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html): this function takes the natural log of all elements in a NumPy array
* [`np.exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html): this function takes the exponential of all elements in a NumPy array
* [`lambda`](https://docs.python.org/3/library/ast.html?highlight=lambda#ast.Lambda): this is a minimal function definition for creating a function model
* [`plt.semilogy`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.semilogy.html): this function will plot x-y data onto a figure with a linear x-axis and $\log_{10}$ y-axis
[`plt.plot`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html): this function will plot x-y data on linear axes
* slicing arrays: view parts of the data loaded into the workspace, slice the arrays e.g. `x[:10]` for the first 10 values in the array, `x`
* boolean array indexing: to view parts of the data that match a given condition use boolean operations to index an array
* [`np.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html): to combine arrays into 2D arrays
* [`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html): to change a 1D vector to a row or column vector
* [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) and [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html): these two functions will save your arrays in zipped array format and text, respectively

* [`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html): 이 함수는 텍스트를 NumPy 배열로 로드합니다
* [`np.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html): 이 함수는 NumPy 배열의 모든 요소의 자연 로그를 계산합니다
* [`np.exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html): 이 함수는 NumPy 배열의 모든 요소의 지수를 계산합니다
* [`lambda`](https://docs.python.org/3/library/ast.html?highlight=lambda#ast.Lambda): 함수 모델을 만들기 위한 최소한의 함수 정의입니다
* [`plt.semilogy`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.semilogy.html): 이 함수는 선형 x축과 $\log_{10}$ y축을 가진 그림에 x-y 데이터를 그립니다
[`plt.plot`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html): 이 함수는 선형 축에 x-y 데이터를 그립니다
* 배열 슬라이싱: 워크스페이스에 로드된 데이터의 일부를 보려면 배열을 슬라이스합니다. 예: `x[:10]`는 배열 `x`의 처음 10개 값을 보여줍니다
* 불리언 배열 인덱싱: 주어진 조건과 일치하는 데이터 부분을 보려면 불리언 연산을 사용하여 배열을 인덱싱합니다
* [`np.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html): 배열을 2D 배열로 결합합니다
* [`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html): 1D 벡터를 행 또는 열 벡터로 변경합니다
* [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)와 [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html): 이 두 함수는 각각 압축된 배열 형식과 텍스트로 배열을 저장합니다

+++

---

## Building Moore's law as an exponential function
## 무어의 법칙을 지수 함수로 구성하기

Your empirical model assumes that the number of transistors per
semiconductor follows an exponential growth,

$\log(\text{transistor_count})= f(\text{year}) = A\cdot \text{year}+B,$

where $A$ and $B$ are fitting constants. You use semiconductor
manufacturers' data to find the fitting constants.

당신의 경험적 모델은 반도체당 트랜지스터 수가 지수 성장을 따른다고 가정합니다,

$\log(\text{transistor_count})= f(\text{year}) = A\cdot \text{year}+B,$

여기서 $A$와 $B$는 피팅 상수입니다. 반도체 제조업체의 데이터를 사용하여 피팅 상수를 찾습니다.

You determine these constants for Moore's law by specifying the
rate for added transistors, 2, and giving an initial number of transistors for a given year.

트랜지스터 추가 비율인 2를 지정하고 주어진 연도에 대한 초기 트랜지스터 수를 제공하여 무어의 법칙에 대한 이러한 상수를 결정합니다.

You state Moore's law in an exponential form as follows,

$\text{transistor_count}= e^{A_M\cdot \text{year} +B_M}.$

Where $A_M$ and $B_M$ are constants that double the number of transistors every two years and start at 2250 transistors in 1971,

무어의 법칙을 다음과 같이 지수 형태로 표현합니다,

$\text{transistor_count}= e^{A_M\cdot \text{year} +B_M}.$

여기서 $A_M$과 $B_M$은 2년마다 트랜지스터 수를 두 배로 늘리고 1971년에 2250개의 트랜지스터에서 시작하는 상수입니다,

1. $\dfrac{\text{transistor_count}(\text{year} +2)}{\text{transistor_count}(\text{year})} = 2 = \dfrac{e^{B_M}e^{A_M \text{year} + 2A_M}}{e^{B_M}e^{A_M \text{year}}} = e^{2A_M} \rightarrow A_M = \frac{\log(2)}{2}$

2. $\log(2250) = \frac{\log(2)}{2}\cdot 1971 + B_M \rightarrow B_M = \log(2250)-\frac{\log(2)}{2}\cdot 1971$

so Moore's law stated as an exponential function is

$\log(\text{transistor_count})= A_M\cdot \text{year}+B_M,$

where

$A_M=0.3466$

$B_M=-675.4$

따라서 지수 함수로 표현된 무어의 법칙은

$\log(\text{transistor_count})= A_M\cdot \text{year}+B_M,$

여기서

$A_M=0.3466$

$B_M=-675.4$

Since the function represents Moore's law, define it as a Python
function using
[`lambda`](https://docs.python.org/3/library/ast.html?highlight=lambda#ast.Lambda)

함수가 무어의 법칙을 나타내므로, [`lambda`](https://docs.python.org/3/library/ast.html?highlight=lambda#ast.Lambda)를 사용하여 Python 함수로 정의합니다

```{code-cell}
A_M = np.log(2) / 2
B_M = np.log(2250) - A_M * 1971
Moores_law = lambda year: np.exp(B_M) * np.exp(A_M * year)
```

In 1971, there were 2250 transistors on the Intel 4004 chip. Use
`Moores_law` to check the number of semiconductors Gordon Moore would expect
in 1973.

1971년에는 Intel 4004 칩에 2250개의 트랜지스터가 있었습니다. `Moores_law`를 사용하여 1973년에 고든 무어가 예상했을 반도체 수를 확인합니다.

```{code-cell}
ML_1971 = Moores_law(1971)
ML_1973 = Moores_law(1973)
print("In 1973, G. Moore expects {:.0f} transistors on Intels chips".format(ML_1973))
print("This is x{:.2f} more transistors than 1971".format(ML_1973 / ML_1971))
```

## Loading historical manufacturing data to your workspace
## 작업 공간에 역사적 제조 데이터 로드하기

Now, make a prediction based upon the historical data for
semiconductors per chip. The [Transistor Count
\[3\]](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors)
each year is in the `transistor_data.csv` file. Before loading a \*.csv
file into a NumPy array, its a good idea to inspect the structure of the
file first. Then, locate the columns of interest and save them to a
variable. Save two columns of the file to the array, `data`.

이제 칩당 반도체의 역사적 데이터를 기반으로 예측을 합니다. 매년 [트랜지스터 수 \[3\]](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors)는 `transistor_data.csv` 파일에 있습니다. \*.csv 파일을 NumPy 배열로 로드하기 전에 먼저 파일의 구조를 검사하는 것이 좋습니다. 그런 다음 관심 있는 열을 찾아 변수에 저장합니다. 파일의 두 열을 배열 `data`에 저장합니다.

Here, print out the first 10 rows of `transistor_data.csv`. The columns are

|Processor|MOS transistor count|Date of Introduction|Designer|MOSprocess|Area|
|---|---|---|---|---|---|
|Intel 4004 (4-bit  16-pin)|2250|1971|Intel|"10,000 nm"|12 mm²|
|...|...|...|...|...|...|

여기서 `transistor_data.csv`의 처음 10행을 출력합니다. 열은 다음과 같습니다

|프로세서|MOS 트랜지스터 수|도입 날짜|설계자|MOS 공정|면적|
|---|---|---|---|---|---|
|Intel 4004 (4-bit  16-pin)|2250|1971|Intel|"10,000 nm"|12 mm²|
|...|...|...|...|...|...|

```{code-cell}
! head transistor_data.csv
```

You don't need the columns that specify __Processor__, __Designer__,
__MOSprocess__, or __Area__. That leaves the second and third columns,
__MOS transistor count__ and __Date of Introduction__, respectively.

__프로세서__, __설계자__, __MOS 공정__ 또는 __면적__을 지정하는 열은 필요하지 않습니다. 따라서 두 번째와 세 번째 열인 __MOS 트랜지스터 수__와 __도입 날짜__만 남습니다.

Next, you load these two columns into a NumPy array using
[`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html).
The extra options below will put the data in the desired format:

* `delimiter = ','`: specify delimeter as a comma ',' (this is the default behavior)
* `usecols = [1,2]`: import the second and third columns from the csv
* `skiprows = 1`: do not use the first row, because its a header row

다음으로, [`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html)를 사용하여 이 두 열을 NumPy 배열로 로드합니다.
아래의 추가 옵션은 데이터를 원하는 형식으로 만들어 줍니다:

* `delimiter = ','`: 구분자를 쉼표 ','로 지정 (이것이 기본 동작임)
* `usecols = [1,2]`: csv에서 두 번째와 세 번째 열을 가져옴
* `skiprows = 1`: 첫 번째 행은 헤더 행이므로 사용하지 않음

```{code-cell}
data = np.loadtxt("transistor_data.csv", delimiter=",", usecols=[1, 2], skiprows=1)
```

You loaded the entire history of semiconducting into a NumPy array named
`data`. The first column is the __MOS transistor count__ and the second
column is the __Date of Introduction__ in a four-digit year.

`data`라는 NumPy 배열에 반도체의 전체 역사를 로드했습니다. 첫 번째 열은 __MOS 트랜지스터 수__이고 두 번째 열은 네 자리 연도의 __도입 날짜__입니다.

Next, make the data easier to read and manage by assigning the two
columns to variables, `year` and `transistor_count`. Print out the first
10 values by slicing the `year` and `transistor_count` arrays with
`[:10]`. Print these values out to check that you have the saved the
data to the correct variables.

다음으로, 두 열을 `year`와 `transistor_count` 변수에 할당하여 데이터를 읽고 관리하기 쉽게 만듭니다. `[:10]`으로 `year`와 `transistor_count` 배열을 슬라이싱하여 처음 10개 값을 출력합니다. 데이터가 올바른 변수에 저장되었는지 확인하기 위해 이 값들을 출력합니다.

```{code-cell}
year = data[:, 1]  # grab the second column and assign
transistor_count = data[:, 0]  # grab the first column and assign

print("year:\t\t", year[:10])
print("trans. cnt:\t", transistor_count[:10])
```

You are creating a function that predicts the transistor count given a
year. You have an _independent variable_, `year`, and a _dependent
variable_, `transistor_count`. Transform the dependent variable to
log-scale,

$y_i = \log($ `transistor_count[i]` $),$

resulting in a linear equation,

$y_i = A\cdot \text{year} +B$.

연도가 주어졌을 때 트랜지스터 수를 예측하는 함수를 만들고 있습니다. _독립 변수_ `year`와 _종속 변수_ `transistor_count`가 있습니다. 종속 변수를 로그 스케일로 변환합니다,

$y_i = \log($ `transistor_count[i]` $),$

그 결과 선형 방정식이 됩니다,

$y_i = A\cdot \text{year} +B$.

```{code-cell}
yi = np.log(transistor_count)
```

## Calculating the historical growth curve for transistors
## 트랜지스터의 역사적 성장 곡선 계산하기

Your model assume that `yi` is a function of `year`. Now, find the best-fit model that minimizes the difference between $y_i$ and $A\cdot \text{year} +B, $ as such

$\min \sum|y_i - (A\cdot \text{year}_i + B)|^2.$

This [sum of squares
error](https://en.wikipedia.org/wiki/Ordinary_least_squares) can be
succinctly represented as arrays as such

$\sum|\mathbf{y}-\mathbf{Z} [A,~B]^T|^2,$

where $\mathbf{y}$ are the observations of the log of the number of
transistors in a 1D array and $\mathbf{Z}=[\text{year}_i^1,~\text{year}_i^0]$ are the
polynomial terms for $\text{year}_i$ in the first and second columns. By
creating this set of regressors in the $\mathbf{Z}-$matrix you set
up an ordinary least squares statistical model.

당신의 모델은 `yi`가 `year`의 함수라고 가정합니다. 이제 $y_i$와 $A\cdot \text{year} +B$ 사이의 차이를 최소화하는 최적 모델을 찾습니다,

$\min \sum|y_i - (A\cdot \text{year}_i + B)|^2.$

이 [제곱 합 오차](https://en.wikipedia.org/wiki/Ordinary_least_squares)는 다음과 같이 배열로 간결하게 표현할 수 있습니다

$\sum|\mathbf{y}-\mathbf{Z} [A,~B]^T|^2,$

여기서 $\mathbf{y}$는 1D 배열의 트랜지스터 수의 로그 관측값이고 $\mathbf{Z}=[\text{year}_i^1,~\text{year}_i^0]$은 첫 번째와 두 번째 열에 있는 $\text{year}_i$의 다항식 항입니다. $\mathbf{Z}-$행렬에서 이러한 회귀 변수 세트를 만들어 일반 최소 제곱 통계 모델을 설정합니다.

`Z` is a linear model with two parameters, i.e. a polynomial with degree `1`.
Therefore we can represent the model with `numpy.polynomial.Polynomial` and
use the fitting functionality to determine the model parameters:

`Z`는 두 개의 매개변수를 가진 선형 모델, 즉 차수가 `1`인 다항식입니다.
따라서 우리는 `numpy.polynomial.Polynomial`로 모델을 표현하고 피팅 기능을 사용하여 모델 매개변수를 결정할 수 있습니다:

```{code-cell}
model = np.polynomial.Polynomial.fit(year, yi, deg=1)
```

By default, `Polynomial.fit` performs the fit in the domain determined by the
independent variable (`year` in this case).
The coefficients for the unscaled and unshifted model can be recovered with the
`convert` method:

기본적으로 `Polynomial.fit`은 독립 변수(이 경우 `year`)에 의해 결정된 도메인에서 피팅을 수행합니다.
스케일링되지 않고 이동되지 않은 모델의 계수는 `convert` 메서드로 복구할 수 있습니다:


```{code-cell}
model = model.convert()
model
```

The individual parameters $A$ and $B$ are the coefficients of our linear model:

개별 매개변수 $A$와 $B$는 선형 모델의 계수입니다:

```{code-cell}
B, A = model
```

Did manufacturers double the transistor count every two years? You have
the final formula,

$\dfrac{\text{transistor_count}(\text{year} +2)}{\text{transistor_count}(\text{year})} = xFactor =
\dfrac{e^{B}e^{A( \text{year} + 2)}}{e^{B}e^{A \text{year}}} = e^{2A}$

where increase in number of transistors is $xFactor,$ number of years is
2, and $A$ is the best fit slope on the semilog function.

제조업체들이 매 2년마다 트랜지스터 수를 두 배로 늘렸을까요? 최종 공식은 다음과 같습니다,

$\dfrac{\text{transistor_count}(\text{year} +2)}{\text{transistor_count}(\text{year})} = xFactor =
\dfrac{e^{B}e^{A( \text{year} + 2)}}{e^{B}e^{A \text{year}}} = e^{2A}$

여기서 트랜지스터 수의 증가는 $xFactor,$ 연도 수는 2, 그리고 $A$는 세미로그 함수에서 최적 피팅된 기울기입니다.

```{code-cell}
print(f"Rate of semiconductors added on a chip every 2 years: {np.exp(2 * A):.2f}")
```

Based upon your least-squares regression model, the number of
semiconductors per chip increased by a factor of $1.98$ every two
years. You have a model that predicts the number of semiconductors each
year. Now compare your model to the actual manufacturing reports.  Plot
the linear regression results and all of the transistor counts.

최소 제곱 회귀 모델에 따르면, 칩당 반도체 수는 2년마다 $1.98$배 증가했습니다. 이제 매년 반도체 수를 예측하는 모델이 있습니다. 이제 당신의 모델을 실제 제조 보고서와 비교해 보세요. 선형 회귀 결과와 모든 트랜지스터 수를 그래프로 표시합니다.

Here, use
[`plt.semilogy`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.semilogy.html)
to plot the number of transistors on a log-scale and the year on a
linear scale. You have defined a three arrays to get to a final model

$y_i = \log(\text{transistor_count}),$

$y_i = A \cdot \text{year} + B,$

and

$\log(\text{transistor_count}) = A\cdot \text{year} + B,$

your variables, `transistor_count`, `year`, and `yi` all have the same
dimensions, `(179,)`. NumPy arrays need the same dimensions to make a
plot. The predicted number of transistors is now

$\text{transistor_count}_{\text{predicted}} = e^Be^{A\cdot \text{year}}$.

여기서는 [`plt.semilogy`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.semilogy.html)를 사용하여 로그 스케일의 트랜지스터 수와 선형 스케일의 연도를 표시합니다. 최종 모델을 얻기 위해 세 개의 배열을 정의했습니다

$y_i = \log(\text{transistor_count}),$

$y_i = A \cdot \text{year} + B,$

그리고

$\log(\text{transistor_count}) = A\cdot \text{year} + B,$

당신의 변수인 `transistor_count`, `year`, 그리고 `yi`는 모두 같은 차원인 `(179,)`를 가집니다. NumPy 배열은 그래프를 만들기 위해 같은 차원이 필요합니다. 이제 예측된 트랜지스터 수는

$\text{transistor_count}_{\text{predicted}} = e^Be^{A\cdot \text{year}}$.

+++

In the next plot, use the
[`fivethirtyeight`](https://matplotlib.org/3.1.1/gallery/style_sheets/fivethirtyeight.html)
style sheet.
The style sheet replicates
https://fivethirtyeight.com elements. Change the matplotlib style with
[`plt.style.use`](https://matplotlib.org/3.3.2/api/style_api.html#matplotlib.style.use).

다음 그래프에서는 [`fivethirtyeight`](https://matplotlib.org/3.1.1/gallery/style_sheets/fivethirtyeight.html) 스타일 시트를 사용합니다.
이 스타일 시트는 https://fivethirtyeight.com 요소를 복제합니다. [`plt.style.use`](https://matplotlib.org/3.3.2/api/style_api.html#matplotlib.style.use)로 matplotlib 스타일을 변경합니다.

```{code-cell}
transistor_count_predicted = np.exp(B) * np.exp(A * year)
transistor_Moores_law = Moores_law(year)
plt.style.use("fivethirtyeight")
plt.semilogy(year, transistor_count, "s", label="MOS transistor count")
plt.semilogy(year, transistor_count_predicted, label="linear regression")


plt.plot(year, transistor_Moores_law, label="Moore's Law")
plt.title(
    "MOS transistor count per microprocessor\n"
    + "every two years \n"
    + "Transistor count was x{:.2f} higher".format(np.exp(A * 2))
)
plt.xlabel("year introduced")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.ylabel("# of transistors\nper microprocessor")
```

_A scatter plot of MOS transistor count per microprocessor every two years with a red line for the ordinary least squares prediction and an orange line for Moore's law._

_마이크로프로세서당 MOS 트랜지스터 수를 2년마다 표시한 산점도로, 빨간색 선은 일반 최소 제곱 예측이고 주황색 선은 무어의 법칙을 나타냅니다._

The linear regression captures the increase in the number of transistors
per semiconductors each year.  In 2015, semiconductor manufacturers
claimed they could not keep up with Moore's law anymore. Your analysis
shows that since 1971, the average increase in transistor count was
x1.98 every 2 years, but Gordon Moore predicted it would be x2
every 2 years. That is an amazing prediction.

선형 회귀는 매년 반도체당 트랜지스터 수의 증가를 포착합니다. 2015년에 반도체 제조업체들은 더 이상 무어의 법칙을 따라갈 수 없다고 주장했습니다. 당신의 분석에 따르면 1971년 이후로 트랜지스터 수의 평균 증가는 2년마다 1.98배였지만, 고든 무어는 2년마다 2배가 될 것이라고 예측했습니다. 그것은 놀라운 예측입니다.

Consider the year 2017. Compare the data to your linear regression
model and Gordon Moore's prediction. First, get the
transistor counts from the year 2017. You can do this with a Boolean
comparator,

`year == 2017`.

Then, make a prediction for 2017 with `Moores_law` defined above
and plugging in your best fit constants into your function

$\text{transistor_count} = e^{B}e^{A\cdot \text{year}}$.

A great way to compare these measurements is to compare your prediction
and Moore's prediction to the average transistor count and look at the
range of reported values for that year. Use the
[`plt.plot`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html)
option,
[`alpha=0.2`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.artist.Artist.set_alpha.html),
to increase the transparency of the data. The more opaque the points
appear, the more reported values lie on that measurement. The green $+$
is the average reported transistor count for 2017. Plot your predictions
for $\pm\frac{1}{2}~years.

2017년을 고려해보세요. 데이터를 선형 회귀 모델과 고든 무어의 예측과 비교하세요. 먼저 2017년의 트랜지스터 수를 가져옵니다. 이것은 불리언 비교기를 사용하여 할 수 있습니다,

`year == 2017`.

그런 다음 위에서 정의한 `Moores_law`로 2017년에 대한 예측을 하고 최적 피팅된 상수를 함수에 대입합니다

$\text{transistor_count} = e^{B}e^{A\cdot \text{year}}$.

이러한 측정을 비교하는 좋은 방법은 당신의 예측과 무어의 예측을 평균 트랜지스터 수와 비교하고 해당 연도의 보고된 값의 범위를 살펴보는 것입니다. [`plt.plot`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html) 옵션인 [`alpha=0.2`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.artist.Artist.set_alpha.html)를 사용하여 데이터의 투명도를 증가시킵니다. 점이 더 불투명하게 나타날수록 해당 측정값에 더 많은 보고된 값이 있음을 의미합니다. 녹색 $+$는 2017년의 평균 보고된 트랜지스터 수입니다. $\pm\frac{1}{2}$ 년에 대한 예측을 그래프로 표시합니다.

```{code-cell}
transistor_count2017 = transistor_count[year == 2017]
print(
    transistor_count2017.max(), transistor_count2017.min(), transistor_count2017.mean()
)
y = np.linspace(2016.5, 2017.5)
your_model2017 = np.exp(B) * np.exp(A * y)
Moore_Model2017 = Moores_law(y)

plt.plot(
    2017 * np.ones(np.sum(year == 2017)),
    transistor_count2017,
    "ro",
    label="2017",
    alpha=0.2,
)
plt.plot(2017, transistor_count2017.mean(), "g+", markersize=20, mew=6)

plt.plot(y, your_model2017, label="Your prediction")
plt.plot(y, Moore_Model2017, label="Moores law")
plt.ylabel("# of transistors\nper microprocessor")
plt.legend()
```

The result is that your model is close to the mean, but Gordon
Moore's prediction is closer to the maximum number of transistors per
microprocessor produced in 2017. Even though semiconductor manufacturers
thought that the growth would slow, once in 1975 and now again
approaching 2025, manufacturers are still producing semiconductors every 2 years that
nearly double the number of transistors.

결과적으로 당신의 모델은 평균에 가깝지만, 고든 무어의 예측은 2017년에 생산된 마이크로프로세서당 최대 트랜지스터 수에 더 가깝습니다. 반도체 제조업체들이 성장이 느려질 것이라고 생각했지만, 1975년에 한 번, 그리고 이제 2025년에 다시 접근하고 있음에도 불구하고, 제조업체들은 여전히 매 2년마다 트랜지스터 수를 거의 두 배로.

The linear regression model is much better at predicting the
average than extreme values because it satisfies the condition to
minimize $\sum |y_i - A\cdot \text{year}[i]+B|^2$.

선형 회귀 모델은 $\sum |y_i - A\cdot \text{year}[i]+B|^2$를 최소화하는 조건을 만족하기 때문에 극단적인 값보다 평균을 예측하는 데 훨씬 더 좋습니다.

+++

## Sharing your results as zipped arrays and a csv
## 결과를 압축된 배열과 csv로 공유하기

The last step, is to share your findings. You created
new arrays that represent a linear regression model and Gordon Moore's
prediction. You started this process by importing a csv file into a NumPy
array using `np.loadtxt`, to save your model use two approaches

1. [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html): save NumPy arrays for other Python sessions
2. [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html): save a csv file with the original data and your predicted data

마지막 단계는 결과를 공유하는 것입니다. 선형 회귀 모델과 고든 무어의 예측을 나타내는 새로운 배열을 만들었습니다. `np.loadtxt`를 사용하여 csv 파일을 NumPy 배열로 가져오는 것으로 이 과정을 시작했으니, 모델을 저장하기 위해 두 가지 접근 방식을 사용하세요

1. [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html): 다른 Python 세션을 위해 NumPy 배열 저장
2. [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html): 원본 데이터와 예측 데이터가 포함된 csv 파일 저장

### Zipping the arrays into a file
### 배열을 파일로 압축하기
Using `np.savez`, you can save thousands of arrays and give them names. The
function `np.load` will load the arrays back into the workspace as a
dictionary. You'll save a five arrays so the next user will have the year,
transistor count, predicted transistor count,  Gordon Moore's
predicted count, and fitting constants. Add one more variable that other users can use to
understand the model, `notes`.

`np.savez`를 사용하면 수천 개의 배열을 저장하고 이름을 지정할 수 있습니다. `np.load` 함수는 배열을 사전으로 다시 작업 공간에 로드합니다. 다섯 개의 배열을 저장하여 다음 사용자가 연도, 트랜지스터 수, 예측된 트랜지스터 수, 고든 무어의 예측 수, 그리고 피팅 상수를 가질 수 있게 합니다. 다른 사용자가 모델을 이해하는 데 사용할 수 있는 변수 `notes`를 하나 더 추가합니다.

```{code-cell}
notes = "the arrays in this file are the result of a linear regression model\n"
notes += "the arrays include\nyear: year of manufacture\n"
notes += "transistor_count: number of transistors reported by manufacturers in a given year\n"
notes += "transistor_count_predicted: linear regression model = exp({:.2f})*exp({:.2f}*year)\n".format(
    B, A
)
notes += "transistor_Moores_law: Moores law =exp({:.2f})*exp({:.2f}*year)\n".format(
    B_M, A_M
)
notes += "regression_csts: linear regression constants A and B for log(transistor_count)=A*year+B"
print(notes)
```

```{code-cell}
np.savez(
    "mooreslaw_regression.npz",
    notes=notes,
    year=year,
    transistor_count=transistor_count,
    transistor_count_predicted=transistor_count_predicted,
    transistor_Moores_law=transistor_Moores_law,
    regression_csts=(A, B),
)
```

```{code-cell}
results = np.load("mooreslaw_regression.npz")
```

```{code-cell}
print(results["regression_csts"][1])
```

```{code-cell}
! ls
```

The benefit of `np.savez` is you can save hundreds of arrays with
different shapes and types. Here, you saved 4 arrays that are double
precision floating point numbers shape = `(179,)`, one array that was
text, and one array of double precision floating point numbers shape =
`(2,).` This is the preferred method for saving NumPy arrays for use in
another analysis.

`np.savez`의 이점은 다양한 형태와 유형의 수백 개의 배열을 저장할 수 있다는 것입니다. 여기서는 shape = `(179,)`인 4개의 배정밀도 부동 소수점 숫자 배열, 하나의 텍스트 배열, 그리고 shape = `(2,)`인 하나의 배정밀도 부동 소수점 숫자 배열을 저장했습니다. 이것이 다른 분석에 사용하기 위해 NumPy 배열을 저장하는 선호되는 방법입니다.

### Creating your own comma separated value file
### 자신만의 쉼표로 구분된 값 파일 만들기

If you want to share data and view the results in a table, then you have to
create a text file. Save the data using `np.savetxt`. This
function is more limited than `np.savez`. Delimited files, like csv's,
need 2D arrays.

데이터를 공유하고 결과를 표로 보고 싶다면 텍스트 파일을 만들어야 합니다. `np.savetxt`를 사용하여 데이터를 저장하세요. 이 함수는 `np.savez`보다 더 제한적입니다. csv와 같은 구분된 파일은 2D 배열이 필요합니다.

Prepare the data for export by creating a new 2D array whose columns
contain the data of interest.

데이터를 내보내기 위해 관심 있는 데이터를 포함하는 열이 있는 새로운 2D 배열을 만들어 준비합니다.

Use the `header` option to describe the data and the columns of
the file. Define another variable that contains file
information as `head`.

`header` 옵션을 사용하여 데이터와 파일의 열을 설명합니다. 파일 정보를 포함하는 다른 변수를 `head`로 정의합니다.

```{code-cell}
head = "the columns in this file are the result of a linear regression model\n"
head += "the columns include\nyear: year of manufacture\n"
head += "transistor_count: number of transistors reported by manufacturers in a given year\n"
head += "transistor_count_predicted: linear regression model = exp({:.2f})*exp({:.2f}*year)\n".format(
    B, A
)
head += "transistor_Moores_law: Moores law =exp({:.2f})*exp({:.2f}*year)\n".format(
    B_M, A_M
)
head += "year:, transistor_count:, transistor_count_predicted:, transistor_Moores_law:"
print(head)
```

Build a single 2D array to export to csv. Tabular data is inherently two
dimensional. You need to organize your data to fit this 2D structure.
Use `year`, `transistor_count`, `transistor_count_predicted`, and
`transistor_Moores_law` as the first through fourth columns,
respectively. Put the calculated constants in the header since they do
not fit the `(179,)` shape. The
[`np.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html)
function appends arrays together to create a new, larger array. Arrange
the 1D vectors as columns using
[`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html)
e.g.

csv로 내보낼 단일 2D 배열을 만듭니다. 표 형식 데이터는 본질적으로 2차원입니다. 이 2D 구조에 맞게 데이터를 구성해야 합니다. `year`, `transistor_count`, `transistor_count_predicted`, 그리고 `transistor_Moores_law`를 각각 첫 번째부터 네 번째 열로 사용합니다. 계산된 상수는 `(179,)` 형태에 맞지 않으므로 헤더에 넣습니다. [`np.block`](https://numpy.org/doc/stable/reference/generated/numpy.block.html) 함수는 배열을 함께 추가하여 새롭고 더 큰 배열을 만듭니다. [`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html)를 사용하여 1D 벡터를 열로 배열합니다. 예:

```python
>>> year.shape
(179,)
>>> year[:,np.newaxis].shape
(179,1)
```

```{code-cell}
output = np.block(
    [
        year[:, np.newaxis],
        transistor_count[:, np.newaxis],
        transistor_count_predicted[:, np.newaxis],
        transistor_Moores_law[:, np.newaxis],
    ]
)
```

Creating the `mooreslaw_regression.csv` with `np.savetxt`, use three
options to create the desired file format:

* `X = output` : use `output` block to write the data into the file
* `delimiter = ','` : use commas to separate columns in the file
* `header = head` : use the header `head` defined above

`np.savetxt`를 사용하여 `mooreslaw_regression.csv`를 만들 때, 원하는 파일 형식을 만들기 위해 세 가지 옵션을 사용합니다:

* `X = output` : `output` 블록을 사용하여 데이터를 파일에 씁니다
* `delimiter = ','` : 파일에서 열을 구분하기 위해 쉼표를 사용합니다
* `header = head` : 위에서 정의한 헤더 `head`를 사용합니다

```{code-cell}
np.savetxt("mooreslaw_regression.csv", X=output, delimiter=",", header=head)
```

```{code-cell}
! head mooreslaw_regression.csv
```

## Wrapping up
## 정리하며

In conclusion, you have compared historical data for semiconductor
manufacturers to Moore's law and created a linear regression model to
find the average number of transistors added to each microprocessor
every two years. Gordon Moore predicted the number of transistors would
double every two years from 1965 through 1975, but the average growth
has maintained a consistent increase of $\times 1.98 \pm 0.01$ every two
years from 1971 through 2019.  In 2015, Moore revised his prediction to
say Moore's law should hold until 2025.
[[2](https://spectrum.ieee.org/computing/hardware/gordon-moore-the-man-whose-name-means-progress)].
You can share these results as a zipped NumPy array file,
`mooreslaw_regression.npz`, or as another csv,
`mooreslaw_regression.csv`.  The amazing progress in semiconductor
manufacturing has enabled new industries and computational power. This
analysis should give you a small insight into how incredible this growth
has been over the last half-century.

결론적으로, 반도체 제조업체의 역사적 데이터를 무어의 법칙과 비교하고 선형 회귀 모델을 만들어 매 2년마다 각 마이크로프로세서에 추가되는 평균 트랜지스터 수를 찾았습니다. 고든 무어는 1965년부터 1975년까지 트랜지스터 수가 매 2년마다 두 배가 될 것이라고 예측했지만, 1971년부터 2019년까지 평균 성장률은 매 2년마다 일관되게 $\times 1.98 \pm 0.01$ 증가를 유지했습니다. 2015년에 무어는 자신의 예측을 수정하여 무어의 법칙이 2025년까지 유지될 것이라고 말했습니다[[2](https://spectrum.ieee.org/computing/hardware/gordon-moore-the-man-whose-name-means-progress)]. 이러한 결과는 압축된 NumPy 배열 파일인 `mooreslaw_regression.npz` 또는 다른 csv인 `mooreslaw_regression.csv`로 공유할 수 있습니다. 반도체 제조 분야의 놀라운 발전은 새로운 산업과 계산 능력을 가능하게 했습니다. 이 분석은 지난 반세기 동안 이 성장이 얼마나 놀라웠는지에 대한 작은 통찰력을 제공할 것입니다.

+++

## References
## 참고 문헌

1. ["Moore's Law." Wikipedia article. Accessed Oct. 1, 2020.](https://en.wikipedia.org/wiki/Moore%27s_law)
2. [Courtland, Rachel. "Gordon Moore: The Man Whose Name Means Progress." IEEE Spectrum. 30 Mar. 2015.](https://spectrum.ieee.org/computing/hardware/gordon-moore-the-man-whose-name-means-progress).
3. ["Transistor Count." Wikipedia article. Accessed Oct. 1, 2020.](https://en.wikipedia.org/wiki/Transistor_count#Microprocessors)