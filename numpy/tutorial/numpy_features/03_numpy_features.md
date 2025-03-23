# Masked Arrays
# 마스크된 배열

## What you'll do
## 할 일

Use the masked arrays module from NumPy to analyze COVID-19 data and deal with missing values.
NumPy의 마스크된 배열 모듈을 사용하여 COVID-19 데이터를 분석하고 누락된 값을 처리합니다.

## What you'll learn
## 배울 내용

- You'll understand what are masked arrays and how they can be created
- 마스크된 배열이 무엇이고 어떻게 생성되는지 이해하게 됩니다
- You'll see how to access and modify data for masked arrays
- 마스크된 배열의 데이터에 접근하고 수정하는 방법을 알게 됩니다
- You'll be able to decide when the use of masked arrays is appropriate in some of your applications
- 애플리케이션에서 마스크된 배열의 사용이 적절한 시기를 결정할 수 있게 됩니다

## What you'll need
## 필요한 것

- Basic familiarity with Python. If you would like to refresh your memory, take a look at the Python tutorial.
- Python에 대한 기본적인 친숙함. 기억을 되살리고 싶다면 Python 튜토리얼을 참고하세요.
- Basic familiarity with NumPy
- NumPy에 대한 기본적인 친숙함
- To run the plots on your computer, you need matplotlib.
- 컴퓨터에서 그래프를 실행하려면 matplotlib이 필요합니다.

## What are masked arrays?
## 마스크된 배열이란 무엇인가?

Consider the following problem. You have a dataset with missing or invalid entries. If you're doing any kind of processing on this data, and want to skip or flag these unwanted entries without just deleting them, you may have to use conditionals or filter your data somehow. The numpy.ma module provides some of the same functionality of NumPy ndarrays with added structure to ensure invalid entries are not used in computation.
다음 문제를 고려해 보세요. 누락되거나 유효하지 않은 항목이 있는 데이터셋이 있습니다. 이 데이터를 처리하면서 이러한 원치 않는 항목을 단순히 삭제하지 않고 건너뛰거나 표시하려면 조건문을 사용하거나 데이터를 필터링해야 할 수 있습니다. numpy.ma 모듈은 유효하지 않은 항목이 계산에 사용되지 않도록 하는 추가 구조와 함께 NumPy ndarray의 일부 기능을 제공합니다.

From the Reference Guide:
참조 가이드에서:

A masked array is the combination of a standard numpy.ndarray and a mask. A mask is either nomask, indicating that no value of the associated array is invalid, or an array of booleans that determines for each element of the associated array whether the value is valid or not. When an element of the mask is False, the corresponding element of the associated array is valid and is said to be unmasked. When an element of the mask is True, the corresponding element of the associated array is said to be masked (invalid).
마스크된 배열은 표준 numpy.ndarray와 마스크의 조합입니다. 마스크는 관련 배열의 값이 유효하지 않음을 나타내는 nomask이거나, 관련 배열의 각 요소에 대해 값이 유효한지 여부를 결정하는 부울 배열입니다. 마스크의 요소가 False인 경우, 관련 배열의 해당 요소는 유효하며 마스크 해제된 것으로 간주됩니다. 마스크의 요소가 True인 경우, 관련 배열의 해당 요소는 마스크된(유효하지 않은) 것으로 간주됩니다.

We can think of a MaskedArray as a combination of:
MaskedArray를 다음의 조합으로 생각할 수 있습니다:

- Data, as a regular numpy.ndarray of any shape or datatype;
- 데이터, 모든 형태나 데이터 유형의 일반 numpy.ndarray로서;
- A boolean mask with the same shape as the data;
- 데이터와 동일한 형태를 가진 부울 마스크;
- A fill_value, a value that may be used to replace the invalid entries in order to return a standard numpy.ndarray.
- fill_value, 표준 numpy.ndarray를 반환하기 위해 유효하지 않은 항목을 대체하는 데 사용될 수 있는 값.

## When can they be useful?
## 언제 유용한가?

There are a few situations where masked arrays can be more useful than just eliminating the invalid entries of an array:
배열의 유효하지 않은 항목을 단순히 제거하는 것보다 마스크된 배열이 더 유용할 수 있는 몇 가지 상황이 있습니다:

- When you want to preserve the values you masked for later processing, without copying the array;
- 배열을 복사하지 않고 나중에 처리하기 위해 마스크한 값을 보존하려는 경우;
- When you have to handle many arrays, each with their own mask. If the mask is part of the array, you avoid bugs and the code is possibly more compact;
- 각각 자체 마스크가 있는 많은 배열을 처리해야 하는 경우. 마스크가 배열의 일부인 경우 버그를 방지하고 코드가 더 간결해질 수 있습니다;
- When you have different flags for missing or invalid values, and wish to preserve these flags without replacing them in the original dataset, but exclude them from computations;
- 누락되거나 유효하지 않은 값에 대해 다른 플래그가 있고, 원본 데이터셋에서 이러한 플래그를 교체하지 않고 보존하지만 계산에서 제외하려는 경우;
- If you can't avoid or eliminate missing values, but don't want to deal with NaN (Not a Number) values in your operations.
- 누락된 값을 피하거나 제거할 수 없지만 연산에서 NaN(숫자가 아님) 값을 처리하고 싶지 않은 경우.

Masked arrays are also a good idea since the numpy.ma module also comes with a specific implementation of most NumPy universal functions (ufuncs), which means that you can still apply fast vectorized functions and operations on masked data. The output is then a masked array. We'll see some examples of how this works in practice below.
마스크된 배열은 numpy.ma 모듈이 대부분의 NumPy 범용 함수(ufuncs)의 특정 구현을 제공하기 때문에 좋은 아이디어입니다. 이는 마스크된 데이터에 빠른 벡터화된 함수와 연산을 계속 적용할 수 있음을 의미합니다. 출력은 마스크된 배열입니다. 아래에서 이것이 실제로 어떻게 작동하는지에 대한 몇 가지 예를 살펴보겠습니다.

## Using masked arrays to see COVID-19 data
## 마스크된 배열을 사용하여 COVID-19 데이터 보기

From Kaggle it is possible to download a dataset with initial data about the COVID-19 outbreak in the beginning of 2020. We are going to look at a small subset of this data, contained in the file who_covid_19_sit_rep_time_series.csv. (Note that this file has been replaced with a version without missing data sometime in late 2020.)
Kaggle에서 2020년 초 COVID-19 발병에 관한 초기 데이터가 포함된 데이터셋을 다운로드할 수 있습니다. who_covid_19_sit_rep_time_series.csv 파일에 포함된 이 데이터의 작은 하위 집합을 살펴보겠습니다. (이 파일은 2020년 후반에 누락된 데이터가 없는 버전으로 대체되었습니다.)

```python
import numpy as np
import os

# The os.getcwd() function returns the current folder; you can change
# the filepath variable to point to the folder where you saved the .csv file
filepath = os.getcwd()
filename = os.path.join(filepath, "who_covid_19_sit_rep_time_series.csv")
```

The data file contains data of different types and is organized as follows:
데이터 파일에는 다양한 유형의 데이터가 포함되어 있으며 다음과 같이 구성되어 있습니다:

- The first row is a header line that (mostly) describes the data in each column that follow in the rows below, and beginning in the fourth column, the header is the date of the observation.
- 첫 번째 행은 아래 행에 이어지는 각 열의 데이터를 (대부분) 설명하는 헤더 행이며, 네 번째 열부터 헤더는 관측 날짜입니다.
- The second through seventh row contain summary data that is of a different type than that which we are going to examine, so we will need to exclude that from the data with which we will work.
- 두 번째부터 일곱 번째 행까지는 우리가 검토할 데이터와 다른 유형의 요약 데이터가 포함되어 있으므로, 우리가 작업할 데이터에서 이를 제외해야 합니다.
- The numerical data we wish to work with begins at column 4, row 8, and extends from there to the rightmost column and the lowermost row.
- 우리가 작업하고자 하는 수치 데이터는 4열, 8행에서 시작하여 가장 오른쪽 열과 가장 아래쪽 행까지 확장됩니다.

Let's explore the data inside this file for the first 14 days of records. To gather data from the .csv file, we will use the numpy.genfromtxt function, making sure we select only the columns with actual numbers instead of the first four columns which contain location data. We also skip the first 6 rows of this file, since they contain other data we are not interested in. Separately, we will extract the information about dates and location for this data.
이 파일의 처음 14일 기록 데이터를 살펴보겠습니다. .csv 파일에서 데이터를 수집하기 위해 numpy.genfromtxt 함수를 사용하여 위치 데이터가 포함된 처음 네 개의 열 대신 실제 숫자가 있는 열만 선택합니다. 또한 이 파일의 처음 6개 행은 우리가 관심 없는 다른 데이터를 포함하고 있으므로 건너뜁니다. 별도로, 이 데이터에 대한 날짜 및 위치 정보를 추출합니다.

```python
# Note we are using skip_header and usecols to read only portions of the
# data file into each variable.
# Read just the dates for columns 4-18 from the first row
dates = np.genfromtxt(
    filename,
    dtype=np.str_,
    delimiter=",",
    max_rows=1,
    usecols=range(4, 18),
    encoding="utf-8-sig",
)
# Read the names of the geographic locations from the first two
# columns, skipping the first six rows
locations = np.genfromtxt(
    filename,
    dtype=np.str_,
    delimiter=",",
    skip_header=6,
    usecols=(0, 1),
    encoding="utf-8-sig",
)
# Read the numeric data from just the first 14 days
nbcases = np.genfromtxt(
    filename,
    dtype=np.int_,
    delimiter=",",
    skip_header=6,
    usecols=range(4, 18),
    encoding="utf-8-sig",
)
```

Included in the numpy.genfromtxt function call, we have selected the numpy.dtype for each subset of the data (either an integer - numpy.int_ - or a string of characters - numpy.str_). We have also used the encoding argument to select utf-8-sig as the encoding for the file (read more about encoding in the official Python documentation. You can read more about the numpy.genfromtxt function from the Reference Documentation or from the Basic IO tutorial.
numpy.genfromtxt 함수 호출에 포함된 것으로, 데이터의 각 하위 집합에 대한 numpy.dtype을 선택했습니다(정수 - numpy.int_ - 또는 문자열 - numpy.str_). 또한 파일의 인코딩으로 utf-8-sig를 선택하기 위해 인코딩 인수를 사용했습니다(공식 Python 문서에서 인코딩에 대해 자세히 알아보세요). numpy.genfromtxt 함수에 대한 자세한 내용은 참조 문서 또는 기본 IO 튜토리얼에서 확인할 수 있습니다.

## Exploring the data
## 데이터 탐색하기

First of all, we can plot the whole set of data we have and see what it looks like. In order to get a readable plot, we select only a few of the dates to show in our x-axis ticks. Note also that in our plot command, we use nbcases.T (the transpose of the nbcases array) since this means we will plot each row of the file as a separate line. We choose to plot a dashed line (using the '--' line style). See the matplotlib documentation for more info on this.
우선, 우리가 가진 전체 데이터 세트를 그래프로 나타내고 어떻게 보이는지 확인할 수 있습니다. 읽기 쉬운 그래프를 얻기 위해 x축 눈금에 표시할 날짜 중 일부만 선택합니다. 또한 그래프 명령에서 nbcases.T(nbcases 배열의 전치)를 사용하는데, 이는 파일의 각 행을 별도의 선으로 그린다는 의미입니다. 점선(--' 선 스타일 사용)으로 그래프를 그립니다. 이에 대한 자세한 정보는 matplotlib 문서를 참조하세요.

```python
import matplotlib.pyplot as plt

selected_dates = [0, 3, 11, 13]
plt.plot(dates, nbcases.T, "--")
plt.xticks(selected_dates, dates[selected_dates])
plt.title("COVID-19 cumulative cases from Jan 21 to Feb 3 2020")
```

The graph has a strange shape from January 24th to February 1st. It would be interesting to know where this data comes from. If we look at the locations array we extracted from the .csv file, we can see that we have two columns, where the first would contain regions and the second would contain the name of the country. However, only the first few rows contain data for the the first column (province names in China). Following that, we only have country names. So it would make sense to group all the data from China into a single row. For this, we'll select from the nbcases array only the rows for which the second entry of the locations array corresponds to China. Next, we'll use the numpy.sum function to sum all the selected rows (axis=0). Note also that row 35 corresponds to the total counts for the whole country for each date. Since we want to calculate the sum ourselves from the provinces data, we have to remove that row first from both locations and nbcases:
1월 24일부터 2월 1일까지 그래프의 모양이 이상합니다. 이 데이터가 어디에서 왔는지 알아보는 것이 흥미로울 것입니다. .csv 파일에서 추출한 locations 배열을 보면 두 개의 열이 있으며, 첫 번째 열은 지역을, 두 번째 열은 국가 이름을 포함합니다. 그러나 처음 몇 개의 행만 첫 번째 열(중국의 성 이름)에 대한 데이터를 포함합니다. 그 이후에는 국가 이름만 있습니다. 따라서 중국의 모든 데이터를 단일 행으로 그룹화하는 것이 합리적입니다. 이를 위해 locations 배열의 두 번째 항목이 중국에 해당하는 행만 nbcases 배열에서 선택합니다. 다음으로 numpy.sum 함수를 사용하여 선택한 모든 행을 합산합니다(axis=0). 또한 35행은 각 날짜에 대한 전체 국가의 총 수를 나타냅니다. 성 데이터에서 직접 합계를 계산하려면 먼저 locations와 nbcases 모두에서 해당 행을 제거해야 합니다:

```python
totals_row = 35
locations = np.delete(locations, (totals_row), axis=0)
nbcases = np.delete(nbcases, (totals_row), axis=0)

china_total = nbcases[locations[:, 1] == "China"].sum(axis=0)
china_total
```

