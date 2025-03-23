# Analyzing the impact of the lockdown on air quality in Delhi, India
# 인도 델리의 공기질에 대한 봉쇄 조치의 영향 분석
A grid showing the India Gate in smog above and clear air below

상단에는 스모그에 둘러싸인 인도 게이트, 하단에는 맑은 공기를 보여주는 격자 이미지

## What you'll do
## 무엇을 할 것인가
Calculate Air Quality Indices (AQI) and perform paired Student's t-test on them.

대기질 지수(AQI)를 계산하고 이에 대한 대응표본 t-검정을 수행합니다.

## What you'll learn
## 배우게 될 것
You'll learn the concept of moving averages

이동 평균의 개념을 배웁니다

You'll learn how to calculate Air Quality Index (AQI)

대기질 지수(AQI)를 계산하는 방법을 배웁니다

You'll learn how to perform a paired Student's t-test and find the t and p values

대응표본 t-검정을 수행하고 t값과 p값을 찾는 방법을 배웁니다

You'll learn how to interpret these values

이러한 값들을 해석하는 방법을 배웁니다

## What you'll need
## 필요한 것
SciPy installed in your environment

환경에 SciPy가 설치되어 있어야 합니다

Basic understanding of statistical terms like population, sample, mean, standard deviation etc.

모집단, 표본, 평균, 표준편차 등의 통계 용어에 대한 기본적인 이해가 필요합니다

## The problem of air pollution
## 대기 오염 문제

Air pollution is one of the most prominent types of pollution we face that has an immediate effect on our daily lives. The COVID-19 pandemic resulted in lockdowns in different parts of the world; offering a rare opportunity to study the effect of human activity (or lack thereof) on air pollution. In this tutorial, we will study the air quality in Delhi, one of the worst affected cities by air pollution, before and during the lockdown from March to June 2020. For this, we will first compute the Air Quality Index for each hour from the collected pollutant measurements. Next, we will sample these indices and perform a paired Student's t-test on them. It will statistically show us that the air quality improved due to the lockdown, supporting our intuition.

대기 오염은 우리의 일상생활에 즉각적인 영향을 미치는 가장 두드러진 오염 유형 중 하나입니다. COVID-19 팬데믹으로 전 세계 여러 지역에 봉쇄 조치가 시행되었고, 이는 대기 오염에 대한 인간 활동(또는 그 부재)의 영향을 연구할 수 있는 드문 기회를 제공했습니다. 이 튜토리얼에서는 대기 오염으로 가장 심각한 영향을 받은 도시 중 하나인 델리의 2020년 3월부터 6월까지의 봉쇄 전과 봉쇄 기간 동안의 대기질을 연구할 것입니다. 이를 위해 먼저 수집된 오염물질 측정값으로부터 매 시간 대기질 지수를 계산할 것입니다. 다음으로, 이러한 지수들을 샘플링하고 대응표본 t-검정을 수행할 것입니다. 이는 봉쇄로 인해 대기질이 개선되었음을 통계적으로 보여주어 우리의 직관을 뒷받침할 것입니다.

Let's start by importing the necessary libraries into our environment.

필요한 라이브러리를 환경으로 가져오는 것부터 시작하겠습니다.

```{code-cell}
import numpy as np
from numpy.random import default_rng
from scipy import stats
```

## Building the dataset
## 데이터셋 구축하기

We will use a condensed version of the Air Quality Data in India dataset. This dataset contains air quality data and AQI (Air Quality Index) at hourly and daily level of various stations across multiple cities in India. The condensed version available with this tutorial contains hourly pollutant measurements for Delhi from May 31, 2019 to June 30, 2020. It has measurements of the standard pollutants that are required for Air Quality Index calculation and a few other important ones: Particulate Matter (PM 2.5 and PM 10), nitrogen dioxide (NO2), ammonia (NH3), sulfur dioxide (SO2), carbon monoxide (CO), ozone (O3), oxides of nitrogen (NOx), nitric oxide (NO), benzene, toluene, and xylene.

우리는 인도의 대기질 데이터 데이터셋의 축약 버전을 사용할 것입니다. 이 데이터셋은 인도의 여러 도시에 걸쳐 다양한 측정소에서 시간별 및 일별 대기질 데이터와 AQI(대기질 지수)를 포함하고 있습니다. 이 튜토리얼에서 사용 가능한 축약 버전은 2019년 5월 31일부터 2020년 6월 30일까지 델리의 시간별 오염물질 측정값을 포함하고 있습니다. 이 데이터셋은 대기질 지수 계산에 필요한 표준 오염물질과 몇 가지 다른 중요한 물질의 측정값을 포함하고 있습니다: 미세먼지(PM 2.5 및 PM 10), 이산화질소(NO2), 암모니아(NH3), 이산화황(SO2), 일산화탄소(CO), 오존(O3), 질소산화물(NOx), 일산화질소(NO), 벤젠, 톨루엔, 자일렌.

Let's print out the first few rows to have a glimpse of our dataset.

데이터셋의 처음 몇 줄을 출력하여 간략히 살펴보겠습니다.

```{code-cell}
! head air-quality-data.csv
```

For the purpose of this tutorial, we are only concerned with standard pollutants required for calculating the AQI, viz., PM 2.5, PM 10, NO2, NH3, SO2, CO, and O3. So, we will only import these particular columns with np.loadtxt. We'll then slice and create two sets: pollutants_A with PM 2.5, PM 10, NO2, NH3, and SO2, and pollutants_B with CO and O3. The two sets will be processed slightly differently, as we'll see later on.

이 튜토리얼의 목적상, 우리는 AQI 계산에 필요한 표준 오염물질, 즉 PM 2.5, PM 10, NO2, NH3, SO2, CO, O3만 다룰 것입니다. 따라서 np.loadtxt를 사용하여 이러한 특정 열만 가져올 것입니다. 그런 다음 데이터를 분할하여 두 세트를 만들 것입니다: PM 2.5, PM 10, NO2, NH3, SO2가 포함된 pollutants_A와 CO와 O3가 포함된 pollutants_B입니다. 이 두 세트는 나중에 볼 수 있듯이 약간 다르게 처리될 것입니다.

```{code-cell}
pollutant_data = np.loadtxt("air-quality-data.csv", dtype=float, delimiter=",",
                            skiprows=1, usecols=range(1, 8))
pollutants_A = pollutant_data[:, 0:5]
pollutants_B = pollutant_data[:, 5:]

print(pollutants_A.shape)
print(pollutants_B.shape)
```

Our dataset might contain missing values, denoted by NaN, so let's do a quick check with np.isfinite.

우리의 데이터셋은 NaN으로 표시된 누락된 값을 포함할 수 있으므로, np.isfinite로 빠르게 확인해 보겠습니다.

```{code-cell}
np.all(np.isfinite(pollutant_data))
```

With this, we have successfully imported the data and checked that it is complete. Let's move on to the AQI calculations!

이로써 데이터를 성공적으로 가져오고 완전한지 확인했습니다. 이제 AQI 계산으로 넘어가겠습니다!

## Calculating the Air Quality Index
## 대기질 지수 계산하기

We will calculate the AQI using the method adopted by the Central Pollution Control Board of India. To summarize the steps:

인도 중앙 오염 통제 위원회가 채택한 방법을 사용하여 AQI를 계산할 것입니다. 단계를 요약하면 다음과 같습니다:

Collect 24-hourly average concentration values for the standard pollutants; 8-hourly in case of CO and O3.

표준 오염물질에 대한 24시간 평균 농도 값을 수집합니다; CO와 O3의 경우 8시간 평균입니다.

Calculate the sub-indices for these pollutants with the formula:

다음 공식을 사용하여 이러한 오염물질에 대한 하위 지수를 계산합니다:

 
Where,

Ip = sub-index of pollutant p
Cp = averaged concentration of pollutant p
BPHi = concentration breakpoint i.e. greater than or equal to Cp
BPLo = concentration breakpoint i.e. less than or equal to Cp
IHi = AQI value corresponding to BPHi
ILo = AQI value corresponding to BPLo

여기서,

Ip = 오염물질 p의 하위 지수
Cp = 오염물질 p의 평균 농도
BPHi = 농도 변곡점, 즉 Cp보다 크거나 같은 값
BPLo = 농도 변곡점, 즉 Cp보다 작거나 같은 값
IHi = BPHi에 해당하는 AQI 값
ILo = BPLo에 해당하는 AQI 값

The maximum sub-index at any given time is the Air Quality Index.

주어진 시간에 최대 하위 지수가 대기질 지수입니다.

The Air Quality Index is calculated with the help of breakpoint ranges as shown in the chart below.

대기질 지수는 아래 차트에 표시된 변곡점 범위를 사용하여 계산됩니다.

Chart of the breakpoint ranges

변곡점 범위 차트

Let's create two arrays to store the AQI ranges and breakpoints so that we can use them later for our calculations.

나중에 계산에 사용할 수 있도록 AQI 범위와 변곡점을 저장할 두 개의 배열을 만들어 보겠습니다.

```{code-cell}
AQI = np.array([0, 51, 101, 201, 301, 401, 501])

breakpoints = {
    'PM2.5': np.array([0, 31, 61, 91, 121, 251]),
    'PM10': np.array([0, 51, 101, 251, 351, 431]),
    'NO2': np.array([0, 41, 81, 181, 281, 401]),
    'NH3': np.array([0, 201, 401, 801, 1201, 1801]),
    'SO2': np.array([0, 41, 81, 381, 801, 1601]),
    'CO': np.array([0, 1.1, 2.1, 10.1, 17.1, 35]),
    'O3': np.array([0, 51, 101, 169, 209, 749])
}
```

## Moving averages
## 이동 평균

For the first step, we have to compute moving averages for pollutants_A over a window of 24 hours and pollutants_B over a window of 8 hours. We will write a simple function moving_mean using np.cumsum and sliced indexing to achieve this.

첫 번째 단계로, pollutants_A에 대해 24시간 창, pollutants_B에 대해 8시간 창에 걸친 이동 평균을 계산해야 합니다. 이를 위해 np.cumsum과 슬라이스 인덱싱을 사용하여 moving_mean이라는 간단한 함수를 작성할 것입니다.

To make sure both the sets are of the same length, we will truncate the pollutants_B_8hr_avg according to the length of pollutants_A_24hr_avg. This will also ensure we have concentrations for all the pollutants over the same period of time.

두 세트가 같은 길이가 되도록 하기 위해, pollutants_A_24hr_avg의 길이에 따라 pollutants_B_8hr_avg를 잘라낼 것입니다. 이렇게 하면 동일한 시간 기간 동안 모든 오염물질에 대한 농도를 확보할 수 있습니다.

```{code-cell}
def moving_mean(a, n):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

pollutants_A_24hr_avg = moving_mean(pollutants_A, 24)
pollutants_B_8hr_avg = moving_mean(pollutants_B, 8)[-(pollutants_A_24hr_avg.shape[0]):]
```

Now, we can join both sets with np.concatenate to form a single data set of all the averaged concentrations. Note that we have to join our arrays column-wise so we pass the axis=1 parameter.

이제 np.concatenate를 사용하여 두 세트를 모든 평균 농도의 단일 데이터 세트로 결합할 수 있습니다. 배열을 열 방향으로 결합해야 하므로 axis=1 매개변수를 전달합니다.

```{code-cell}
pollutants = np.concatenate((pollutants_A_24hr_avg, pollutants_B_8hr_avg), axis=1)
```

## Sub-indices
## 하위 지수

The subindices for each pollutant are calculated according to the linear relationship between the AQI and standard breakpoint ranges with the formula as above:

각 오염물질에 대한 하위 지수는 위 공식을 사용하여 AQI와 표준 변곡점 범위 사이의 선형 관계에 따라 계산됩니다:

 
The compute_indices function first fetches the correct upper and lower bounds of AQI categories and breakpoint concentrations for the input concentration and pollutant with the help of arrays AQI and breakpoints we created above. Then, it feeds these values into the formula to calculate the sub-index.

compute_indices 함수는 먼저 위에서 만든 AQI 및 breakpoints 배열을 사용하여 입력 농도와 오염물질에 대한 AQI 카테고리의 정확한 상한 및 하한과 변곡점 농도를 가져옵니다. 그런 다음 이러한 값을 공식에 대입하여 하위 지수를 계산합니다.

```{code-cell}
def compute_indices(pol, con):
    bp = breakpoints[pol]
    
    if pol == 'CO':
        inc = 0.1
    else:
        inc = 1
    
    if bp[0] <= con < bp[1]:
        Bl = bp[0]
        Bh = bp[1] - inc
        Ih = AQI[1] - inc
        Il = AQI[0]

    elif bp[1] <= con < bp[2]:
        Bl = bp[1]
        Bh = bp[2] - inc
        Ih = AQI[2] - inc
        Il = AQI[1]

    elif bp[2] <= con < bp[3]:
        Bl = bp[2]
        Bh = bp[3] - inc
        Ih = AQI[3] - inc
        Il = AQI[2]

    elif bp[3] <= con < bp[4]:
        Bl = bp[3]
        Bh = bp[4] - inc
        Ih = AQI[4] - inc
        Il = AQI[3]

    elif bp[4] <= con < bp[5]:
        Bl = bp[4]
        Bh = bp[5] - inc
        Ih = AQI[5] - inc
        Il = AQI[4]

    elif bp[5] <= con:
        Bl = bp[5]
        Bh = bp[5] + bp[4] - (2 * inc)
        Ih = AQI[6]
        Il = AQI[5]

    else:
        print("Concentration out of range!")
        
    return ((Ih - Il) / (Bh - Bl)) * (con - Bl) + Il
```

We will use np.vectorize to utilize the concept of vectorization. This simply means we don't have loop over each element of the pollutant array ourselves. Vectorization is one of the key advantages of NumPy.

벡터화 개념을 활용하기 위해 np.vectorize를 사용할 것입니다. 이는 단순히 우리가 직접 오염물질 배열의 각 요소에 대해 반복할 필요가 없다는 것을 의미합니다. 벡터화는 NumPy의 핵심 장점 중 하나입니다.

```{code-cell}
vcompute_indices = np.vectorize(compute_indices)
```

By calling our vectorized function vcompute_indices for each pollutant, we get the sub-indices. To get back an array with the original shape, we use np.stack.

각 오염물질에 대해 벡터화된 함수 vcompute_indices를 호출하여 하위 지수를 얻습니다. 원래 모양의 배열을 되찾기 위해 np.stack을 사용합니다.

```{code-cell}
sub_indices = np.stack((vcompute_indices('PM2.5', pollutants[..., 0]),
                        vcompute_indices('PM10', pollutants[..., 1]),
                        vcompute_indices('NO2', pollutants[..., 2]),
                        vcompute_indices('NH3', pollutants[..., 3]),
                        vcompute_indices('SO2', pollutants[..., 4]),
                        vcompute_indices('CO', pollutants[..., 5]),
                        vcompute_indices('O3', pollutants[..., 6])), axis=1)
```

## Air quality indices
## 대기질 지수

Using np.max, we find out the maximum sub-index for each period, which is our Air Quality Index!

np.max를 사용하여 각 기간에 대한 최대 하위 지수를 찾습니다. 이것이 우리의 대기질 지수입니다!

```{code-cell}
aqi_array = np.max(sub_indices, axis=1)
```

With this, we have the AQI for every hour from June 1, 2019 to June 30, 2020. Note that even though we started out with the data from 31st May, we truncated that during the moving averages step.

이로써 2019년 6월 1일부터 2020년 6월 30일까지 매 시간의 AQI를 얻었습니다. 5월 31일 데이터부터 시작했지만 이동 평균 단계에서 잘라냈다는 점에 유의하세요.

## Paired Student's t-test on the AQIs
## AQI에 대한 대응표본 t-검정

Hypothesis testing is a form of descriptive statistics used to help us make decisions with the data. From the calculated AQI data, we want to find out if there was a statistically significant difference in average AQI before and after the lockdown was imposed. We will use the left-tailed, paired Student's t-test to compute two test statistics- the t statistic and the p value. We will then compare these with the corresponding critical values to make a decision.

가설 검정은 데이터로 결정을 내리는 데 도움이 되는 기술통계의 한 형태입니다. 계산된 AQI 데이터를 통해 봉쇄 조치가 시행되기 전과 후의 평균 AQI에 통계적으로 유의미한 차이가 있었는지 알아보고자 합니다. 왼쪽 꼬리, 대응표본 t-검정을 사용하여 두 가지 검정 통계량인 t 통계량과 p값을 계산할 것입니다. 그런 다음 이를 해당 임계값과 비교하여 결정을 내릴 것입니다.

Normal distribution plot showing area of rejection in one-tailed test (left tailed)

단측 검정(왼쪽 꼬리)에서 기각 영역을 보여주는 정규 분포 그래프

## Sampling
## 샘플링

We will now import the datetime column from our original dataset into a datetime64 dtype array. We will use this array to index the AQI array and obtain subsets of the dataset.

이제 원본 데이터셋에서 datetime 열을 datetime64 dtype 배열로 가져올 것입니다. 이 배열을 사용하여 AQI 배열을 인덱싱하고 데이터셋의 하위 집합을 얻을 것입니다.

```{code-cell}
datetime = np.loadtxt("air-quality-data.csv", dtype='M8[h]', delimiter=",",
                         skiprows=1, usecols=(0, ))[-(pollutants_A_24hr_avg.shape[0]):]
```

Since total lockdown commenced in Delhi from March 24, 2020, the after-lockdown subset is of the period March 24, 2020 to June 30, 2020. The before-lockdown subset is for the same length of time before 24th March.

2020년 3월 24일부터 델리에서 전면 봉쇄가 시작되었으므로, 봉쇄 후 하위 집합은 2020년 3월 24일부터 2020년 6월 30일까지의 기간입니다. 봉쇄 전 하위 집합은 3월 24일 이전의 동일한 기간입니다.

```{code-cell}
after_lock = aqi_array[np.where(datetime >= np.datetime64('2020-03-24T00'))]

before_lock = aqi_array[np.where(datetime <= np.datetime64('2020-03-21T00'))][-(after_lock.shape[0]):]

print(after_lock.shape)
print(before_lock.shape)
```

To make sure our samples are approximately normally distributed, we take samples of size n = 30. before_sample and after_sample are the set of random observations drawn before and after the total lockdown. We use random.Generator.choice to generate the samples.

우리의 표본이 대략적으로 정규 분포를 따르도록 하기 위해 크기 n = 30인 표본을 선택합니다. before_sample과 after_sample은 전면 봉쇄 전후에 추출된 무작위 관측값 세트입니다. random.Generator.choice를 사용하여 표본을 생성합니다.

```{code-cell}
rng = default_rng()

before_sample = rng.choice(before_lock, size=30, replace=False)
after_sample = rng.choice(after_lock, size=30, replace=False)
```

## Defining the hypothesis
## 가설 정의하기

Let us assume that there is no significant difference between the sample means before and after the lockdown. This will be the null hypothesis. The alternative hypothesis would be that there is a significant difference between the means and the AQI improved. Mathematically,

봉쇄 전후의 표본 평균 사이에 유의미한 차이가 없다고 가정해 봅시다. 이것이 귀무가설이 될 것입니다. 대립가설은 평균 사이에 유의미한 차이가 있고 AQI가 개선되었다는 것입니다. 수학적으로,



## Calculating the test statistics
## 검정 통계량 계산하기

We will use the t statistic to evaluate our hypothesis and even calculate the p value from it. The formula for the t statistic is:

가설을 평가하고 p값까지 계산하기 위해 t 통계량을 사용할 것입니다. t 통계량의 공식은 다음과 같습니다:

 
where,

 = mean differences of samples
 = variance of mean differences
 = sample size

여기서,

 = 표본 평균 차이
 = 평균 차이의 분산
 = 표본 크기

```{code-cell}
def t_test(x, y):
    diff = y - x
    var = np.var(diff, ddof=1)
    num = np.mean(diff)
    denom = np.sqrt(var / len(x))
    return np.divide(num, denom)

t_value = t_test(before_sample, after_sample)
```

For the p value, we will use SciPy's stats.distributions.t.cdf() function. It takes two arguments- the t statistic and the degrees of freedom (dof). The formula for dof is n - 1.

p값을 위해 SciPy의 stats.distributions.t.cdf() 함수를 사용할 것입니다. 이 함수는 두 가지 인수, 즉 t 통계량과 자유도(dof)를 받습니다. 자유도 공식은 n - 1입니다.

```{code-cell}
dof = len(before_sample) - 1

p_value = stats.distributions.t.cdf(t_value, dof)

print("The t value is {} and the p value is {}.".format(t_value, p_value))
```

## What do the t and p values mean?
## t값과 p값은 무엇을 의미하는가?

We will now compare the calculated test statistics with the critical test statistics. The critical t value is calculated by looking up the t-distribution table.

이제 계산된 검정 통계량을 임계 검정 통계량과 비교할 것입니다. 임계 t값은 t분포표를 참조하여 계산됩니다.

Table of selected t values at different confidence levels. T value for 29 dof at 95% confidence level is highlighted with a yellow square

다양한 신뢰 수준에서 선택된 t값 표. 신뢰 수준 95%에서 자유도 29의 t값이 노란색 사각형으로 강조되어 있습니다

From the table above, the critical value is 1.699 for 29 dof at a confidence level of 95%. Since we are using the left tailed test, our critical value is -1.699. Clearly, the calculated t value is less than the critical value so we can safely reject the null hypothesis.

위 표에서 자유도 29, 신뢰 수준 95%에 대한 임계값은 1.699입니다. 왼쪽 꼬리 검정을 사용하고 있으므로 우리의 임계값은 -1.699입니다. 계산된 t값이 임계값보다 작기 때문에 귀무가설을 안전하게 기각할 수 있습니다.

The critical p value, denoted by 
, is usually chosen to be 0.05, corresponding to a confidence level of 95%. If the calculated p value is less than 
, then the null hypothesis can be safely rejected. Clearly, our p value is much less than 
, so we can reject the null hypothesis.

임계 p값(
로 표시)은 일반적으로 0.05로 선택되며, 이는 95%의 신뢰 수준에 해당합니다. 계산된 p값이 
보다 작으면 귀무가설을 안전하게 기각할 수 있습니다. 분명히 우리의 p값은 
보다 훨씬 작으므로 귀무가설을 기각할 수 있습니다.

Note that this does not mean we can accept the alternative hypothesis. It only tells us that there is not enough evidence to reject 
. In other words, we fail to reject the alternative hypothesis so, it may be true.

이것이 대립가설을 수용할 수 있다는 의미는 아닙니다. 이는 단지 
을 기각할 충분한 증거가 없다는 것만 알려줍니다. 즉, 대립가설을 기각하지 못하므로 그것이 사실일 수 있습니다.
