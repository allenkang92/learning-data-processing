# Plotting Fractals
# 프랙탈 그리기
Fractal picture

프랙탈 이미지

Fractals are beautiful, compelling mathematical forms that can be oftentimes created from a relatively simple set of instructions. In nature they can be found in various places, such as coastlines, seashells, and ferns, and even were used in creating certain types of antennas. The mathematical idea of fractals was known for quite some time, but they really began to be truly appreciated in the 1970’s as advancements in computer graphics and some accidental discoveries lead researchers like Benoît Mandelbrot to stumble upon the truly mystifying visualizations that fractals possess.

프랙탈은 비교적 간단한 명령어 집합으로 만들어질 수 있는 아름답고 매력적인 수학적 형태입니다. 자연에서는 해안선, 조개껍데기, 고사리 등 다양한 곳에서 발견될 수 있으며, 심지어 특정 유형의 안테나를 만드는 데도 사용되었습니다. 프랙탈의 수학적 개념은 꽤 오랫동안 알려져 왔지만, 컴퓨터 그래픽의 발전과 우연한 발견으로 인해 베누아 만델브로트와 같은 연구자들이 프랙탈이 가진 진정으로 신비로운 시각화에 우연히 접하게 되면서 1970년대에 들어서야 진정으로 인정받기 시작했습니다.

Today we will learn how to plot these beautiful visualizations and will start to do a bit of exploring for ourselves as we gain familiarity of the mathematics behind fractals and will use the ever powerful NumPy universal functions to perform the necessary calculations efficiently.

오늘은 이러한 아름다운 시각화를 그리는 방법을 배우고, 프랙탈 뒤에 숨은 수학에 익숙해지면서 스스로 약간의 탐색을 시작할 것입니다. 또한 필요한 계산을 효율적으로 수행하기 위해 항상 강력한 NumPy 범용 함수를 사용할 것입니다.

## What you’ll do
## 무엇을 할 것인가
Write a function for plotting various Julia sets

다양한 줄리아 집합을 그리는 함수 작성하기

Create a visualization of the Mandelbrot set

만델브로트 집합의 시각화 만들기

Write a function that computes Newton fractals

뉴턴 프랙탈을 계산하는 함수 작성하기

Experiment with variations of general fractal types

일반적인 프랙탈 유형의 변형을 실험하기

## What you’ll learn
## 배우게 될 것
A better intuition for how fractals work mathematically

프랙탈이 수학적으로 어떻게 작동하는지에 대한 더 나은 직관

A basic understanding about NumPy universal functions and Boolean Indexing

NumPy 범용 함수와 불리언 인덱싱에 대한 기본적인 이해

The basics of working with complex numbers in NumPy

NumPy에서 복소수로 작업하는 기본 사항

How to create your own unique fractal visualizations

자신만의 고유한 프랙탈 시각화를 만드는 방법

## What you’ll need
## 필요한 것
Matplotlib

make_axis_locatable function from mpl_toolkits API

mpl_toolkits API의 make_axis_locatable 함수

which can be imported as follows:

다음과 같이 가져올 수 있습니다:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
Some familiarity with Python, NumPy and matplotlib

Python, NumPy 및 matplotlib에 대한 일부 친숙함

An idea of elementary mathematical functions, such as exponents, sin, polynomials etc

지수, 사인, 다항식 등과 같은 기초 수학 함수에 대한 이해

A very basic understanding of complex numbers would be useful

복소수에 대한 매우 기본적인 이해가 유용할 것입니다

Knowledge of derivatives may be helpful

미분에 대한 지식이 도움이 될 수 있습니다

## Warmup
## 워밍업

To gain some intuition for what fractals are, we will begin with an example.

프랙탈이 무엇인지에 대한 직관을 얻기 위해, 예제로 시작하겠습니다.

Consider the following equation:

다음 방정식을 고려해보세요:


where z is a complex number (i.e of the form 
 )

여기서 z는 복소수입니다(즉, 
 형태)

For our convenience, we will write a Python function for it

편의를 위해, 이를 위한 Python 함수를 작성하겠습니다

def f(z):
    return np.square(z) - 1
Note that the square function we used is an example of a NumPy universal function; we will come back to the significance of this decision shortly.

우리가 사용한 제곱 함수는 NumPy 범용 함수의 예입니다; 이 결정의 중요성에 대해 곧 다시 살펴보겠습니다.

To gain some intuition for the behaviour of the function, we can try plugging in some different values.

함수의 동작에 대한 직관을 얻기 위해, 몇 가지 다른 값을 대입해 볼 수 있습니다.

For 
, we would expect to get 
:


의 경우, 
를 얻을 것으로 예상됩니다:

f(0)
np.int64(-1)
Since we used a universal function in our design, we can compute multiple inputs at the same time:

설계에 범용 함수를 사용했기 때문에, 여러 입력을 동시에 계산할 수 있습니다:

z = [4, 1-0.2j, 1.6]
f(z)
array([15.  +0.j , -0.04-0.4j,  1.56+0.j ])
Some values grow, some values shrink, some don’t experience much change.

일부 값은 증가하고, 일부 값은 감소하며, 일부는 큰 변화를 경험하지 않습니다.

To see the behaviour of the function on a larger scale, we can apply the function to a subset of the complex plane and plot the result. To create our subset (or mesh), we can make use of the meshgrid function.

더 큰 규모로 함수의 동작을 보기 위해, 복소평면의 일부에 함수를 적용하고 결과를 그릴 수 있습니다. 우리의 부분집합(또는 메시)을 만들기 위해, meshgrid 함수를 활용할 수 있습니다.

x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
mesh = x + (1j * y)  # Make mesh of complex plane
Now we will apply our function to each value contained in the mesh. Since we used a universal function in our design, this means that we can pass in the entire mesh all at once. This is extremely convenient for two reasons: It reduces the amount of code needed to be written and greatly increases the efficiency (as universal functions make use of system level C programming in their computations).

이제 함수를 메시에 포함된 각 값에 적용할 것입니다. 설계에 범용 함수를 사용했기 때문에, 전체 메시를 한 번에 전달할 수 있습니다. 이는 두 가지 이유로 매우 편리합니다: 작성해야 할 코드의 양을 줄이고 효율성을 크게 향상시킵니다(범용 함수는 계산에 시스템 수준의 C 프로그래밍을 활용하기 때문).

Here we plot the absolute value (or modulus) of each element in the mesh after one “iteration” of the function using a 3D scatterplot:

여기서는 3D 산점도를 사용하여 함수의 한 번의 “반복” 후 메시의 각 요소의 절대값(또는 크기)을 그립니다:

output = np.abs(f(mesh))  # Take the absolute value of the output (for plotting)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(x, y, output, alpha=0.2)

ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_zlabel('Absolute value')
ax.set_title('One Iteration: $ f(z) = z^2 - 1$');
../_images/92e245b7fb0631b8c1bb99ab1b9106a6d65580dd552bed2fee014bfaa0c9b90b.png
This gives us a rough idea of what one iteration of the function does. Certain areas (notably in the areas closest to 
) remain rather small while other areas grow quite considerably. Note that we lose information about the output by taking the absolute value, but it is the only way for us to be able to make a plot.

이는 함수의 한 번의 반복이 어떤 작용을 하는지에 대한 대략적인 아이디어를 제공합니다. 특정 영역(특히 
에 가장 가까운 영역)은 상대적으로 작게 유지되지만 다른 영역은 상당히 크게 성장합니다. 절대값을 취함으로써 출력에 대한 정보를 잃게 되지만, 그래프를 만들 수 있는 유일한 방법입니다.

Let’s see what happens when we apply 2 iterations to the mesh:

메시에 2번의 반복을 적용하면 어떻게 되는지 살펴보겠습니다:

output = np.abs(f(f(mesh)))

ax = plt.axes(projection='3d')

ax.scatter(x, y, output, alpha=0.2)

ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')
ax.set_zlabel('Absolute value')
ax.set_title('Two Iterations: $ f(z) = z^2 - 1$');
../_images/10bfa793964b4939d8bd32f348fea349061d0444cdb5ad11b42d023afa95510d.png
Once again, we see that values around the origin remain small, and values with a larger absolute value (or modulus) “explode”.

다시 한번, 원점 주변의 값은 작게 유지되고, 더 큰 절대값(또는 크기)을 가진 값은 “폭발”하는 것을 볼 수 있습니다.

From first impression, its behaviour appears to be normal, and may even seem mundane. Fractals tend to have more to them then what meets the eye; the exotic behavior shows itself when we begin applying more iterations.

첫인상에서는 그 동작이 정상적으로 보이며, 심지어 평범해 보일 수도 있습니다. 프랙탈은 눈에 보이는 것보다 더 많은 것을 가지고 있습니다; 더 많은 반복을 적용하기 시작할 때 그 특이한 행동이 자신을 드러냅니다.

Consider three complex numbers:

세 개의 복소수를 고려해보세요:

,

,


Given the shape of our first two plots, we would expect that these values would remain near the origin as we apply iterations to them. Let us see what happens when we apply 10 iterations to each value:

첫 두 그래프의 모양을 고려할 때, 우리는 이 값들에 반복을 적용함에 따라 원점 근처에 머물 것으로 예상할 수 있습니다. 각 값에 10번의 반복을 적용할 때 어떤 일이 일어나는지 살펴보겠습니다:

selected_values = np.array([0.4 + 0.4j, 0.41 + 0.4j, 0.4 + 0.41j])
num_iter = 9

outputs = np.zeros((num_iter+1, selected_values.shape[0]), dtype=complex)
outputs[0] = selected_values

for i in range(num_iter):
    outputs[i+1] = f(outputs[i])  # Apply 10 iterations, save each output

fig, axes = plt.subplots(1, selected_values.shape[0], figsize=(16, 6))
axes[1].set_xlabel('Real axis')
axes[0].set_ylabel('Imaginary axis')

for ax, data in zip(axes, outputs.T):
    cycle = ax.scatter(data.real, data.imag, c=range(data.shape[0]), alpha=0.6)
    ax.set_title(f'Mapping of iterations on {data[0]}')

fig.colorbar(cycle, ax=axes, location="bottom", label='Iteration');
../_images/1e988574feb704f423fb6546206a6e0116b7bacc49e13a0b34a4644a9d48f2c4.png
To our surprise, the behaviour of the function did not come close to matching our hypothesis. This is a prime example of the chaotic behaviour fractals possess. In the first two plots, the value “exploded” on the last iteration, jumping way beyond the region that it was contained in previously. The third plot on the other hand remained bounded to a small region close to the origin, yielding completely different behaviour despite the tiny change in value.

놀랍게도, 함수의 동작은 우리의 가설과 거의 일치하지 않았습니다. 이는 프랙탈이 가진 혼돈스러운 행동의 대표적인 예입니다. 처음 두 그래프에서는 값이 마지막 반복에서 “폭발”하여 이전에 포함되어 있던 영역을 훨씬 넘어섰습니다. 반면 세 번째 그래프는 원점에 가까운 작은 영역에 제한되어 있어, 값의 작은 변화에도 불구하고 완전히 다른 행동을 보여주었습니다.

This leads us to an extremely important question: How many iterations can be applied to each value before they diverge (“explode”)?

이는 매우 중요한 질문으로 이어집니다: 값들이 발산(“폭발”)하기 전에 얼마나 많은 반복을 각 값에 적용할 수 있을까요?

As we saw from the first two plots, the further the values were from the origin, the faster they generally exploded. Although the behaviour is uncertain for smaller values (like 
), we can assume that if a value surpasses a certain distance from the origin (say 2) that it is doomed to diverge. We will call this threshold the radius.

첫 두 그래프에서 보았듯이, 값들이 원점에서 멀어질수록 일반적으로 더 빠르게 폭발했습니다. 더 작은 값(
과 같은)에 대해서는 행동이 불확실하지만, 값이 원점으로부터 특정 거리(예: 2)를 초과하면 발산할 운명이라고 가정할 수 있습니다. 우리는 이 임계값을 반경이라고 부를 것입니다.

This allows us to quantify the behaviour of the function for a particular value without having to perform as many computations. Once the radius is surpassed, we are allowed to stop iterating, which gives us a way of answering the question we posed. If we tally how many computations were applied before divergence, we gain insight into the behaviour of the function that would be hard to keep track of otherwise.

이는 많은 계산을 수행하지 않고도 특정 값에 대한 함수의 동작을 정량화할 수 있게 해줍니다. 반경이 초과되면 반복을 중지할 수 있으며, 이는 우리가 제기한 질문에 답하는 방법을 제공합니다. 발산 전에 적용된 계산의 수를 기록하면, 그렇지 않으면 추적하기 어려운 함수의 동작에 대한 통찰력을 얻을 수 있습니다.

Of course, we can do much better and design a function that performs the procedure on an entire mesh.

물론, 우리는 전체 메시에 대해 절차를 수행하는 함수를 더 잘 설계할 수 있습니다.

def divergence_rate(mesh, num_iter=10, radius=2):

    z = mesh.copy()
    diverge_len = np.zeros(mesh.shape)  # Keep tally of the number of iterations

    # Iterate on element if and only if |element| < radius (Otherwise assume divergence)
    for i in range(num_iter):
        conv_mask = np.abs(z) < radius
        diverge_len[conv_mask] += 1
        z[conv_mask] = f(z[conv_mask])

    return diverge_len
The behaviour of this function may look confusing at first glance, so it will help to explain some of the notation.

이 함수의 동작은 처음에는 혼란스러워 보일 수 있으므로, 일부 표기법을 설명하는 것이 도움이 될 것입니다.

Our goal is to iterate over each value in the mesh and to tally the number of iterations before the value diverges. Since some values will diverge quicker than others, we need a procedure that only iterates over values that have an absolute value that is sufficiently small enough. We also want to stop tallying values once they surpass the radius. For this, we can use Boolean Indexing, a NumPy feature that when paired with universal functions is unbeatable. Boolean Indexing allows for operations to be performed conditionally on a NumPy array without having to resort to looping over and checking for each array value individually.

우리의 목표는 메시의 각 값에 대해 반복하고 값이 발산하기 전의 반복 횟수를 기록하는 것입니다. 일부 값은 다른 값보다 더 빨리 발산할 수 있으므로, 충분히 작은 절대값을 가진 값에 대해서만 반복하는 절차가 필요합니다. 또한 값이 반경을 초과하면 기록을 중지하고 싶습니다. 이를 위해 불리언 인덱싱을 사용할 수 있습니다. 불리언 인덱싱은 범용 함수와 함께 사용될 때 타의 추종을 불허하는 NumPy 기능입니다. 불리언 인덱싱을 사용하면 각 배열 값을 개별적으로 반복하고 확인할 필요 없이 조건부로 NumPy 배열에 대한 작업을 수행할 수 있습니다.

In our case, we use a loop to apply iterations to our function 
 and keep tally. Using Boolean indexing, we only apply the iterations to values that have an absolute value less than 2.

우리의 경우, 함수 
에 반복을 적용하고 기록하기 위해 루프를 사용합니다. 불리언 인덱싱을 사용하여 절대값이 2보다 작은 값에만 반복을 적용합니다.

With that out of the way, we can go about plotting our first fractal! We will use the imshow function to create a colour-coded visualization of the tallies.

그것을 처리했으니, 우리의 첫 번째 프랙탈을 그릴 수 있습니다! 기록의 색상 코딩된 시각화를 만들기 위해 imshow 함수를 사용할 것입니다.

x, y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
mesh = x + (1j * y)

output = divergence_rate(mesh)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()

ax.set_title('$f(z) = z^2 -1$')
ax.set_xlabel('Real axis')
ax.set_ylabel('Imaginary axis')

im = ax.imshow(output, extent=[-2, 2, -2, 2])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax, label='Number of iterations');
../_images/1985ab37c2d2110f9b3e3d2ab367a1182dcb38c4d7ba74c4bf3c7b34f0124648.png
What this stunning visual conveys is the complexity of the function’s behaviour. The yellow region represents values that remain small, while the purple region represents the divergent values. The beautiful pattern that arises on the border of the converging and diverging values is even more fascinating when you realize that it is created from such a simple function.

이 멋진 시각적 이미지가 전달하는 것은 함수 동작의 복잡성입니다. 노란색 영역은 작게 유지되는 값을 나타내고, 보라색 영역은 발산하는 값을 나타냅니다. 수렴하는 값과 발산하는 값의 경계에서 발생하는 아름다운 패턴은 그것이 이렇게 간단한 함수에서 만들어졌다는 것을 깨달을 때 더욱 매혹적입니다.

## Julia set
## 줄리아 집합

What we just explored was an example of a fractal visualization of a specific Julia Set.

우리가 방금 탐색한 것은 특정 줄리아 집합의 프랙탈 시각화의 예였습니다.

Consider the function 
 where 
 is a complex number. The filled-in Julia set of 
 is the set of all complex numbers z in which the function converges at 
. Likewise, the boundary of the filled-in Julia set is what we call the Julia set. In our above visualization, we can see that the yellow region represents an approximation of the filled-in Julia set for 
 and the greenish-yellow border would contain the Julia set.

복소수인 
가 있는 함수 
를 고려해보세요. 
의 채워진 줄리아 집합은 함수가 
에서 수렴하는 모든 복소수 z의 집합입니다. 마찬가지로, 채워진 줄리아 집합의 경계가 우리가 줄리아 집합이라고 부르는 것입니다. 위의 시각화에서, 노란색 영역은 
에 대한 채워진 줄리아 집합의 근사치를 나타내고, 녹색-노란색 테두리가 줄리아 집합을 포함할 것입니다.

To gain access to a wider range of “Julia fractals”, we can write a function that allows for different values of 
 to be passed in:

더 넓은 범위의 “줄리아 프랙탈”에 접근하기 위해, 다른 값의 
을 전달할 수 있는 함수를 작성할 수 있습니다:

def julia(mesh, c=-1, num_iter=10, radius=2):

    z = mesh.copy()
    diverge_len = np.zeros(z.shape)

    for i in range(num_iter):
        conv_mask = np.abs(z) < radius
        z[conv_mask] = np.square(z[conv_mask]) + c
        diverge_len[conv_mask] += 1

    return diverge_len
To make our lives easier, we will create a couple meshes that we will reuse throughout the rest of the examples:

우리의 삶을 더 쉽게 만들기 위해, 남은 예제들에서 재사용할 몇 가지 메시를 만들겠습니다:

x, y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
small_mesh = x + (1j * y)

x, y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
mesh = x + (1j * y)
We will also write a function that we will use to create our fractal plots:

또한 프랙탈 그래프를 만들기 위해 사용할 함수를 작성할 것입니다:

def plot_fractal(fractal, title='Fractal', figsize=(6, 6), cmap='rainbow', extent=[-2, 2, -2, 2]):

    plt.figure(figsize=figsize)
    ax = plt.axes()

    ax.set_title(f'${title}$')
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imaginary axis')

    im = ax.imshow(fractal, extent=extent, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Number of iterations')
Using our newly defined functions, we can make a quick plot of the first fractal again:

새로 정의한 함수를 사용하여 첫 번째 프랙탈을 다시 빠르게 그릴 수 있습니다:

output = julia(mesh, num_iter=15)
kwargs = {'title': 'f(z) = z^2 -1'}

plot_fractal(output, **kwargs);
../_images/bb3abae2b95f4a184e3b54453d60942b4d7dd389d00464231deb0480f2ee7667.png
We also can explore some different Julia sets by experimenting with different values of 
. It can be surprising how much influence it has on the shape of the fractal.

또한 다른 값의 
로 실험함으로써 다른 줄리아 집합을 탐색할 수 있습니다. 그것이 프랙탈의 모양에 얼마나 많은 영향을 미치는지는 놀라울 수 있습니다.

For example, setting 
 
 gives us a very elegant cloud shape, while setting c = 
 
 yields a completely different pattern.

예를 들어, 
 
를 설정하면 매우 우아한 구름 모양이 나오는 반면, c = 
 
를 설정하면 완전히 다른 패턴이 나옵니다.

output = julia(mesh, c=np.pi/10, num_iter=20)
kwargs = {'title': r'f(z) = z^2 + \dfrac{\pi}{10}', 'cmap': 'plasma'}

plot_fractal(output, **kwargs);
../_images/e13f833d75df16d4ed598647e031a6bb22bae1cde43ef69b076a2fa0da244e72.png
output = julia(mesh, c=-0.75 + 0.4j, num_iter=20)
kwargs = {'title': r'f(z) = z^2 - \dfrac{3}{4} + 0.4i', 'cmap': 'Greens_r'}

plot_fractal(output, **kwargs);
../_images/f303f91b122ec2c2e1dbbdddef130e472e741ea90f93fcdfeb20083189b7c637.png
## Mandelbrot set
## 만델브로트 집합

Closely related to the Julia set is the famous Mandelbrot set, which has a slightly different definition. Once again, we define 
 where 
 is a complex number, but this time our focus is on our choice of 
. We say that 
 is an element of the Mandelbrot set if f converges at 
. An equivalent definition is to say that 
 is an element of the Mandelbrot set if 
 can be iterated infinitely and not ‘explode’. We will tweak our Julia function slightly (and rename it appropriately) so that we can plot a visualization of the Mandelbrot set, which possesses an elegant fractal pattern.

줄리아 집합과 밀접하게 관련된 것은 유명한 만델브로트 집합으로, 약간 다른 정의를 가지고 있습니다. 다시 한번, 복소수인 
가 있는 
를 정의하지만, 이번에는 우리의 초점은 
의 선택에 있습니다. 
가 만델브로트 집합의 요소라고 하면 f가 
에서 수렴합니다. 동등한 정의는 
가 무한히 반복될 수 있고 ‘폭발’하지 않는다면 
는 만델브로트 집합의 요소라고 말하는 것입니다. 우리는 줄리아 함수를 약간 수정하고(적절히 이름을 바꿔) 우아한 프랙탈 패턴을 가진 만델브로트 집합의 시각화를 그릴 수 있도록 할 것입니다.

def mandelbrot(mesh, num_iter=10, radius=2):

    c = mesh.copy()
    z = np.zeros(mesh.shape, dtype=np.complex128)
    diverge_len = np.zeros(z.shape)

    for i in range(num_iter):
        conv_mask = np.abs(z) < radius
        z[conv_mask] = np.square(z[conv_mask]) + c[conv_mask]
        diverge_len[conv_mask] += 1

    return diverge_len
output = mandelbrot(mesh, num_iter=50)
kwargs = {'title': 'Mandelbrot \\ set', 'cmap': 'hot'}

plot_fractal(output, **kwargs);
../_images/5b82b597f66a8218fee5c1ca7e7d7d99ecba9b2a549f3419c1901a65c48408e3.png
## Generalizing the Julia set
## 줄리아 집합 일반화하기

We can generalize our Julia function even further by giving it a parameter for which universal function we would like to pass in. This would allow us to plot fractals of the form 
 where g is a universal function selected by us.

우리가 전달하고 싶은 범용 함수를 위한 매개변수를 줌으로써 줄리아 함수를 더욱 일반화할 수 있습니다. 이는 g가 우리가 선택한 범용 함수인 
 형태의 프랙탈을 그릴 수 있게 해줍니다.

def general_julia(mesh, c=-1, f=np.square, num_iter=100, radius=2):

    z = mesh.copy()
    diverge_len = np.zeros(z.shape)

    for i in range(num_iter):
        conv_mask = np.abs(z) < radius
        z[conv_mask] = f(z[conv_mask]) + c
        diverge_len[conv_mask] += 1

    return diverge_len
One cool set of fractals that can be plotted using our general Julia function are ones of the form 
 for some positive integer 
. A very cool pattern which emerges is that the number of regions that ‘stick out’ matches the degree in which we raise the function to while iterating over it.

우리의 일반적인 줄리아 함수를 사용하여 그릴 수 있는 멋진 프랙탈 집합 중 하나는 양의 정수 
에 대해 
 형태의 것들입니다. 매우 멋진 패턴이 나타나는데, ‘튀어나오는’ 영역의 수가 함수를 반복하면서 올리는 정도와 일치한다는 것입니다.

fig, axes = plt.subplots(2, 3, figsize=(8, 8))
base_degree = 2

for deg, ax in enumerate(axes.ravel()):
    degree = base_degree + deg
    power = lambda z: np.power(z, degree)  # Create power function for current degree

    diverge_len = general_julia(mesh, f=power, num_iter=15)
    ax.imshow(diverge_len, extent=[-2, 2, -2, 2], cmap='binary')
    ax.set_title(f'$f(z) = z^{degree} -1$')
../_images/9783ff4a1ba17ccfbea798e94462090de1e9530b63ae9ffab6a1a63dfa9f2d36.png
Needless to say, there is a large amount of exploring that can be done by fiddling with the inputted function, value of 
, number of iterations, radius and even the density of the mesh and choice of colours.

말할 필요도 없이, 입력된 함수, 
의 값, 반복 횟수, 반경, 심지어 메시의 밀도와 색상 선택을 만지작거리며 많은 탐색을 할 수 있습니다.

## Newton Fractals
## 뉴턴 프랙탈

Newton fractals are a specific class of fractals, where iterations involve adding or subtracting the ratio of a function (often a polynomial) and its derivative to the input values. Mathematically, it can be expressed as:

뉴턴 프랙탈은 특정 종류의 프랙탈로, 반복이 함수(종종 다항식)와 그 미분의 비율을 입력 값에 더하거나 빼는 것을 포함합니다. 수학적으로, 이는 다음과 같이 표현될 수 있습니다:

 

We will define a general version of the fractal which will allow for different variations to be plotted by passing in our functions of choice.

우리는 선택한 함수를 전달하여 다양한 변형을 그릴 수 있는 프랙탈의 일반 버전을 정의할 것입니다.

def newton_fractal(mesh, f, df, num_iter=10, r=2):

    z = mesh.copy()
    diverge_len = np.zeros(z.shape)

    for i in range(num_iter):
        conv_mask = np.abs(z) < r
        pz = f(z[conv_mask])
        dp = df(z[conv_mask])
        z[conv_mask] = z[conv_mask] - pz/dp
        diverge_len[conv_mask] += 1

    return diverge_len
Now we can experiment with some different functions. For polynomials, we can create our plots quite effortlessly using the NumPy Polynomial class, which has built in functionality for computing derivatives.

이제 몇 가지 다른 함수들로 실험할 수 있습니다. 다항식의 경우, 미분을 계산하는 내장 기능이 있는 NumPy Polynomial 클래스를 사용하여 매우 쉽게 그래프를 만들 수 있습니다.

For example, let’s try a higher-degree polynomial:

예를 들어, 고차 다항식을 시도해 봅시다:

p = np.polynomial.Polynomial([-16, 0, 0, 0, 15, 0, 0, 0, 1])
p
which has the derivative:

이것의 미분은 다음과 같습니다:

p.deriv()
output = newton_fractal(mesh, p, p.deriv(), num_iter=15, r=2)
kwargs = {'title': r'f(z) = z - \dfrac{(z^8 + 15z^4 - 16)}{(8z^7 + 60z^3)}', 'cmap': 'copper'}

plot_fractal(output, **kwargs)
../_images/6d3cc8af070e225005626059c66ab70288f7afb6c6a31b3c9d80ef9a1cf231a1.png
Beautiful! Let’s try another one:

아름답습니다! 또 다른 것을 시도해 봅시다:

f(z) = 

 
 

This makes 
 
 
 
 

f(z) = 

 
 

이는 
 
 
 
을 만듭니다 

def f_tan(z):
    return np.square(np.tan(z))


def d_tan(z):
    return 2*np.tan(z) / np.square(np.cos(z))
output = newton_fractal(mesh, f_tan, d_tan, num_iter=15, r=50)
kwargs = {'title': r'f(z) = z - \dfrac{sin(z)cos(z)}{2}', 'cmap': 'binary'}

plot_fractal(output, **kwargs);
../_images/e55fe4cf039bba747d28687bbd8e84870cc68c39dde8783dd468d0493a872126.png
Note that you sometimes have to play with the radius in order to get a neat looking fractal.

깔끔하게 보이는 프랙탈을 얻기 위해 때로는 반경을 조정해야 한다는 점에 유의하세요.

Finally, we can go a little bit wild with our function selection

마지막으로, 함수 선택에 있어 약간 과감해질 수 있습니다


 

def sin_sum(z, n=10):
    total = np.zeros(z.size, dtype=z.dtype)
    for i in range(1, n+1):
        total += np.power(np.sin(z), i)
    return total


def d_sin_sum(z, n=10):
    total = np.zeros(z.size, dtype=z.dtype)
    for i in range(1, n+1):
        total += i * np.power(np.sin(z), i-1) * np.cos(z)
    return total
We will denote this one ‘Wacky fractal’, as its equation would not be fun to try and put in the title.

이 것은 방정식을 제목에 넣으려고 시도하는 것이 재미없을 것이기 때문에 ‘괴상한 프랙탈’이라고 표시하겠습니다.

output = newton_fractal(small_mesh, sin_sum, d_sin_sum, num_iter=10, r=1)
kwargs = {'title': 'Wacky \\ fractal', 'figsize': (6, 6), 'extent': [-1, 1, -1, 1], 'cmap': 'terrain'}

plot_fractal(output, **kwargs)
../_images/b6b6af119cf94b4dc5c3cbfbf70a589e79ae9139f0dee69a5080fee41b078b37.png
It is truly fascinating how distinct yet similar these fractals are with each other. This leads us to the final section.

이러한 프랙탈들이 서로 얼마나 뚜렷하면서도 유사한지는 정말 매혹적입니다. 이는 우리를 마지막 섹션으로 이끕니다.

## Creating your own fractals
## 자신만의 프랙탈 만들기

What makes fractals more exciting is how much there is to explore once you become familiar with the basics. Now we will wrap up our tutorial by exploring some of the different ways one can experiment in creating unique fractals. I encourage you to try some things out on your own (if you have not done so already).

프랙탈을 더 흥미롭게 만드는 것은 기본 사항에 익숙해지면 얼마나 많은 것을 탐색할 수 있는지입니다. 이제 고유한 프랙탈을 만드는 데 실험할 수 있는 다양한 방법을 탐색하여 튜토리얼을 마무리하겠습니다. 직접 몇 가지를 시도해 보시기 바랍니다(아직 하지 않았다면).

One of the first places to experiment would be with the function for the generalized Julia set, where we can try passing in different functions as parameters.

실험할 첫 번째 장소 중 하나는 다른 함수를 매개변수로 전달해 볼 수 있는 일반화된 줄리아 집합을 위한 함수일 것입니다.

Let’s start by choosing


시작하기 위해 


를 선택해 봅시다

def f(z):
    return np.tan(np.square(z))
output = general_julia(mesh, f=f, num_iter=15, radius=2.1)
kwargs = {'title': 'f(z) = tan(z^2)', 'cmap': 'gist_stern'}

plot_fractal(output, **kwargs);
../_images/5d2d765af10ad864e9ba9b73d2c83c69613fbd427754f7c3ea4ea73f324df2ec.png
What happens if we compose our defined function inside of a sine function?

정의한 함수를 사인 함수 안에 합성하면 어떻게 될까요?

Let’s try defining


다음을 정의해 봅시다


def g(z):
    return np.sin(f(z))
output = general_julia(mesh, f=g, num_iter=15, radius=2.1)
kwargs = {'title': 'g(z) = sin(tan(z^2))', 'cmap': 'plasma_r'}

plot_fractal(output, **kwargs);
../_images/60b9eb4486ee0b010e6537b3e1d8524cc6e24e37a62eb989bc538f6db2a0e29e.png
Next, let’s create a function that applies both f and g to the inputs each iteration and adds the result together:

다음으로, 각 반복마다 입력에 f와 g를 모두 적용하고 결과를 함께 더하는 함수를 만들어 봅시다:


def h(z):
    return f(z) + g(z)
output = general_julia(small_mesh, f=h, num_iter=10, radius=2.1)
kwargs = {'title': 'h(z) = tan(z^2) + sin(tan(z^2))', 'figsize': (7, 7), 'extent': [-1, 1, -1, 1], 'cmap': 'jet'}

plot_fractal(output, **kwargs);
../_images/f2fd6629436adc7841b7e9ff1bf3480cc2cd391fcc64c47897b7114cacf18576.png
You can even create beautiful fractals through your own errors. Here is one that got created accidently by making a mistake in computing the derivative of a Newton fractal:

자신의 실수를 통해 아름다운 프랙탈을 만들 수도 있습니다. 여기 뉴턴 프랙탈의 미분을 계산하는 데 실수를 하여 우연히 만들어진 것이 있습니다:

def accident(z):
    return z - (2 * np.power(np.tan(z), 2) / (np.sin(z) * np.cos(z)))
output = general_julia(mesh, f=accident, num_iter=15, c=0, radius=np.pi)
kwargs = {'title': 'Accidental \\ fractal', 'cmap': 'Blues'}

plot_fractal(output, **kwargs);
../_images/c73f025c41c3005dac1ef073c6ef0a228ffc0a598dcedc853ce1cbdaac5cd9a5.png
Needless to say, there are a nearly endless supply of interesting fractal creations that can be made just by playing around with various combinations of NumPy universal functions and by tinkering with the parameters.

말할 필요도 없이, NumPy 범용 함수의 다양한 조합으로 놀고 매개변수를 조정하는 것만으로도 거의 무한한 공급의 흥미로운 프랙탈 창작물을 만들 수 있습니다.

## In conclusion
## 결론

We learned a lot about generating fractals today. We saw how complicated fractals requiring many iterations could be computed efficiently using universal functions. We also took advantage of boolean indexing, which allowed for less computations to be made without having to individually verify each value. Finally, we learned a lot about fractals themselves. As a recap:

오늘 우리는 프랙탈 생성에 대해 많이 배웠습니다. 범용 함수를 사용하여 많은 반복이 필요한 복잡한 프랙탈을 효율적으로 계산하는 방법을 보았습니다. 또한 각 값을 개별적으로 확인하지 않고도 계산을 줄일 수 있는 불리언 인덱싱을 활용했습니다. 마지막으로, 우리는 프랙탈 자체에 대해 많이 배웠습니다. 요약하자면:

Fractal images are created by iterating a function over a set of values, and keeping tally of how long it takes for each value to pass a certain threshold

The colours in the image correspond to the tally counts of the values

The filled-in Julia set for 
 consists of all complex numbers z in which 
 converges

The Julia set for 
 is the set of complex numbers that make up the boundary of the filled-in Julia set

The Mandelbrot set is all values 
 in which 
 converges at 0

Newton fractals use functions of the form 
 

The fractal images can vary as you adjust the number of iterations, radius of convergence, mesh size, colours, function choice and parameter choice

프랙탈 이미지는 일련의 값에 대해 함수를 반복하고, 각 값이 특정 임계값을 통과하는 데 걸리는 시간을 기록함으로써 생성됩니다

이미지의 색상은 값의 기록 횟수에 해당합니다


에 대한 채워진 줄리아 집합은 
가 수렴하는 모든 복소수 z로 구성됩니다


에 대한 줄리아 집합은 채워진 줄리아 집합의 경계를 구성하는 복소수 집합입니다

만델브로트 집합은 
가 0에서 수렴하는 모든 값 
입니다

뉴턴 프랙탈은 
 형태의 함수를 사용합니다

프랙탈 이미지는 반복 횟수, 수렴 반경, 메시 크기, 색상, 함수 선택 및 매개변수 선택을 조정함에 따라 달라질 수 있습니다

## On your own
## 스스로 해보기

Play around with the parameters of the generalized Julia set function, try playing with the constant value, number of iterations, function choice, radius, and colour choice.

일반화된 줄리아 집합 함수의 매개변수를 가지고 놀아보세요, 상수 값, 반복 횟수, 함수 선택, 반경, 그리고 색상 선택을 시도해보세요.

Visit the “List of fractals by Hausdorff dimension” Wikipedia page (linked in the Further reading section) and try writing a function for a fractal not mentioned in this tutorial.

“하우스도르프 차원별 프랙탈 목록” 위키피디아 페이지(추가 읽기 섹션에 링크됨)를 방문하여 이 튜토리얼에서 언급되지 않은 프랙탈에 대한 함수를 작성해보세요.