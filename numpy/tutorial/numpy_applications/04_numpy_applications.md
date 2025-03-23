# Determining Static Equilibrium in NumPy
# NumPy로 정적 평형 확인하기

When analyzing physical structures, it is crucial to understand the mechanics keeping them stable. Applied forces on a floor, a beam, or any other structure, create reaction forces and moments. These reactions are the structure resisting movement without breaking. In cases where structures do not move despite having forces applied to them, [Newton's second law](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton's_second_law) states that both the acceleration and sum of forces in all directions in the system must be zero. You can represent and solve this concept with NumPy arrays.

물리적 구조물을 분석할 때, 그것들을 안정적으로 유지하는 역학을 이해하는 것이 중요합니다. 바닥, 빔 또는 다른 구조물에 가해진 힘은 반작용력과 모멘트를 생성합니다. 이러한 반작용은 구조물이 파손되지 않고 움직임에 저항하는 것입니다. 힘이 가해졌음에도 구조물이 움직이지 않는 경우, [뉴턴의 제2법칙](https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion#Newton's_second_law)에 따르면 시스템의 모든 방향에서 가속도와 힘의 합은 0이어야 합니다. 이 개념을 NumPy 배열로 표현하고 해결할 수 있습니다.

## What you'll do:
## 무엇을 할 것인가:
- In this tutorial, you will use NumPy to create vectors and moments using NumPy arrays
- Solve problems involving cables and floors holding up structures
- Write NumPy matrices to isolate unkowns
- Use NumPy functions to perform linear algebra operations

- 이 튜토리얼에서는 NumPy 배열을 사용하여 벡터와 모멘트를 생성합니다
- 케이블과 바닥이 구조물을 지지하는 문제를 해결합니다
- 미지수를 분리하기 위해 NumPy 행렬을 작성합니다
- 선형 대수 연산을 수행하기 위해 NumPy 함수를 사용합니다

## What you'll learn:
## 배우게 될 것:
- How to represent points, vectors, and moments with NumPy.
- How to find the [normal of vectors](https://en.wikipedia.org/wiki/Normal_(geometry))
- Using NumPy to compute matrix calculations

- NumPy로 점, 벡터 및 모멘트를 표현하는 방법
- [벡터의 법선](https://en.wikipedia.org/wiki/Normal_(geometry))을 찾는 방법
- NumPy를 사용하여 행렬 계산하는 방법

## What you'll need:
## 필요한 것:
- NumPy
- [Matplotlib](https://matplotlib.org/)

imported with the following comands:
다음 명령어로 가져옵니다:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
```

In this tutorial you will use the following NumPy tools:
이 튜토리얼에서는 다음 NumPy 도구를 사용합니다:

* [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) : this function determines the measure of vector magnitude
* [`np.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html) : this function takes two matrices and produces the cross product

* [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) : 이 함수는 벡터 크기를 측정합니다
* [`np.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html) : 이 함수는 두 행렬을 받아 외적을 계산합니다

+++

## Solving equilibrium with Newton's second law
## 뉴턴의 제2법칙으로 평형 해결하기

Your model consists of a beam under a sum of forces and moments. You can start analyzing this system with Newton's second law:

당신의 모델은 힘과 모멘트의 합을 받는 빔으로 구성되어 있습니다. 뉴턴의 제2법칙으로 이 시스템을 분석하기 시작할 수 있습니다:

$$\sum{\text{force}} = \text{mass} \times \text{acceleration}.$$

In order to simplify the examples looked at, assume they are static, with acceleration $=0$. Due to our system existing in three dimensions, consider forces being applied in each of these dimensions. This means that you can represent these forces as vectors. You come to the same conclusion for [moments](https://en.wikipedia.org/wiki/Moment_(physics)), which result from forces being applied a certain distance away from an object's center of mass.

살펴볼 예제를 단순화하기 위해, 가속도 $=0$인 정적 상태라고 가정합니다. 우리 시스템이 3차원에 존재하기 때문에, 각 차원에 가해지는 힘을 고려합니다. 이는 이러한 힘을 벡터로 표현할 수 있다는 의미입니다. [모멘트](https://en.wikipedia.org/wiki/Moment_(physics))에 대해서도 같은 결론에 도달합니다. 모멘트는 물체의 질량 중심에서 일정 거리 떨어진 곳에 힘이 가해질 때 발생합니다.

Assume that the force $F$ is represented as a three-dimensional vector

힘 $F$가 3차원 벡터로 표현된다고 가정합니다

$$F = (F_x, F_y, F_z)$$

where each of the three components represent the magnitude of the force being applied in each corresponding direction. Assume also that each component in the vector

여기서 세 구성 요소는 각 해당 방향으로 적용되는 힘의 크기를 나타냅니다. 또한 벡터의 각 구성 요소

$$r = (r_x, r_y, r_z)$$

is the distance between the point where each component of the force is applied and the centroid of the system. Then, the moment can be computed by

는 힘의 각 구성 요소가 적용되는 지점과 시스템의 중심 사이의 거리입니다. 그러면 모멘트는 다음과 같이 계산할 수 있습니다

$$r \times F = (r_x, r_y, r_z) \times (F_x, F_y, F_z).$$

Start with some simple examples of force vectors

간단한 힘 벡터의 예를 시작해 봅시다

```{code-cell}
forceA = np.array([1, 0, 0])
forceB = np.array([0, 1, 0])
print("Force A =", forceA)
print("Force B =", forceB)
```

This defines `forceA` as being a vector with magnitude of 1 in the $x$ direction and `forceB` as magnitude 1 in the $y$ direction.

이는 `forceA`를 $x$ 방향으로 크기가 1인 벡터로, `forceB`를 $y$ 방향으로 크기가 1인 벡터로 정의합니다.

It may be helpful to visualize these forces in order to better understand how they interact with each other.
Matplotlib is a library with visualization tools that can be utilized for this purpose.
Quiver plots will be used to demonstrate [three dimensional vectors](https://matplotlib.org/3.3.4/gallery/mplot3d/quiver3d.html), but the library can also be used for [two dimensional demonstrations](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html).

이러한 힘이 서로 어떻게 상호작용하는지 더 잘 이해하기 위해 시각화하는 것이 도움이 될 수 있습니다.
Matplotlib은 이 목적을 위해 활용할 수 있는 시각화 도구가 있는 라이브러리입니다.
[3차원 벡터](https://matplotlib.org/3.3.4/gallery/mplot3d/quiver3d.html)를 보여주기 위해 quiver 플롯이 사용되지만, 이 라이브러리는 [2차원 시연](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html)에도 사용할 수 있습니다.

```{code-cell}
fig = plt.figure()

d3 = fig.add_subplot(projection="3d")

d3.set_xlim(-1, 1)
d3.set_ylim(-1, 1)
d3.set_zlim(-1, 1)

x, y, z = np.array([0, 0, 0])  # defining the point of application.  Make it the origin

u, v, w = forceA  # breaking the force vector into individual components
d3.quiver(x, y, z, u, v, w, color="r", label="forceA")

u, v, w = forceB
d3.quiver(x, y, z, u, v, w, color="b", label="forceB")

plt.legend()
plt.show()
```

There are two forces emanating from a single point. In order to simplify this problem, you can add them together to find the sum of forces. Note that both `forceA` and `forceB` are three-dimensional vectors, represented by NumPy as arrays with three components. Because NumPy is meant to simplify and optimize operations between vectors, you can easily compute the sum of these two vectors as follows:

단일 지점에서 나오는 두 개의 힘이 있습니다. 이 문제를 단순화하기 위해 이들을 합하여 힘의 합을 찾을 수 있습니다. `forceA`와 `forceB`는 모두 NumPy에서 세 개의 구성 요소를 가진 배열로 표현되는 3차원 벡터임을 주목하세요. NumPy는 벡터 간의 연산을 단순화하고 최적화하기 위한 것이므로, 다음과 같이 이 두 벡터의 합을 쉽게 계산할 수 있습니다:

```{code-cell}
forceC = forceA + forceB
print("Force C =", forceC)
```

Force C now acts as a single force that represents both A and B.
You can plot it to see the result.

힘 C는 이제 A와 B 모두를 나타내는 단일 힘으로 작용합니다.
결과를 보기 위해 그래프로 표현할 수 있습니다.

```{code-cell}
fig = plt.figure()

d3 = fig.add_subplot(projection="3d")

d3.set_xlim(-1, 1)
d3.set_ylim(-1, 1)
d3.set_zlim(-1, 1)

x, y, z = np.array([0, 0, 0])

u, v, w = forceA
d3.quiver(x, y, z, u, v, w, color="r", label="forceA")
u, v, w = forceB
d3.quiver(x, y, z, u, v, w, color="b", label="forceB")
u, v, w = forceC
d3.quiver(x, y, z, u, v, w, color="g", label="forceC")

plt.legend()
plt.show()
```

However, the goal is equilibrium.
This means that you want your sum of forces to be $(0, 0, 0)$ or else your object will experience acceleration.
Therefore, there needs to be another force that counteracts the prior ones.

그러나 목표는 평형 상태입니다.
이는 힘의 합이 $(0, 0, 0)$이어야 한다는 의미이며, 그렇지 않으면 물체가 가속도를 경험하게 됩니다.
따라서 이전의 힘들을 상쇄하는 또 다른 힘이 필요합니다.

You can write this problem as $A+B+R=0$, with $R$ being the reaction force that solves the problem.

이 문제를 $A+B+R=0$으로 쓸 수 있으며, 여기서 $R$은 문제를 해결하는 반작용력입니다.

In this example this would mean:

이 예제에서 이것은 다음을 의미합니다:

$$(1, 0, 0) + (0, 1, 0) + (R_x, R_y, R_z) = (0, 0, 0)$$

Broken into $x$, $y$, and $z$ components this gives you:

$x$, $y$, $z$ 성분으로 나누면 다음과 같습니다:

$$\begin{cases}
1+0+R_x=0\\
0+1+R_y=0\\
0+0+R_z=0
\end{cases}$$

solving for $R_x$, $R_y$, and $R_z$ gives you a vector $R$ of $(-1, -1, 0)$.

$R_x$, $R_y$ 및 $R_z$에 대해 풀면 벡터 $R$은 $(-1, -1, 0)$이 됩니다.


If plotted, the forces seen in prior examples should be nullified.
Only if there is no force remaining is the system considered to be in equilibrium.

그래프로 표현하면, 이전 예제에서 본 힘들은 무효화되어야 합니다.
남아있는 힘이 없는 경우에만 시스템이 평형 상태에 있다고 간주됩니다.

```{code-cell}
R = np.array([-1, -1, 0])

fig = plt.figure()

d3.set_xlim(-1, 1)
d3.set_ylim(-1, 1)
d3.set_zlim(-1, 1)

d3 = fig.add_subplot(projection="3d")

x, y, z = np.array([0, 0, 0])

u, v, w = forceA + forceB + R  # add them all together for sum of forces
d3.quiver(x, y, z, u, v, w)

plt.show()
```

The empty graph signifies that there are no outlying forces. This denotes a system in equilibrium.

빈 그래프는 외부 힘이 없음을 의미합니다. 이는 시스템이 평형 상태에 있음을 나타냅니다.


## Solving Equilibrium as a sum of moments
## 모멘트의 합으로 평형 해결하기

Next let's move to a more complicated application.
When forces are not all applied at the same point, moments are created.

다음으로 더 복잡한 응용으로 넘어가 보겠습니다.
모든 힘이 같은 지점에 적용되지 않을 때 모멘트가 생성됩니다.

Similar to forces, these moments must all sum to zero, otherwise rotational acceleration will be experienced.  Similar to the sum of forces, this creates a linear equation for each of the three coordinate directions in space.

힘과 마찬가지로, 이러한 모멘트들은 모두 합이 0이어야 하며, 그렇지 않으면 회전 가속도가 발생합니다. 힘의 합과 마찬가지로, 이는 공간의 세 좌표 방향 각각에 대한 선형 방정식을 생성합니다.

A simple example of this would be from a force applied to a stationary pole secured in the ground.
The pole does not move, so it must apply a reaction force.
The pole also does not rotate, so it must also be creating a reaction moment.
Solve for both the reaction force and moments.

이에 대한 간단한 예는 지면에 고정된 고정 기둥에 힘이 가해지는 경우입니다.
기둥이 움직이지 않으므로 반작용력을 적용해야 합니다.
기둥도 회전하지 않으므로 반작용 모멘트도 생성해야 합니다.
반작용력과 모멘트를 모두 풀어야 합니다.

Lets say a 5N force is applied perpendicularly 2m above the base of the pole.

5N의 힘이 기둥 기저부에서 2m 위에 수직으로 적용된다고 가정해 봅시다.

```{code-cell}
f = 5  # Force in newtons
L = 2  # Length of the pole

R = 0 - f
M = 0 - f * L
print("Reaction force =", R)
print("Reaction moment =", M)
```

## Finding values with physical properties
## 물리적 속성으로 값 찾기

Let's say that instead of a force acting perpendicularly to the beam, a force was applied to our pole through a wire that was also attached to the ground.
Given the tension in this cord, all you need to solve this problem are the physical locations of these objects.

빔에 수직으로 작용하는 힘 대신, 지면에도 부착된 와이어를 통해 우리 기둥에 힘이 가해졌다고 가정해 봅시다.
이 코드의 장력이 주어지면, 이 문제를 해결하기 위해 필요한 것은 이러한 물체들의 물리적 위치뿐입니다.

![Image representing the problem](_static/static_eqbm-fig01.png)

In response to the forces acting upon the pole, the base generated reaction forces in the x and y directions, as well as a reaction moment.

기둥에 작용하는 힘에 대한 반응으로, 기저부는 x와 y 방향으로 반작용력과 반작용 모멘트를 생성했습니다.

Denote the base of the pole as the origin.
Now, say the cord is attached to the ground 3m in the x direction and attached to the pole 2m up, in the z direction.

기둥의 기저부를 원점으로 표시합니다.
이제 코드가 x 방향으로 3m 떨어진 지면과 z 방향으로 2m 위의 기둥에 부착되어 있다고 가정합니다.

Define these points in space as NumPy arrays, and then use those arrays to find directional vectors.

이러한 공간의 점들을 NumPy 배열로 정의한 다음, 그 배열을 사용하여 방향 벡터를 찾습니다.

```{code-cell}
poleBase = np.array([0, 0, 0])
cordBase = np.array([3, 0, 0])
cordConnection = np.array([0, 0, 2])

poleDirection = cordConnection - poleBase
print("Pole direction =", poleDirection)
cordDirection = cordBase - cordConnection
print("Cord direction =", cordDirection)
```

In order to use these vectors in relation to forces you need to convert them into unit vectors.
Unit vectors have a magnitude of one, and convey only the direction of the forces.

이러한 벡터를 힘과 관련하여 사용하려면 단위 벡터로 변환해야 합니다.
단위 벡터는 크기가 1이며, 힘의 방향만을 전달합니다.

```{code-cell}
cordUnit = cordDirection / np.linalg.norm(cordDirection)
print("Cord unit vector =", cordUnit)
```

You can then multiply this direction with the magnitude of the force in order to find the force vector.

그런 다음 이 방향에 힘의 크기를 곱하여 힘 벡터를 찾을 수 있습니다.

Let's say the cord has a tension of 5N:

코드의 장력이 5N이라고 가정해 봅시다:

```{code-cell}
cordTension = 5
forceCord = cordUnit * cordTension
print("Force from the cord =", forceCord)
```

In order to find the moment you need the cross product of the force vector and the radius.

모멘트를 찾기 위해서는 힘 벡터와 반경의 외적이 필요합니다.

```{code-cell}
momentCord = np.cross(forceCord, poleDirection)
print("Moment from the cord =", momentCord)
```

Now all you need to do is find the reaction force and moment.

이제 반작용력과 모멘트를 찾기만 하면 됩니다.

```{code-cell}
equilibrium = np.array([0, 0, 0])
R = equilibrium - forceCord
M = equilibrium - momentCord
print("Reaction force =", R)
print("Reaction moment =", M)
```

### Another Example
### 또 다른 예제
Let's look at a slightly more complicated model.  In this example you will be observing a beam with two cables and an applied force.  This time you need to find both the tension in the cords and the reaction forces of the beam. *(Source: [Vector Mechanics for Engineers: Statics and Dynamics](https://www.mheducation.com/highered/product/Vector-Mechanics-for-Engineers-Statics-and-Dynamics-Beer.html), Problem 4.106)*

조금 더 복잡한 모델을 살펴보겠습니다. 이 예제에서는 두 개의 케이블과 적용된 힘이 있는 빔을 관찰할 것입니다. 이번에는 코드의 장력과 빔의 반작용력을 모두 찾아야 합니다. *(출처: [엔지니어를 위한 벡터 역학: 정역학 및 동역학](https://www.mheducation.com/highered/product/Vector-Mechanics-for-Engineers-Statics-and-Dynamics-Beer.html), 문제 4.106)*


![image.png](_static/problem4.png)

Define distance *a* as 3 meters

거리 *a*를 3미터로 정의합니다


As before, start by defining the location of each relevant point as an array.

앞서와 같이, 각 관련 점의 위치를 배열로 정의하는 것으로 시작합니다.

```{code-cell}
A = np.array([0, 0, 0])
B = np.array([0, 3, 0])
C = np.array([0, 6, 0])
D = np.array([1.5, 0, -3])
E = np.array([1.5, 0, 3])
F = np.array([-3, 0, 2])
```

From these equations, you start by determining vector directions with unit vectors.

이러한 방정식으로부터, 단위 벡터로 벡터 방향을 결정하는 것부터 시작합니다.

```{code-cell}
AB = B - C
AC = C - A
BD = D - B
BE = E - B
CF = F - C

UnitBD = BD / np.linalg.norm(BD)
UnitBE = BE / np.linalg.norm(BE)
UnitCF = CF / np.linalg.norm(CF)

RadBD = np.cross(AB, UnitBD)
RadBE = np.cross(AB, UnitBE)
RadCF = np.cross(AC, UnitCF)
```

This lets you represent the tension (T) and reaction (R) forces acting on the system as

이를 통해 시스템에 작용하는 장력(T)과 반작용(R) 힘을 다음과 같이 표현할 수 있습니다

$$\left[
\begin{array}
~1/3 & 1/3 & 1 & 0 & 0\\
-2/3 & -2/3 & 0 & 1 & 0\\
-2/3 & 2/3 & 0 & 0 & 1\\
\end{array}
\right]
\left[
\begin{array}
~T_{BD}\\
T_{BE}\\
R_{x}\\
R_{y}\\
R_{z}\\
\end{array}
\right]
=
\left[
\begin{array}
~195\\
390\\
-130\\
\end{array}
\right]$$

and the moments as

그리고 모멘트는 다음과 같습니다

$$\left[
\begin{array}
~2 & -2\\
1 & 1\\
\end{array}
\right]
\left[
\begin{array}
~T_{BD}\\
T_{BE}\\
\end{array}
\right]
=
\left[
\begin{array}
~780\\
1170\\
\end{array}
\right]$$

Where $T$ is the tension in the respective cord and $R$ is the reaction force in a respective direction. Then you just have six equations:

여기서 $T$는 각 코드의 장력이고 $R$은 각 방향의 반작용력입니다. 그러면 여섯 개의 방정식이 있습니다:


$\sum F_{x} = 0 = T_{BE}/3+T_{BD}/3-195+R_{x}$

$\sum F_{y} = 0 = (-\frac{2}{3})T_{BE}-\frac{2}{3}T_{BD}-390+R_{y}$

$\sum F_{z} = 0 = (-\frac{2}{3})T_{BE}+\frac{2}{3}T_{BD}+130+R_{z}$

$\sum M_{x} = 0 = 780+2T_{BE}-2T_{BD}$

$\sum M_{z} = 0 = 1170-T_{BE}-T_{BD}$


You now have five unknowns with five equations, and can solve for:

이제 다섯 개의 미지수와 다섯 개의 방정식이 있으며, 다음을 풀 수 있습니다:

$\ T_{BD} = 780N$

$\ T_{BE} = 390N$

$\ R_{x} = -195N$

$\ R_{y} = 1170N$

$\ R_{z} = 130N$

+++

## Wrapping up
## 마무리

You have learned how to use arrays to represent points, forces, and moments in three dimensional space. Each entry in an array can be used to represent a physical property broken into directional components. These can then be easily manipulated with NumPy functions.

3차원 공간에서 점, 힘 및 모멘트를 표현하기 위해 배열을 사용하는 방법을 배웠습니다. 배열의 각 항목은 방향 성분으로 나뉜 물리적 특성을 나타내는 데 사용할 수 있습니다. 이것들은 NumPy 함수로 쉽게 조작할 수 있습니다.

### Additional Applications
### 추가 응용

This same process can be applied to kinetic problems or in any number of dimensions. The examples done in this tutorial assumed three dimensional problems in static equilibrium. These methods can easily be used in more varied problems. More or less dimensions require larger or smaller arrays to represent. In systems experiencing acceleration, velocity and acceleration can be similarly be represented as vectors as well.

동일한 과정을 운동 문제나 임의의 차원에 적용할 수 있습니다. 이 튜토리얼에서 수행한 예제는 정적 평형 상태의 3차원 문제를 가정했습니다. 이러한 방법은 더 다양한 문제에서도 쉽게 사용할 수 있습니다. 더 많거나 적은 차원은 더 크거나 작은 배열로 표현해야 합니다. 가속도를 경험하는 시스템에서는 속도와 가속도도 비슷하게 벡터로 표현할 수 있습니다.

### References
### 참고 문헌

1. [Vector Mechanics for Engineers: Statics and Dynamics (Beer & Johnston & Mazurek & et al.)](https://www.mheducation.com/highered/product/Vector-Mechanics-for-Engineers-Statics-and-Dynamics-Beer.html)
2. [NumPy Reference](https://numpy.org/doc/stable/reference/)