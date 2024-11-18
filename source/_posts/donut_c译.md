---
title: Dount.c 文章分析
subtitle: 我英语不是很好哦！！！
data: 2021/11/17 00:26:00
math: true
tags: tech
---

## 屏幕映射

​		在屏幕上，我们只能看见两个坐标( (x, y) 或者其他) 所以我们需要将在三维空间中的物体投影 (project) 在我们的屏幕上。这就需要我们来得知它们的标准坐标 (Standard Coordinate) 到我们屏幕坐标 (Screen Coordinate) 的比例。(详见下图)

![](https://www.a1k0n.net/img/perspective.png)

​																												图一.

数学表达式如下:
$$
\frac {y'}{z} = \frac {y}{z}.\\
y' = \frac {yz'}{z}.
\tag 1
$$
​		如图一所示，我们的Z轴 (Z-Axis) 是一个固定的常量，所以我们可以给它重命名为 $K_1$。所以我们将会得到投影方程 (projection equation):
$$
(x',y') = (\frac{K_1x}{z},\frac{K_1y}{z})\tag 2
$$
我们可以通过修改$K_1$的值来让图形更好的出现在我们的屏幕上。

​		我们在绘制一些点的时候，可能会在相同的位置 $(x',y')$ 上绘制不同的点，这些点的深度 (depths) 不同，所以我们需要一个缓冲区 (buffer) 来储存并维护这些深度值 (aka: Z轴)。这样我们在绘制点的时候可以检测这个点是否已经绘制过了。这也将有助于计算
$$
z^{-1} = \frac{1}{z}
$$
并使用。

## 如何画出一个Dount

​		有一种方法就是先画出一个2D的圆环，在让其绕着Dount的中心轴进行旋转。

![](https://www.a1k0n.net/img/torusxsec.png)

​		我们要有一个半径为 $R_1$ ,以 $(R_2,0,0)$ 为圆心的圆，并画在 xy面 上。

​		我们再设一个从 0 到 2$\pi$ 的角度 $\theta$ :
$$
(x,y,z) = (R_2,0,0) + (R_1cos\theta,R_1sin\theta,0)\tag 3
$$
​		现在我们让这个物体以另一个角度 $\phi$ 绕着Y轴  (Y-Axis) 旋转。我们要旋转所有或者说是任意的 (arbitrary) 3D坐标点，我们需要将他们乘上[旋转矩阵]([旋转矩阵 - 维基百科，自由的百科全书 (wikipedia.org)](https://zh.wikipedia.org/wiki/旋转矩阵))(Rotation Matrix) 。当我们得到刚才的那些坐标点后，我们再将它们绕着Y轴进行旋转:
$$
(R_2+R_1cos\theta,\ R_1sin\theta,\ 0) \cdot 
\begin{pmatrix}
cos\phi & 0 & sin\phi\\ 
0 & 1 & 0\\ 
-sin\phi & 0 & cos\phi
\end{pmatrix}=((R_2+R_1cos\theta)cos\phi,\ R_1sin\theta,\ -(R_2+R_1cos\theta)sin\phi)\tag 4
$$
但是我们需要让这个物体绕着至少两个轴进行旋转来达到动画的效果。这些旋转的角度我们可以称为A，B。通过角度A来绕着X轴进行旋转，同理，通过角度B可以让物体绕着Z轴旋转。这一过程是复杂的，但是我们知道，这也是通过旋转矩阵来实现的，我们可以用矩阵乘法来实现:
$$
(R_2+R_1cos\theta,\ R_1sin\theta,\ 0) \cdot 
\begin{pmatrix}
cos\phi & 0 & sin\phi\\ 
0 & 1 & 0\\ 
-sin\phi & 0 & cos\phi
\end{pmatrix}\cdot
\begin{pmatrix}
1 & 0 & 0\\
0 & cosA & sinA \\
0 & -sinA & cosA 
\end{pmatrix} \cdot
\begin{pmatrix}
cosB & sinB & 0\\
-sinB & cosB & 0\\
0 & 0 & 1
\end{pmatrix}\tag 5
$$
​		通过上述的推导，我们获得了这个以原点为中心的物体的各个坐标点，并且绕着两个轴 (two axes) 进行旋转。为了获得屏幕坐标我们需要做以下两点:

- 将这个物体 (aka: dount) 移到观察者的前面(观察者在原点)。所以我们需要向 **z** 一些常量 (constant) 让它向后退。
- 将3D图形投影到2D屏幕上

我们将引入的常量命名为 $K_2$ 。这时我们的投影方程就变成了:
$$
(x',y') = (\frac{K_1x}{K_2+z},\frac{K_1y}{K_2+z})\tag 6
$$


这时我们可以通过调整(tweak) $K_1$ 和 $K_2$ 来改变视场，并使物体的深度变平或放大。

​		我们可以在我们的程序中实现一个 **3x3矩阵乘法** 运算。并且以一种直观的方式实现出来:
$$
\left( \begin{matrix} x \\ y \\ z \end{matrix} \right) =
\left( \begin{matrix}
 (R_2 + R_1 \cos \theta) (\cos B \cos \phi + \sin A \sin B \sin \phi) - 
   R_1 \cos A \sin B \sin \theta \\

 (R_2 + R_1 \cos \theta) (\cos \phi \sin B - \cos B \sin A \sin \phi) + 
   R_1 \cos A \cos B \sin \theta \\
 \cos A (R_2 + R_1 \cos \theta) \sin \phi + R_1 \sin A \sin \theta
\end{matrix} \right) \tag 7
$$
这个表达式 (expression) 看起来就很复杂，让人不知道如何进行计算，但是我们仔细观察发现在右侧的矩阵中有部分相同的表达式，

比如：$R_2 + R_1 \cos \theta$ 我们可以提前就计算好这些数据。

## 光照计算(未完待续)

​		到了现在，我们知道了再屏幕的哪一个地方进行绘制像素了，但是我们还没有加入阴影。对于光照的计算，我想各位应该在微积分中学过关于3D图形求法向量的知识吧。这里我们就将使用它来计算表面法线。让表面法线与光线，这两个向量进行**点积(**cross product)，这将会让我们等到表面法线与光线的夹角的cos值:如果这个值是**大于0**，则证明这个表面是面对着光线的( 值的大小由角度来决定 )。如果这个值**小于0**，则代表着这个表面是背对光线的。值越高，这部分的光线就越强。  





