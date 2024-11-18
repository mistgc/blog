---
title: 关于二分查找的细节问题思考
subtitle: Binary Search.
data: 2022/3/9 22:46:00
math: true
tags: tech
---
![](https://gitee.com/mistgc/pic-go/raw/master//20220309190051.png)
# 关于二分查找的细节问题思考
> "《编程珠玑》中提到：在时间充足的情况下，有90%的人写不出完全正确的二分查找法。"
>
> "第一篇二分查找的论文发表于1946年，然而第一个没有bug的二分查找法却是在1962年才出现。"

笔者最开始还觉得这书里写得有点夸张了，但是前几天做了一道题时，使用二分查找时，结果一直有误，然后笔者就开始深究。

## 写在前面

这里笔者先抛出四个问题，让大家先思考一下。给定一个数组 $array = [1,2,3,5,5,5,8,9]$，现在需要解决下列问题：
- 找到第一个 $\ge 5$ 的元素
- 找到最后一个 $< 5$ 的元素
- 找到第一个 $> 5$ 的元素
- 找到最后一个 $\le 5$ 的元素

## 二分查找的思想
二分查找的本质就是在**有序数组**中通过不断缩小查找的范围来提高查找的效率。

时间复杂度为 $O(\log n)$。

## 细节问题
### 溢出问题
我们一般在写二分查找法时，都是通过 $mid = \frac{l+r}{2}$ 来确定中间数的。但是如果当 $l$ 和 $r$ 相当大的时候，$l+r$ 可能会溢出。所以我们采用下面的方式在确定 $mid$ 的值。
$$
mid = l + \frac{r - l}{2}
$$

### 取整问题

我们知道 *C/C++* 中整数的除法是**向下取整**的。我们要实现**向上取整**可以在向下取整的结果上加 1。

```c
// From: https://www.zhihu.com/question/36132386/answer/105595067
int binary_search_2(int a[], int n, int key)
{
    int m, l = 0, r = n - 1;//闭区间[0, n - 1]
    while (l < r)
    {
        m = l + ((r + 1 - l) >> 1);//向上取整
        if (a[m] <= key) l = m;
        else r = m - 1;
    }
    if (a[l] == key) return l;
    return -1;
}
```
这段代码中我们看到，当 $n = 5$ 时，第一次 $m = 0 + \frac{4 + 1 - 0}{2} = 2$ ，但实际的向上取整应该是 $m = 0 + (\frac{4 - 0}{2} + 1) = 3$。

当 $n = 6$ 时，第一次 $m = 3$，这和实际向上取整的值相等。

现在我们来好好想想这个问题，向上取整的值真的等于向下取整的值加 1 吗？其实不是，当一个数除以另一个数时，如果结果本身就是一个整数，那它就不需要再去取整。只有当结果不为整数时，才需要向上取整。

我们再来看看上面的代码：

当 $n = 5$ 时，$r = n - 1 = 4,~l = 0$ 则 $ans = \frac{r - l}{2} = 2$，结果本身就是一个整数，不需要再取整。

当 $n = 6$ 时，$r = n - 1 = 5,~l = 0$ 则 $ans = \frac{r - l}{2} = 2.5$，结果并不是整数。所以我们需要取整。使用向上取整还是向下取整，又实际的问题来确定，我们这里就假定使用向上取整。这时，取整的结果就是向下取整的结果加 1。即：
$$
ans = \lfloor \frac{r - l}{2} \rfloor + 1
$$
伪代码实现为：
```c
ans = (r - l) / 2 + 1
```

根据上述，我们发现需要分成两种情况讨论，1）结果本身为整数，2）结果不为整数。这样就会用到判断语句，会使算法低效。所以我们需要一种可以用于两种情况的代码来。没错，就是前文中的代码，用数学符号表示：
$$
ans = \lfloor \frac{r - l + 1}{2}\rfloor
$$
伪代码实现为：
```c
ans = (r - l + 1) / 2
```
> 这里和前文中的有点不一样，因为我们这里还会考虑 $r + 1 - l$ 中 $r + 1$ 是否会溢出。所以这里写出了 $r - l + 1$。

### 边界问题

笔者写这篇文章的主要目的就是为了探讨关于**关于二分查找法的边界问题**。

因为边界问题的讲解非常麻烦，所以我们这里通过前面的问题来讲解。

给定一个数组 $array = [1,2,3,5,5,5,8,9]$，现在需要解决下列问题：
- 找到第一个 $\ge 5$ 的元素
- 找到最后一个 $< 5$ 的元素
- 找到第一个 $> 5$ 的元素
- 找到最后一个 $\le 5$ 的元素

我们先用第一个问题实例来讲：我们通过直接观察其实就可以知道，第一个问题的答案是 $5$ （下标为 $3$ 的 $5$ ）。这里先使用模板代码讲解一下：
```c++
int binary_search(vector<int>& arr, int target) {
	int mid, l = 0, r = arr.size() - 1;	// The interval is [0, n - 1].
	while(l < r) {
		mid = l + (r - l) / 2;
		if(arr[mid] >= target) r = mid;
		else l = mid + 1;
	}
	return arr[l];	// arr[l] or arr[r]
}
```
*笔者这里不推荐背代码这种费时费力，还容易错的方式。我们要学习背后的原理。*

我们使用二分查找，首先就需要申明一些必要的变量 $mid,~ l,~ r$ 。$mid$ 代表中位数，$l$ 代表左指针，$r$ 代表右指针。并且通过变量初始化来指定区间。（由题$target =  5$）

`while(l < r) {...}`，左右指针不断逼近，当 $l = r$ 时，跳出循环。所以`return arr[l]` 和 `return arr[r]` 是一样的。

重要的部分来了。我们这里先假设使用向下取整，等分析后续代码后再来判断到底是使用**向下取整**还是**向上取整**。循环体中的判断语句根据题目的要求而设定——找到第一个 $\ge 5$ 的元素。即：`if(arr[mid] >= target) {...} else {...}`。

当 `arr[mid] >= target` 成立时，当前的 `arr[mid]` 可能就是我们的目标值，但也有可能目标值还在它的左边。所以在下一次循环中，我们需要将当前的 $mid$ 也加入查找区间中。即:
$$
target ~~~\text{in}~~~ \{~arr[l],~arr[l +1],~arr[l+2],~\dots~,~arr[mid]~\}
$$
当 `arr[mid] >= target` 不成立时，当前的 `arr[mid]` 值一定不是我们的目标值，在下一次循环中，应该不能将当前的 $mid$ 加入查找区间中。即：
$$
target ~~~\text{in}~~~ \{~arr[mid +1],~arr[mid+2],~\dots~,~arr[r]~\}
$$

现在我们来分析一下，到底是使用向上取整还是向下取整。因为 `while(l < r) {...}`，我们知道在循环中时，$l < r$ 一定成立，假设 $l = 1,~	r = 2$，这时如果采用向下取整，则 $mid = 1$。

当 `arr[mid] >= target` 成立时，根据代码得到 $r = mid = 1$，即 $l = r$ 成立，跳出循环。

当 `arr[mid] >= target` 不成立时，根据代码得到 $l = mid + 1 = 2$，即 $l = r$ 成立，跳出循环。

向下取整不会造成死循环，所以我们使用**向下取整**。

---

这里再来一个反例，使用向上取整才不会造成死循环的。上面的第4题就是这样的。这里直接把代码写出来。
```c++
int binary_search(vector<int>& arr, int target) {
	int mid, l = 0, r = arr.size() - 0;	// The interval is [0, n - 1].
	while(l < r) {
		mid = l + (r - l + 1) / 2;
		if(arr[mid] <= target) l = mid;
		else r = mid - 1;
	}
	return arr[l];	// arr[l] or arr[r]
}
```
**向下取整：**
同样，因为 `while(l < r) {...}`，我们知道在循环中时，$l < r$ 一定成立，假设 $l = 1,~	r = 2$，这时如果采用向下取整，则 $mid = 1$。

当 `arr[mid] >= target` 成立时，根据代码得到 $l = mid = 1$，即 $l < r$ 循环继续，并且仍然是 $l = 1,~r=2$ 所以会进行**死循环**。

当 `arr[mid] >= target` 不成立时，根据代码得到 $r = mid - 1 = 0$，即 $l > r$ 成立，跳出循环。

我们发现这样会造成死循环。

**向上取整：**
这时采用向下取整，则 $mid = 2$。

当 `arr[mid] >= target` 成立时，根据代码得到 $l = mid = 2$，即 $l = r$ 成立，跳出循环。

当 `arr[mid] >= target` 不成立时，根据代码得到 $r = mid - 1 = 1$，即 $l = r$ 成立，跳出循环。

这样就不会造成死循环，所以我们使用**向上取整**。

最后给出测试用例的所有代码：
```c++
#include <iostream>
#include <windows.h>

using namespace std;

int main() {
	int array[] = {1, 2, 3, 5, 5, 5, 8, 9};
	int mid, l = 0, r = sizeof(array) / sizeof(array[0]) - 1;
	// Find the first element greater than or equal to 5. (e >= 5)
	/*
	while(l < r) {
		mid = l + (r - l) / 2;
		if(array[mid] >= 5) r = mid;
		else l = mid + 1;
		cout << "mid = " << mid << " l = " << l << " r = " << r << endl;
		Sleep(100);
	}
	*/
	// Find the last element less than or equal to 5. (e <= 5)
	/*
	while(l < r) {
		mid = l + (r - l + 1) / 2;
		if(array[mid] <= 5) l = mid;
		else r = mid - 1;
		cout << "mid = " << mid << " l = " << l << " r = " << r << endl;
		Sleep(100);
	}
	*/
	// Find the first element greater than 5. (e > 5)
	/*
	while(l < r) {
		mid = l + (r - l) / 2;
		if(array[mid] > 5) r = mid;
		else l = mid + 1;
		cout << "mid = " << mid << " l = " << l << " r = " << r << endl;
		Sleep(100);
	}
	*/
	// Find the last element less than 5. (e < 5)
	/*
	while(l < r) {
		mid = l + (r - l + 1) / 2;
		if(array[mid] < 5) l = mid;
		else r = mid - 1;
		cout << "mid = " << mid << " l = " << l << " r = " << r << endl;
		Sleep(100);
	}
	*/
	cout << array[l] << endl;
	return 0;
}
```

## 结语
虽然二分查找的思想比较简单，但是到遇到关于边界问题和取整问题时，二分查找的调试相当复杂。如果有相关问题需要探讨，可以去笔者 *About* 页面，联系笔者。

*我与诸君暂离，日后顶峰相见。*

**Pause.**
