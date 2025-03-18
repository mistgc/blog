---
title: torch.expand() And torch.repeat()
subtitle:
date: 2025/3/18 13:42:00
tag: tech, python
---

## `torch.expand()`

`torch.expand()` makes Tensor to expand to a given shape.

```python
In [1]: import torch

In [2]: a = torch.randn(3, 4)

In [3]: a.unsqueeze(0).shape
Out[3]: torch.Size([1, 3, 4])

In [4]: a.unsqueeze(0).expand(5, 3, 4).shape
Out[4]: torch.Size([5, 3, 4])

In [5]: a.unsqueeze(0).expand(5, -1, -1).shape
Out[5]: torch.Size([5, 3, 4])
```

## `torch.repeat()`

`torch.repeat()` makes each dimension in a Tensor to repeat given times.

```python
In [1]: import torch

In [2]: a = torch.randn(3, 4)

In [3]: a.unsqueeze(0).shape
Out[3]: torch.Size([1, 3, 4])

In [4]: a.unsqueeze(0).repeat(5, 1, 1).shape
Out[4]: torch.Size([5, 3, 4])

In [5]: a.unsqueeze(0).repeat(5, 2, 3).shape
Out[5]: torch.Size([5, 6, 12])
```
