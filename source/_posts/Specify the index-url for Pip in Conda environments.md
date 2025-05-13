---
title: Specify The `index-url` for Pip in Conda Environments
subtitle: Pip in Conda
date: 2025/5/13 22:46:00
tags: tech, python
---

```yaml
name: foo
channels:
  - defaults
dependencies:
  - python
  - pip
  - pip:
    - --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    - --extra-index-url https://download.pytorch.org/whl/cu116
    - torch==1.12.1+cu116
```
