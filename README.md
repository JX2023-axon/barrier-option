# Derivatives

## 项目简介

本项目围绕 Heston 随机波动率模型下障碍期权的 PDE 数值求解展开，包含模型推导、数值实验以及可复用的 Python 模块封装。

## 文件说明

1. `HestonModel.ipynb`
   - Heston 模型的 PDE 推导。

2. `BarrierOption.ipynb`
   - 以向下敲出看涨欧式期权为例的数值实验。
   - 包含定价计算、收敛性测试以及 Greeks 计算。

3. `barrier_heston_adi.py`
   - 将主要数值求解流程封装为可复用模块。
   - 便于后续直接导入并作为函数调用。

## 环境依赖

项目当前使用的主要 Python 依赖如下：

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

可通过以下命令安装：

```bash
pip install -r requirements.txt
```
