# sky-atmosphere

## 题面

室外场景换动态天空后，正午发绿、日落不发红：Rayleigh 用 1/λ² 近似，蓝天过绿；波长传参用 μm 但公式按 nm，系数差 10⁹；Mie 相位用 Rayleigh 公式，太阳周围没有光晕；g 取 1.0，HG 分母为 0 出现 inf；光学深度按常量，地平线不发红；太阳低于地平线时仍加直射项，夜空变蓝；tonemap 用截断而不是曲线，高光死白；曝光为负时颜色为负，后续幂运算出错；批量共用累积缓冲，样本之间串；俯仰角超过 90 度不钳制，天顶外出现黑带。sky/scatter.py 是核心，tests/test_scatter.py 15个红。要修的是：修复上述 10 个缺陷，使 15 个测试全部通过，且不得修改 tests/test_scatter.py。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始（未修复）状态必须恰好是 `15 failed`，且不能出现 collection、import 或 syntax error。所有失败都来自 `sky/scatter.py` 中的桩函数抛出 `NotImplementedError`。

输出尾部形如：

```text
15 failed in 0.12s
```

实现只允许使用 Python 标准库（`math`）与 pytest；不得联网、不得使用随机数，结果需可复现。数值断言容差为 `1e-6`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_scatter.py` 是验收测试，不可修改；只能修改实现文件（核心为 `sky/scatter.py`）。仓库初始状态不包含任何 Git 元数据，也不需要执行 `git init` 或推送。

## 文件结构

```text
sky-atmosphere/
├── README.md
├── conftest.py
├── .gitignore
├── sky/
│   ├── __init__.py
│   └── scatter.py
└── tests/
    └── test_scatter.py
```
