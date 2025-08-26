# PySR缓存和方程保存功能说明

## 概述
为了避免重复训练PySR模型并方便使用结果，我们实现了一个智能缓存和方程保存系统。该系统可以：
- 自动保存训练好的PySR模型
- 检测数据是否发生变化
- 从缓存中快速加载已训练的模型
- **将最佳方程保存为干净的JSON格式**
- 管理缓存文件

## 主要功能

### 1. 自动缓存
```python
from src.scripts.Experiments.regression_analysis import run_pysr

# 第一次运行 - 训练并缓存模型
model = run_pysr("instance_name", use_cache=True)

# 第二次运行 - 从缓存加载
model = run_pysr("instance_name", use_cache=True)  # 很快就完成
```

### 2. 数据完整性检查
系统会自动检测输入数据是否发生变化：
- 如果数据没变，直接从缓存加载
- 如果数据改变，重新训练模型

### 3. 强制重新训练
```python
# 忽略缓存，强制重新训练
model = run_pysr("instance_name", use_cache=False)
```

### 4. 方程保存功能
```python
from src.scripts.Experiments.regression_analysis import run_pysr, save_best_equation

# 自动保存方程（推荐）
model = run_pysr("instance_name", use_cache=True, save_equation=True)

# 手动保存方程
model = run_pysr("instance_name", save_equation=False)
save_best_equation(model, "instance_name")

# 批量处理和保存
from src.scripts.Experiments.regression_analysis import batch_save_equations
results = batch_save_equations(["name1", "name2"], use_cache=True)
```

### 5. 缓存管理
```python
from src.scripts.Experiments.regression_analysis import clear_cache

# 清除特定实例的缓存
clear_cache("instance_name")

# 清除所有缓存
clear_cache()
```

## 文件结构
```
results/pysr_results/
├── cache/                       # 缓存目录
│   ├── instance1_model.pkl      # 模型文件
│   ├── instance1_model_info.json # 元数据文件
│   ├── instance2_model.pkl
│   └── instance2_model_info.json
├── instance1.summary.json       # 干净的方程文件
├── instance2.summary.json
└── ...
```

### 方程JSON文件格式
方程文件格式非常干净，只包含纯净的SymPy方程字符串：
```json
"x0**2 + 0.5*x0 - 1.0"
```

**特点：**
- 使用 `model.sympy()` 获取纯净方程，而非 `model.get_best()`
- 无多余的键值对结构，只有方程本身
- 无缩进，紧凑格式
- SymPy标准格式（x0, x1等变量名，**表示幂）
- 易于程序读取和处理
- 人类可读

**常见方程格式示例：**
```json
"x0*2.5 + 3.14"              // 线性方程
"x0**2 + 0.5*x0 - 1.0"       // 多项式
"sin(x0) + cos(x0*2.0)"      // 三角函数
"exp(x0)/(1 + x0)"           // 指数函数
"log(Abs(x0) + 1.0e-6)"      // 对数函数
```

### 元数据文件内容
```json
{
  "name": "instance_name",
  "timestamp": "1703123456.789",
  "data_hash": "abc123def456",
  "model_type": "PySRRegressor"
}
```

## 使用建议

### 1. 开发阶段
```python
# 在开发和调试时使用缓存
model = run_pysr(name, use_cache=True)
```

### 2. 生产运行
```python
# 如果需要确保使用最新数据，可以禁用缓存
model = run_pysr(name, use_cache=False)
```

### 3. 批量处理
```python
from src.scripts.category import get_all_instance_names
from src.scripts.Experiments.regression_analysis import batch_save_equations

# 批量处理并保存所有方程
names = get_all_instance_names()
results = batch_save_equations(names, use_cache=True)

# 或者手动循环
for name in names:
    # 使用缓存加速批量处理，自动保存方程
    model = run_pysr(name, use_cache=True, save_equation=True)
```

### 4. 读取保存的方程
```python
import json
from src.scripts.paths import get_pysr_summary_path

# 读取单个方程
name = "instance_name"
equation_path = get_pysr_summary_path(name)
with open(equation_path, 'r') as f:
    equation = json.load(f)
print(f"Best equation for {name}: {equation}")

# 批量读取所有方程
equations = {}
for name in names:
    try:
        equation_path = get_pysr_summary_path(name)
        with open(equation_path, 'r') as f:
            equations[name] = json.load(f)
    except FileNotFoundError:
        equations[name] = None
```

## 性能优势
- **首次训练**: 正常时间（如100次迭代可能需要几分钟到几小时）
- **缓存加载**: 几秒钟内完成
- **存储开销**: 每个模型约几KB到几MB

## 注意事项
1. 缓存文件会占用磁盘空间，定期清理不需要的缓存
2. 如果修改了PySR的参数（如niterations），建议清除相关缓存
3. 缓存基于数据内容的MD5哈希，数据格式变化会触发重新训练
