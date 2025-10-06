import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 1. 生成示例数据
# 为了演示，我们生成一组带有噪声的多项式数据
np.random.seed(0)
x = np.linspace(0, 10, 50)
y = 0.5 * x**2 - 2 * x + 5 + np.random.normal(0, 3, 50)

# 2. 拟合线性模型
linear_coeffs = np.polyfit(x, y, 1)
y_pred_linear = np.poly1d(linear_coeffs)(x)

# 3. 拟合多项式模型（这里以2次为例，您可以根据需要调整）
poly_coeffs = np.polyfit(x, y, 4)
y_pred_poly = np.poly1d(poly_coeffs)(x)

# 4. 拟合指数模型
# 定义指数函数
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# 使用 curve_fit 找到最佳参数
try:
    popt_exp, pcov_exp = curve_fit(exponential_func, x, y)
    y_pred_exp = exponential_func(x, *popt_exp)
except RuntimeError:
    y_pred_exp = np.full_like(y, np.nan)
    print("Warning: 指数模型拟合失败，请检查数据或初始猜测值。")

# 5. 使用 R-squared ($R^2$) 评估拟合优度
r2_linear = r2_score(y, y_pred_linear)
r2_poly = r2_score(y, y_pred_poly)
if not np.isnan(y_pred_exp).any():
    r2_exp = r2_score(y, y_pred_exp)
else:
    r2_exp = np.nan

# 6. 打印结果
print("模型决定系数 ($R^2$) 结果：")
print(f"线性模型: {r2_linear:.4f}")
print(f"多项式模型 (2次): {r2_poly:.4f}")
if not np.isnan(r2_exp):
    print(f"指数模型: {r2_exp:.4f}")
else:
    print("指数模型: 拟合失败")

# 7. 找出最佳模型
results = {
    'Linear': r2_linear,
    'Polynomial (2nd Degree)': r2_poly,
    'Exponential': r2_exp
}
best_model = max(results, key=lambda k: results[k] if not np.isnan(results[k]) else -np.inf)
print(f"\n结论：最佳拟合模型是 '{best_model}'，其 $R^2$ 值为 {results[best_model]:.4f}。")

# 8. 可视化结果（可选）
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='原始数据', color='blue')
plt.plot(x, y_pred_linear, label=f'线性拟合 ($R^2$={r2_linear:.4f})', color='green', linestyle='--')
plt.plot(x, y_pred_poly, label=f'多项式拟合 ($R^2$={r2_poly:.4f})', color='red', linestyle='-')
if not np.isnan(y_pred_exp).any():
    plt.plot(x, y_pred_exp, label=f'指数拟合 ($R^2$={r2_exp:.4f})', color='purple', linestyle='-.')

plt.title('不同模型拟合效果比较', fontsize=16)
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()