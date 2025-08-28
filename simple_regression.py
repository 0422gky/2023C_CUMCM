import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("开始r_i,t和每日品类销量的回归分析...")

# 1. 加载数据
df = pd.read_csv('file_23_r_with_category.csv')
df['销售日期'] = pd.to_datetime(df['销售日期'])
df['r_i,t'] = pd.to_numeric(df['r_i,t'], errors='coerce')
df['每日品类销量'] = pd.to_numeric(df['每日品类销量'], errors='coerce')
df = df.dropna(subset=['r_i,t', '每日品类销量'])

print(f"数据加载完成，共 {len(df)} 行")

# 2. 基础统计分析
print("\n=== 基础统计分析 ===")
print("r_i,t统计：", df['r_i,t'].describe())
print("每日品类销量统计：", df['每日品类销量'].describe())
print("相关系数：", df['r_i,t'].corr(df['每日品类销量']))

# 3. 简单线性回归
X = df[['r_i,t']].values
y = df['每日品类销量'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== 回归结果 ===")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"系数: {model.coef_[0]:.4f}")
print(f"截距: {model.intercept_:.4f}")

# 4. 可视化
plt.figure(figsize=(12, 8))

# 散点图和回归线
plt.subplot(2, 2, 1)
plt.scatter(df['r_i,t'], df['每日品类销量'], alpha=0.6, s=20)
plt.plot(df['r_i,t'], model.predict(df[['r_i,t']]), 'r-', lw=2)
plt.xlabel('定价系数r_i,t')
plt.ylabel('每日品类销量(千克)')
plt.title('r_i,t vs 每日品类销量')
plt.grid(True, alpha=0.3)

# 预测vs实际
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测 vs 实际')
plt.grid(True, alpha=0.3)

# 残差分析
plt.subplot(2, 2, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')
plt.grid(True, alpha=0.3)

# 时间序列
plt.subplot(2, 2, 4)
daily_stats = df.groupby('销售日期').agg({
    'r_i,t': 'mean',
    '每日品类销量': 'sum'
}).reset_index()

plt.plot(daily_stats['销售日期'], daily_stats['r_i,t'], label='平均r_i,t', color='red')
ax2 = plt.twinx()
ax2.plot(daily_stats['销售日期'], daily_stats['每日品类销量'], label='总销量', color='blue', alpha=0.7)
plt.xlabel('日期')
plt.ylabel('平均r_i,t', color='red')
ax2.set_ylabel('总销量(千克)', color='blue')
plt.title('时间趋势')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n分析完成！图表已保存为 regression_analysis.png")
