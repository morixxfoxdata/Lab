import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# パラメータ設定
N = 1000    # 原信号の次元数
M = 100     # 出力ベクトルの次元数
non_zero_elements = 20      # 正規分布に従うランダムな非ゼロ成分数
np.random.seed(0)
# 1000次元の原信号ベクトルを生成(大部分は0)
original_signal = np.zeros(N)
# indexをランダムに選択
non_zero_indices = np.random.choice(N, non_zero_elements, replace=False)

# 選ばれたindexに正規分布の値を設定
original_signal[non_zero_indices] = np.random.randn(non_zero_elements)

# 1000, 100の観測行列を生成(各要素は正規分布に従う)
observation_matrix = np.random.randn(N, M)

# 出力ベクトルを計算(行列積)
output_vector = observation_matrix.T @ original_signal

"""
基底追跡(L1)によって推定する
"""
def l1_norm(x):
    return np.sum(np.abs(x))

# L2ノルム最小化の目的関数
def l2_norm(x):
    return np.sum(x**2)

# 制約条件（A*x = y）
cons = ({'type': 'eq', 'fun': lambda x:  np.dot(observation_matrix.T, x) - output_vector})

# 最適化問題の初期値
# x0 = np.zeros(1000)
x0 = np.random.randn(N)  # 初期値をランダムに設定してみる
# 最適化実行
# res_l1 = minimize(l1_norm, x0, method='SLSQP', constraints=cons)
# res_l2 = minimize(l2_norm, x0, method='SLSQP', constraints=cons)


# print(output_vector.shape)
def plot_y():
    # 出力ベクトルのインデックスを生成
    indices = np.arange(len(output_vector))
    plt.figure(figsize=(10, 5))
    plt.stem(indices, output_vector)
    plt.title('Output Vector')
    plt.xlabel('Index of y')
    plt.ylabel('Signal Magnitude')
    plt.grid(True)
    plt.show()

def plot_x():
    indices = np.arange(len(original_signal))
    plt.figure(figsize=(10, 5))
    plt.stem(indices, original_signal)
    plt.title('Original Signal')
    plt.xlabel('index of x0')
    plt.ylabel('Signal Magnitude')
    plt.grid(True)
    plt.show()

def plot_signals(type, original_signal, predicted_signal):
    indices = np.arange(len(original_signal))  # 信号のインデックス

    plt.figure(figsize=(15, 5))
    plt.stem(indices, original_signal, linefmt='b-', markerfmt='bo', basefmt='r-', label='Original Signal')
    plt.stem(indices, predicted_signal, linefmt='g-', markerfmt='gx', basefmt='r-', label='Predicted Signal')
    plt.title(f'Comparison of Original and {type} Predicted Signals')
    plt.xlabel('Index')
    plt.ylabel('Signal Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == '__main__':
    # print(path)
    # plot_x()
    plot_y()
    # 結果の表示
    # print("Optimal solution:", res.x)
    # print("Success:", res_l1.success)
    # plot_signals(original_signal, res_l1.x)
    # plot_signals('l1', original_signal, res_l1.x)