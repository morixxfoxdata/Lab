import numpy as np
import matplotlib.pyplot as plt
# パラメータ設定
N = 1000    # 原信号の次元数
M = 100     # 出力ベクトルの次元数
non_zero_elements = 20      # 正規分布に従うランダムな非ゼロ成分数

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
    plt.stem(indices, original_signal, use_line_collection=False)
    plt.title('Original Signal')
    plt.xlabel('index of x0')
    plt.ylabel('Signal Magnitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # print(path)
    # plot_x()
    plot_y()