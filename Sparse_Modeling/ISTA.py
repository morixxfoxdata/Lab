import numpy as np
import matplotlib.pyplot as plt
def soft_thresholding(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def ista(A, y, lambda_, num_iterations, t):
    x = A.T @ y
    for i in range(num_iterations):
        gradient = A.T @ (A @ x - y)
        x = soft_thresholding(x - t * gradient, lambda_ * t)
    return x

def admm_lasso(A, y, lambda_, rho, max_iter=100):
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    v = np.zeros(n)
    
    A_T_A = A.T @ A
    A_T_y = A.T @ y
    L = np.linalg.inv(A_T_A + rho * np.eye(n))  # Precompute the matrix inverse

    for _ in range(max_iter):
        # x update (solve linear system)
        x = L @ (A_T_y + rho * (z - v))
        # z update (soft thresholding)
        z = soft_thresholding(x + v, lambda_ / rho)
        # u update (Lagrange multiplier)
        v += x - z
    
    return x

# テストデータ生成（例）
np.random.seed(0)
# m, n = 100, 20  # mは観測数、nは変数の数
# A = np.random.randn(m, n)
# x_true = np.random.randn(n)
# y = A @ x_true + np.random.randn(m) * 0.5  # 観測ベクトル
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
observation_matrix = np.random.randn(M, N)

# 出力ベクトルを計算(行列積)
output_vector = observation_matrix @ original_signal


# パラメータ設定
lambda_ = 0.1  # 正則化パラメータ
t = 0.001      # ステップサイズ（適切に選ばないと収束しにくいので注意）
num_iterations = 100  # 反復回数
# ISTAを実行
x_estimated_ista = ista(observation_matrix, output_vector, lambda_=1, num_iterations=num_iterations, t=0.001)
x_estimated_ADMM = admm_lasso(observation_matrix, output_vector, lambda_=1, rho=1)

def plot_signals(type, original_signal, predicted_signal, num_iterations):
    indices = np.arange(len(original_signal))  # 信号のインデックス

    plt.figure(figsize=(15, 5))
    plt.stem(indices, original_signal, linefmt='b-', markerfmt='bo', basefmt='r-', label='Original Signal')
    plt.stem(indices, predicted_signal, linefmt='g-', markerfmt='gx', basefmt='r-', label='Predicted Signal')
    plt.title(f'Comparison between Original and {type} Predicted Signals. iter={num_iterations}')
    plt.xlabel('Index')
    plt.ylabel('Signal Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # print("真の係数:", original_signal)
    # print("推定係数:", x_estimated)
    plot_signals("ISTA", original_signal, x_estimated_ista, num_iterations=num_iterations)
    plot_signals("ADMM", original_signal, x_estimated_ADMM, num_iterations=num_iterations)