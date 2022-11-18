import numpy as np
from torch.utils.tensorboard import SummaryWriter


# データを作る
np.random.seed(123)
x = np.random.randn(100)
y = x.cumsum()  # xの累積和

# ここではmatplotlib での以下に相当するものをTensorBoard で表示します。
# t = np.arange(100)
# plt.plot(t, x)
# plt.plot(t, y)

# log_dirでlogのディレクトリを指定
writer = SummaryWriter(log_dir="./logs")

# xとyの値を記録していく
for i in range(100):
    writer.add_scalar("x", x[i], i)
    writer.add_scalar("y", y[i], i)

# writerを閉じる
writer.close()
