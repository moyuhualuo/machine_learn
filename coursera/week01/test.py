import matplotlib.pyplot as plt

# 创建所有的图形并存储在列表中
figs = []

# 创建第一个图形
fig1, ax1 = plt.subplots()
ax1.plot(0, 0)
figs.append(fig1)

# 创建第二个图形
fig2, ax2 = plt.subplots()
ax2.plot(0, 0)
figs.append(fig2)

# 创建更多图形...

# 一次性显示所有的图形
for fig in figs:
    fig.show()

# 或者保存图形到文件
for i, fig in enumerate(figs):
    fig.savefig(f'figure_{i}.png')

# 关闭图形

