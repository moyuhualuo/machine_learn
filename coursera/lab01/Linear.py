import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def draw_Linear():
    def compute(x, y, w_array, b_array):
        m = x.shape[0]
        total_costs = np.zeros((len(w_array), len(b_array)))
        for i, w in enumerate(w_array):
            for j, b in enumerate(b_array):
                cost_sum = 0
                for k in range(m):
                    f_wb = w * x[k] + b
                    cost = (f_wb - y[k]) ** 2
                    cost_sum += cost
                total_costs[i, j] = (1 / (2 * m)) * np.sum(cost_sum)

        return total_costs

    def update_w(val):
        w = slider_w.val
        y = w * x_train + slider_b.val
        line.set_ydata(y)
        fig.canvas.draw_idle()
    def update_b(val):
        b = slider_b.val
        y = slider_w.val * x_train + b

        line.set_ydata(y)
        fig.canvas.draw_idle()

    '''
    def compute_cost(x, y, w, b): 
        """
        Computes the cost function for linear regression.
    
        Args:
          x (ndarray (m,)): Data, m examples 
          y (ndarray (m,)): target values
          w,b (scalar)    : model parameters  
    
        Returns
            total_cost (float): The cost of using w,b as the parameters for linear regression
                   to fit the data points in x and y
        """
        # number of training examples
        m = x.shape[0] 
    
        cost_sum = 0 
        for i in range(m): 
            f_wb = w * x[i] + b   
            cost = (f_wb - y[i]) ** 2  
            cost_sum = cost_sum + cost  
        total_cost = (1 / (2 * m)) * cost_sum  
    
        return total_cost
    '''
    # 训练数据
    x_train = np.array([1.0, 2.0])
    y_train = np.array([250.0, 500.0])
    # 窗口和子图
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(bottom=0.25)
    # 初始w, b的值
    initial_w = 0.0
    initial_b = 0.0
    # 预测值
    '''fx = ax + b'''
    y = initial_w * x_train + initial_b
    line, = ax[0].plot(x_train, y, c='b', label='Prediction')
    ax[0].scatter(x_train, y_train, marker='x', c='r')
    ax[0].legend()
    # 二次函数范围
    w_range = np.arange(-50, 501, 1)
    b_range = np.arange(-50, 500, 1)
    y_range = compute(x_train, y_train, w_range, b_range)
    # 创建图像
    lin = ax[1].plot(w_range, y_range)
    ax[1].set_xlabel('W')
    ax[1].set_ylabel('dis')
    ax[1].legend(['f = 1/m * (y_i - y) ^ 2'])
    # 滑块
    slider_ax_w = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_w = Slider(slider_ax_w, 'w', -50, 500, valinit=initial_w, valstep=0.1)
    slider_ax_b = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider_b = Slider(slider_ax_b, 'b', -50, 500, valinit=initial_b, valstep=0.1)

    slider_w.on_changed(update_w)
    slider_b.on_changed(update_b)

    # 创建3维图像
    ax_3d = fig.add_subplot(133, projection='3d')
    W, B = np.meshgrid(w_range, b_range)
    ax_3d.plot_surface(W, B, y_range.T, cmap='viridis')  # 注意要对y_range进行转置

    ax_3d.set_xlabel('w_values')
    ax_3d.set_ylabel('b_values')
    ax_3d.set_zlabel('dis_values')


    plt.show()

draw_Linear()