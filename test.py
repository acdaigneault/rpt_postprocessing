from scipy.optimize import least_squares
import numpy as np




def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    rng = np.random.default_rng(seed)
    y = a + b * np.exp(t * c)
    error = noise * rng.standard_normal(t.size)
    outliers = rng.integers(0, t.size, n_outliers)
    error[outliers] *= 10
    return y + error


a = 0.5
b = 2.0
c = -1
t_min = 0
t_max = 10
n_points = 15


t_train = np.linspace(t_min, t_max, n_points)

y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)



def fun(x, t, y):

    return x[0] + x[1] * np.exp(x[2] * t) - y


x0 = np.array([1.0, 1.0, 0.0])

res_lsq = least_squares(fun, x0, args=(t_train, y_train))

res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,

                            args=(t_train, y_train))

res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,

                        args=(t_train, y_train))


t_test = np.linspace(t_min, t_max, n_points * 10)
y_true = gen_data(t_test, a, b, c)
y_lsq = gen_data(t_test, *res_lsq.x)
y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
y_log = gen_data(t_test, *res_log.x)


import matplotlib.pyplot as plt

plt.plot(t_train, y_train, 'o')
plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
plt.plot(t_test, y_lsq, label='linear loss')
plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
plt.plot(t_test, y_log, label='cauchy loss')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()