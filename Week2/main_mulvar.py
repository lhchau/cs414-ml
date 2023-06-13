import numpy as np

feature_size = 2
data_size = 500
x_ori = np.random.normal(loc=0, scale=1, size=(feature_size, data_size))
x = np.concatenate([np.ones((1, data_size)), x_ori])

theta_ori = np.arange(start=4, stop=4+feature_size)
y = theta_ori.T @ x_ori + 3
y = y.squeeze()

theta = np.random.random(size=(feature_size+1))
# learning_rate = 1/np.mean(x**2)
learning_rate = 0.01
tolerance = 0.01
cnt = 0
while True:
    cnt += 1
    dtheta = np.mean(x * (theta.T @ x - y), axis=1)
    theta = theta - learning_rate * dtheta
    if (abs(dtheta) - tolerance < 0).all():
        break    
    
print(cnt)
print(theta)