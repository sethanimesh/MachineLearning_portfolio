### Importing libraries


```python
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

### Spliting training and testing 


```python
df = pd.read_csv('data/salary_dataset.csv')
feature = ['YearsExperience']

X_train, X_test, y_train, y_test = train_test_split(df[feature], df['Salary'], random_state=42, test_size=.2)
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values
X_train.shape

```




    (24, 1)



### Plotting the data


```python
plt.scatter(X_train, y_train, marker='x', c='r')
plt.ylabel("Salary")
plt.xlabel("Years of Experience")
plt.show()
```


    
![png](output_5_0.png)
    


### Compute Cost


```python
def compute_cost(x, y, w, b):
    m = x.shape[0]
    
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1/(2*m))*cost_sum

    return total_cost
```

### Gradient Descent


```python
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w*x[i]+b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db
```


```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    
    #an array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        #calculate the gadient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        #update parameters 
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        
        #save cost J at each iteration
        if i<100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
            
        
    
    return w, b, J_history, p_history
```


```python
#initialise parameters
w_init = 0
b_init = 0

#some gradients descent settings
iterations = 100000
tmp_alpha = 1.0e-3

#run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(X_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"(w, b) found by gradient descent: ({w_final.item():8.4f}, {b_final.item():8.4f})")
```

    (w, b) found by gradient descent: (9423.8153, 24380.2015)



```python
#plot cost versus iterations
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000+np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iterations(start)"); ax2.set_title("Cost vs. iterations(end)")
ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step'); ax2.set_xlabel('iteration step')
plt.show()
```


    
![png](output_12_0.png)
    


### Predictions


```python
y_pred = w_final*X_test + b_final
y_pred
```




    array([[115791.21011463],
           [ 71499.27809353],
           [102597.86866153],
           [ 75268.80422298],
           [ 55478.79204334],
           [ 60190.69970516]])



### Accuracy


```python
#Mean absolute error
mae = np.mean(np.abs(y_pred-y_test))
print(f'Mean Absolute Error: {mae:.4f}')
```

    Mean Absolute Error: 25131.9846



```python
mse = np.mean((y_pred - y_test) ** 2)
r_squared = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r_squared:.4f}')
```

    Mean Squared Error: 1000068735.2498
    R-squared: -10.7472



```python

```
