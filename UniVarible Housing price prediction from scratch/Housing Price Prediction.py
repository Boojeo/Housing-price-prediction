import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# import the data
data = pd.read_csv("Housing.csv")
X = data["area"].values
y= data["price"].values


# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1,1)).flatten()

# Use manual normalization for learning
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

# Split the data into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled, test_size=0.2,random_state=43)

# Compute cost function
def compute_cost(X, y , w,b) :
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f = w * X[i] +b
        cost += (f - y[i])**2
    cost /=  (2*m)

    return cost

#compute gradient function
def compute_gradient(X, y , w,b) :
    djdw =0
    djdb =0

    m = X.shape[0]

    for i in range(m):
        f = w * X[i] +b
        djdw += (f - y[i]) * X[i]
        djdb += (f- y[i])
    djdw /= m
    djdb /= m

    return djdw, djdb

def gradient_descent(X,y,w,b, alpha, iterations):

    cost_history = []
    for i in range(iterations): 
        djdw , djdb = compute_gradient(X,y,w,b)
        w = w - alpha * djdw
        b = b - alpha * djdb
        cost_history.append(compute_cost(X, y, w, b))

    return w,b, cost_history

# predict target using output of gradient descent w,b 

def predict_scaled(X, w, b):
    return np.dot(X, w) + b

def predict(X, w, b, y_mean, y_std):
    y_scaled = np.dot(X, w) + b
    return y_scaled * y_std + y_mean

#checking cost before training
print(f"Cost befor train {compute_cost(X_train, y_train, 0, 0)}")



# Train the model
w = 0 
b = 0
iterations = 2000
alpha = 0.01
w, b,  cost_history = gradient_descent(X_train,y_train,w,b,alpha=alpha, iterations=iterations)


# training and test cost after training
print(f"Training cost: {compute_cost(X_train, y_train, w, b)}")
print(f"Test cost: {compute_cost(X_test, y_test, w, b)}")


# using matplotlib to see how cost decrease over iterations
plt.plot(range(0, iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

# using matplotlib to check how our plot predict new data

y_pred_original = predict(X_test, w, b, y_mean, y_std)
plt.scatter(X_test * X.std() + X.mean(), y_test * y_std + y_mean, label="Actual")
plt.plot(X_test * X.std() + X.mean(), y_pred_original, color="red", label="Predicted")
plt.xlabel("Area (original)")
plt.ylabel("Price (original)")
plt.legend()
plt.show()


"""
    Review to our model and how we can achive better results:
        Using multivariable model
        Vectorization
        Learning Rate Tuning
        Feature Engineering

"""