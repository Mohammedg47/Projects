import math, copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    file = pd.read_csv('../Data/Salary_dataset.csv')
    

    y_train = file['Salary'].to_numpy()
    x_train = file['YearsExperience'].to_numpy()

    w_init = 0
    b_init = 0

    iterations = 10000
    tmp_alpha = 1.0e-2

    w_final, b_final = gradientDescent(x_train, y_train, w_init, b_init, tmp_alpha, 
                                        iterations, computeCost, computeGradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    # plot
    predictions = w_final * x_train + b_final

    plt.scatter(x_train, y_train, marker='o', c='r', label='Actual Salary')
    plt.plot(x_train, predictions, c='b', label=f'Prediction (w={w_final:.2f}, b={b_final:.2f})')
    plt.xlabel('Years of experiance')
    plt.ylabel('Salary')
    plt.title('Salary Prediction')
    plt.legend()
    plt.show()


def computeCost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    totalCost = (1/(2*m)) * cost
    return totalCost

def computeGradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

def gradientDescent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        if i<100000:
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])

        if i% math.ceil(num_iters/10) == 0:

            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b

if __name__ == '__main__':
    main()