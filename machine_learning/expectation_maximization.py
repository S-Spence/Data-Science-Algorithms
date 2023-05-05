"""
This file contains an algorithm for the simple expectation maximization.
"""
import numpy as np
import math

# implement expectation maximization algorithm
def simple_expectation_maximization(x, k, prob, m, std, threshold):
    """
    Expectation Maximization algorithm
    Inputs:
        - X: training data
        - k: number of clusters (equal to num classes)
        - prob: starting probability (numpy array)
        - m: starting means (numpy array)
        - std: starting standard deviation (numpy array)
        - threshold: float
    """
    cols = x.shape[1]
    rows = x.shape[0]
    
    initializeRows = np.ones([rows, 1])
    initializeCols = np.ones([1, cols])
 
    iterations = 0

    while True:
        prev_mean = np.copy(m)
        prev_std = np.copy(std)
        prev_prob = prob
        
        g = []
        probKg = []
        """
        Expectation step
        """
        for cluster in range(k):
            val = np.reshape(m[:, cluster], (len(m[:,cluster]), 1))
            calc_1 = np.matmul(val, np.ones([1,cols])) 
            calc_2 = x - calc_1
            calc_3 = np.power(calc_2, 2)
            calc_4 = np.round(np.exp(np.divide(np.divide(-1 * np.sum(calc_3, axis = 0), np.power(std[cluster], 2)), 2)), 4)
            calc_5 = np.power(math.sqrt(2 * math.pi)* std[cluster], rows)
            calc_e_step = np.divide(calc_4, calc_5)
            g.append(calc_e_step)
            probKg.append(prob[:, cluster] * g[cluster])
        
        # convert to numpy arrays for calculations
        g = np.array(g)
        probKg = np.array(probKg)
        # Reshape the sum of probKg
        reshaped_calc = np.reshape(np.sum(probKg, axis=0), (1, cols))

        prob_ikn = np.divide(probKg, (np.matmul(np.ones([k, 1]), reshaped_calc)))

        """
        Maximization step
        """
        sum_prob_ikn = np.sum(prob_ikn, axis=1)
        
        for cluster in range(k):
            cols = prob_ikn[cluster, :].shape[0]
            val = np.reshape(prob_ikn[cluster, :], (1, cols))
            calc_1 = np.matmul(initializeRows, val)
            calc_2 = np.multiply(calc_1, x)
            calc_3 = np.divide(calc_2, sum_prob_ikn[cluster])
            m[:, cluster] = np.sum(calc_3, axis=1)
            
            calc_1 = np.multiply(np.reshape(m[:, cluster], (len(m[:, cluster]),1)), initializeCols)
            calc_2 = np.sum(np.power(x - calc_1, 2), axis=0)
            calc_3 = np.sum(np.multiply(calc_2, prob_ikn[cluster, :]), axis=0)
            calc_4 = np.divide(calc_3, sum_prob_ikn[cluster])
            calc_5 = np.round(np.divide(calc_4, rows), 4)
            std[cluster] = math.sqrt(calc_5)
            
        prob = np.divide(sum_prob_ikn, np.sum(sum_prob_ikn))
        prob = np.reshape(prob, (1, len(prob)))
        
        iterations += 1
        
        # Determine convergence
        mean_delta = max(np.sqrt(np.sum(np.power(m - prev_mean, 2), axis=0)))
        s_mean = np.mean(np.sqrt(np.sum(np.power(m, 2), axis=0)))
        conv_mean = mean_delta <= (s_mean * threshold)

        std_delta = math.sqrt(np.sum(np.power(std - prev_std, 2), axis=0))
        s_std = np.mean(np.sqrt(np.sum(np.power(std, 2))))
        conv_std = std_delta <= (s_std * threshold)

        prob_delta = max(np.sqrt(np.sum(np.power(prob - prev_prob, 2), axis=0)))
        s_prob = np.mean(math.sqrt(np.sum(np.power(prob, 2))))
        conv_prob = prob_delta <= (s_prob * threshold)

        # If the mean, std, and probabiity have converged, break
        if conv_mean and conv_std  and conv_prob:
            break

    return prob, m, std, prob_ikn, iterations

def prepare_data_EM(data, num_clusters, random_values):
    """
    A helper function to prepare the data for the simple
    expectation maximization algorithm. 
    random_values is a numpy array of length
    num_clusters.
    """
    k = num_clusters
    x = np.array(data)
    
    # get mean across x axis
    m = np.mean(x, axis = 0).T
    m = np.reshape(m, (len(m), 1))
    
    # get std across x axis (ddof = 1 to match matlab calculations)
    std = np.std(x, axis=0, ddof=1).T
    std = np.reshape(std, (len(std), 1))

    initializeK = np.ones([1, k])
    rnd = np.reshape(random_values, (1, k))
    m = np.matmul(m, initializeK) + np.matmul(std, rnd)
    std = np.matmul(np.mean(std, axis=0),initializeK)
    prob = initializeK/k
    
    return k, x, m, std, prob

def get_clusters(prob_ikn, num_clusters):
    y_pred = list(prob_ikn.argmax(axis=0))

    clusters = np.array(y_pred)
    
    output_clusters = []
    
    for cluster in range(num_clusters):
        output_clusters.append(np.where(clusters == cluster)[0])
        
    return output_clusters