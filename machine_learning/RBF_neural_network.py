"""
Radial Basis Function Neural Network
"""
import numpy as np
import common_helpers as common

def RBF_NN_Train(data: np.array, targets: np.array, spread: float) -> dict:
    """
    Develop a training model for each class in the dataset using a 1 vs. all approach
    """
    model_output = {}
    observations, features = np.shape(data)
    H = np.zeros((observations, observations))
    for i in range(observations):
        for j in range(observations):
            weight = data[i, :]
            H[j, i] = np.exp(-(np.matmul((data[j, :]-weight), (data[j, :]-weight).T))/ (2 * np.power(spread, 2)))
    
    w_hat = np.matmul(np.matmul(np.linalg.pinv(np.matmul(H.T, H)), H.T), targets)
    yt = np.matmul(H, w_hat).T
    
    y_pred = np.ones(len(targets))
    
    # Get all indices with negative predictions
    y_pred_indices = np.where(yt<0)[0]
    # Update all indices where a prediction was negative
    np.put(y_pred, y_pred_indices, -1)
    
    correct_pred = 0
    for i, val in enumerate(y_pred):
        if y_pred[i] == targets[i]:
            correct_pred += 1
    pred_error = 1 - (correct_pred/len(targets))
    
    model_output["w_hat"] = w_hat
    model_output["w"] = data
    model_output["spread"] = spread
    model_output["error"] = pred_error
    
    return model_output

def RBF_NN_Classify(data: np.array, model: dict) -> list:
    """
    Classify observations in the dataset using a 1 vs. all method per class
    """
    obs_1, features_1 = np.shape(data)
    obs_2, features_2 = np.shape(model["w"])
    
    H = np.zeros((obs_1, obs_2))
    
    for i in range(obs_2):
        for j in range(obs_1):
            w = model["w"][i, :]
            H[j, i] = np.exp(-(np.matmul((data[j, :]-w), (data[j, :]-w).T))/ (2 * np.power(model["spread"], 2)))
            
    y = np.matmul(H, model["w_hat"]).T
    y_hat = np.zeros((1, obs_1))

    for i in range(obs_1):
        for j in range(obs_2):
            w = model["w"][j, :]
            w_hat = model["w_hat"][j]
            y_hat[0][i] = y_hat[0][i] + w_hat * np.exp(-(np.matmul((data[i, :]-w), (data[i, :]-w).T))/ (2 * np.power(model["spread"], 2)))
    
    # Create an array of 1s the length of the data
    y_pred = np.ones(len(y))
    
    # Get all indices with negative predictions
    y_pred_indices = np.where(y<0)[0]
    # Update all indices where a prediction was negative
    np.put(y_pred, y_pred_indices, -1)
    
    return y, y_pred

def RBF_NN_Test_Function(data: np.array, models: list, labels: list) -> list:
    """
    Pass in a list of models trained with 1 vs. all for each class in the dataset
    Pass in the testing labels to calculate accuracy
    Call the RBF_NN_Classify function to perform classifications on the dataset for each class
    Combine predictions for all classes
    Calculate accuracy
    
    Input:
        - data: np array of the dataset
        - models: a list of trained models
    Returns:
        - Predictions: an array of predictions
        - Accuracy: the model accuracy 
    """
    y_list = []
    y_pred_list = []
    
    for model in models:
        y, y_pred = RBF_NN_Classify(data, model)
        y_list.append(y)
        y_pred_list.append(y_pred)
    
    temp = np.array([])
    for i in range(len(y_list)):
        temp = common.stack_columns(temp, y_list[i])
    
    temp= temp.T
    
    y_pred = list(temp.argmax(axis=0))
    
    correct_pred = 0
    for i, val in enumerate(y_pred):
        if y_pred[i] == labels[i]:
            correct_pred += 1
    accuracy = (correct_pred/len(data)*100)
    
    return y_pred, accuracy