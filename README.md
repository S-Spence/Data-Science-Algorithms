# Data-Science-Algorithms
This repository contains various Machine Learning, Computational Statistics, and AI algorithms from an Algorithms course in my graduate program.There are several algorithm files in the data_processing, machine_learning, and game theory directories demonstrating custom implementations in Python. The jupyter notebooks demonstrate the usage of these algorithms. The only implementations in this repository that used packages beyond numpy and pandas to implement the algorithms were PCA and the Feed Forward Neural Network example using Keras. 

## Algorithm Sample Files

- Spence_PA1: This file demonstrates data manipulation for analysis, data normalization, outlier removal, feature ranking, principal component analysis, and game theory. This file includes samples of the following algorithms:
    - Mahalanobis Distance (outlier removal)
    - Bhattacharyya distance (feature ranking)
    - Fischer's Linear Discriminant Ratio (feature ranking)

- Spence_PA2: This file demonstrates Expectation Maximization, Bayes Classifier, Feed Forward Neural Networks, Parzen Window, and the MiniMax Algorithm applied to TicTacToe. This file includes samples of the following algorithms:
    - Expectation Maximization (ML: clustering)
    - Bayes Classifier (ML: classification)
    - Feed Forward Neural Network (ML: classification)
    - K-Fold Validation (ML: sampling)
    - Parzen Windows (ML: Classification)
    - MiniMax Algorithm (Game Theory)

- Spence_HW2: This file demonstrates data analysis, outlier removal, data transformations, and dimensionality reduction. This file includes samples of the following algorithms:
    - Mahalanobis Distance (outlier removal)
    - Discrete Cosine Transform (data transformation)
    - Eigen Decomposition (dimensionality reduction)
    
- Spence_HW3: This file demonstrates feature ranking, k-fold validation, and the parzen window algorithm for classification and plotting kernels. This file includes samples of the following algorithms:
    - Fischer's Discriminant Ratio (feature ranking)
    - K-Fold Validation (sampling)
    - Parzen Windows (ML: classification)
    
 - Spence_HW4: This file includes a custom implementation of a Radial Basic Function Neural network and a manual walkthrough of the MiniMax algorithm used in Spence_PA2.ipynb. This file includes samples of the following algorithms:
    - Radial Basis Function Neural Network (classification)
    - MiniMax Algorithm (game theory)
    
### TODO:
- Refactor Spence_HW3.ipynb and Spence_HW4.ipynb to use the abstracted functions in the algorithm files.
- Add more detailed descriptions about the algorithms to the README and the algorithm files.
- Add more tests

