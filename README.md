# PD-Health

# Parkinson's Disease Severity Prediction using Deep Learning

## Introduction

### Purpose
Parkinson's Disease Severity Prediction using Deep Learning is a project aimed at predicting the severity of Parkinson's disease in patients based on the analysis of voice data.

### Scope
The scope of the project includes recognizing the severity of Parkinson's disease by analyzing changes in patients' voices over a certain period.

### Definitions, Acronyms, Abbreviations
- **DNN**: Deep Neural Network
- **UPDRS**: Unified Parkinson's Disease Rating Scale
- **Total UPDRS**: Total Unified Parkinson's Disease Rating Scale
- **Motor UPDRS**: Motor Unified Parkinson's Disease Rating Scale

### Detailed Problem Definition
The project employs a deep learning approach to predict the severity of Parkinson's disease. The methodology involves collecting voice data, normalizing it, and designing a deep neural network for training and testing.

### References
- [Das R. (2010) "A comparison of multiple classification methods for diagnosis of Parkinson disease". Expert Systems With Applications; 37:1568-1572.](#)
- [Genain N, Huberth M, Vidyashankar R. (2014) "Predicting Parkinson’s Disease Severity from Patient Voice Features."](#)
- [Benmalek E, Elmhamdi J, Jilbab A. (2015) "UPDRS tracking using linear regression and neural network for Parkinson’s disease prediction." International Journal Of Emerging Trends & Technology In Computer Science (IJETTCS); 4:189-193.](#)

## The Overall Description

### External Interface Requirements

#### System Interfaces
- **Anaconda Spyder**: Scientific Python Development Environment for development and testing.

#### Software Interfaces
- **Python**: Interpreted, high-level programming language used for implementation.
- **Android Studio**: Integrated development environment for Android development.

#### Hardware Interfaces
The project is implemented on a system with Intel Core i5-5200U CPU @2.20GHz and 8 GB RAM.

#### Communication Interfaces
- **Media Recorder API**: Used for capturing and encoding audio if supported by the device hardware.

### System Features

With the help of 16 biomedical factors, the project analyzes the values of Motor UPDRS and Total UPDRS, including various voice measures such as Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE.

---

## Specific Requirements

### Performance Requirements

The input dataset comprises 16 biomedical voice features, and the output variable is Total UPDRS score. The project evaluates the motor UPDRS and total UPDRS scores, providing a scale for severity prediction.

### Safety Requirements

To ensure the predictions are valid, input parameters should be within the given range after normalization.

### Software Quality Attributes

- **Correctness**: The algorithm predicts appropriate values for the complete dataset with high accuracy.
- **Extensibility**: The algorithm can adapt to different datasets.
- **Learnability**: The DNN continuously updates its training data, improving accuracy and performance.
- **Efficiency**: The DNN with three hidden layers and the classification algorithm used provides high accuracy and efficiency.

