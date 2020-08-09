def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    accuracy = float(len(prediction[ground_truth == prediction]) / len(ground_truth))

    #presicion = 
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    accuracy = float(len(prediction[ground_truth == prediction]) / len(ground_truth))
    return accuracy

import numpy as np

result = multiclass_accuracy(np.array([1,0,1,1,0,0,0,0,0,0]), np.array([1,0,1,1,0,1,1,1,1,1]))

print(result)