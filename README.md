## Churn customer prediction
Using embedding combine the categorical data and numerical data to predict whether the customer is Exited <br>
It can be use to find the churn customer and try to keep them in the future <br>

## Performance
***Training and Validation loss***
![Image 1](train_valid_curve.png) <br><br>

***Without class weight*** <br>
The precision and recall of class 1 is lower thand class 0
![Image 1](test_performance_without_class_weight.png) <br><br>

***With class weight*** <br>
As the number of class 1 data is less than class 0, i assign a class weight [0.8, 1] <br>
The f1-score of class 1 is little bit improved
![Image 1](test_performance_with_class_weight.png) <br><br>


## Technical term description
***Embeddings*** <br>
--> reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space <br>

***Batch normalization*** <br>
--> reduce training time and make tranable  (less Covariate Shift and less vanishing gradients)

***Dropout*** <br>
(ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random) <br>
--> avoid curbs the individual power of each neuron (prevent over-fitting) <br>

***Early stopping*** <br>
if not stop early, the model will overfit and generate a larger loss
![Image 1](train_valid_curve_expectation.png) <br><br>

***TensorBoard*** <br>
is a tool for providing the (INSTANT) measurements and visualizations needed during the machine learning workflow <br>

***Classification report*** <br>
Precision: tp / actual result = tp / (tp + fp)   <br>
Recall: tp / predicted result = tp/(tp + fn)  <br>
F1 = 2 * (precision * recall) / (precision + recall) <br>
support is the number of samples of the true response <br>

## Guideline of TensorBoard
type command: tensorboard --logdir=C:/Users/Auyin/PycharmProjects/churn-prediction/train_valid_log/

## Dataset
https://www.kaggle.com/c/churn-modelling

