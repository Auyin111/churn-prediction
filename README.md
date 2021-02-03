## Churn customer prediction
Using embedding combine the categorical data and numerical data to predict whether the customer is Exited <br>
It can be use to find the churn customer and try to keep them in the future <br>

## Performance
After test diff. kind of parmas., a best model is found by comparing best validation loss<br>
Finally, the corresponding classification report is created <br>
![performance](readme%20photo/performance.png)<br><br>

## Model tuning
Allow to test diff. kind of parmas. easily in a dictionary, such as <br>
optimizer, dataloader, loss function, model structure and parmas <br>
![declare_tuning_parmas](readme%20photo/declare_tuning_parmas.png) <br><br>

The training and validation curve will be stored and display on tensorboard ***instantly*** <br>
you can stop a model if you find the trend is not good in any time <br>
![train_valid_curve](readme%20photo/train_valid_curve.png) <br><br>
![train_valid_curve2](readme%20photo/train_valid_curve2.png) <br><br>

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
![train_valid_curve_expectation](readme%20photo/train_valid_curve_expectation.png) <br><br>
***TensorBoard*** <br>
is a tool for providing the (***instant***) measurements and visualizations needed during the machine learning workflow <br>

***Classification report*** <br>
Precision: tp / actual result = tp / (tp + fp)   <br>
Recall: tp / predicted result = tp/(tp + fn)  <br>
F1 = 2 * (precision * recall) / (precision + recall) <br>
support is the number of samples of the true response <br>

## Open TensorBoard through the command line
    tensorboard --logdir= YOUR PATH

## Churn Modelling Dataset
[Kaggle dataset](https://www.kaggle.com/c/churn-modelling)


