## Churn customer prediction
Using embedding combine the categorical data and numerical data to predict which customers are about to leave your service soon <br>
Then, we can develop proper strategy to re-engage them before it is too late <br>

## Performance
After test diff. kind of parmas., a best model is found by comparing best ***average*** CV validation loss<br>
Finally, the corresponding ***test set*** classification report is created <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/classification_report__test_set.png" width="50%" height="50%"> <br>
The f1-score of 'Not exited' is quite good but the recall of 'Exited' can not perform well <br><br>

Then, I view the CV training and validation curve to find the reason <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/accuracy_of_training_and_validation_curve.png" width="80%" height="80%"> <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/loss_of_training_and_validation_curve.png" width="80%" height="80%"> <br>

Compare the training and validation curve in tensorboard, I found that the ***best model*** performance ***(loss and accuracy)*** of training and validation are very close to each other <br>
It should be cause by lack of enough training data or the features are not enough to predict the result. <br>

Finally, I created ***train set*** classification report and compare to ***test set*** classification report <br>
So, it can prove that the bad recall of 'Exited' is not come from Overfitting but come from Underfitting.
Hence, more data should be collect and more features should be defined. <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/classification_report__train_set.png" width="50%" height="50%"> <br>

## Model tuning
Allow to test diff. kind of parmas. easily in a dictionary, such as <br>
optimizer, dataloader, loss function, model structure and parmas <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/declare_tuning_parmas.png" width="40%" height="40%"> <br>

The cross validation training and validation curve will be stored and display on tensorboard ***instantly*** <br>
you can stop a model if you find the trend is not good in any time <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/tsboard_demo.png" width="60%" height="60%"> <br>

Also, a df will show the model best epoch average loss <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/cross_validation_performance.png" width="60%" height="60%"> <br>


## Technical term description
***Embeddings*** <br>
--> reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space <br>

***Batch normalization*** <br>
--> reduce training time and make tranable  (less Covariate Shift and less vanishing gradients)

***Dropout*** <br>
(ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random) <br>
--> avoid curbs the individual power of each neuron (prevent over-fitting) <br>

***Early stopping*** <br>
if not stop early, the model will overfit and generate a larger loss <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/train_valid_curve_expectation.png" width="50%" height="50%"> <br>
***TensorBoard*** <br>
is a tool for providing the (***instant***) measurements and visualizations needed during the machine learning workflow <br>

***Classification report*** <br>
Precision: tp / actual result = tp / (tp + fp)   <br>
Recall: tp / predicted result = tp/(tp + fn)  <br>
F1 = 2 * (precision * recall) / (precision + recall) <br>
support is the number of samples of the true response <br>

***StratifiedKFold*** <br>
As it is a imbalance dataset, using stratified k fold can have a fair validation and testing
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/StratifiedKFold.png" width="40%" height="40%"> <br>

## Open TensorBoard through the command line
    tensorboard --logdir= YOUR PATH

## Churn Modelling Dataset
[Kaggle dataset](https://www.kaggle.com/c/churn-modelling)


