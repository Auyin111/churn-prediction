# Churn customer prediction #
Using embedding combine the categorical data and numerical data to predict which customers are about to leave your service soon <br>
Then, we can develop proper strategy to re-engage them before it is too late <br>

## Performance and further improvement ##
### Baseline: Without class weight and oversampling ###
After testing different kind of parameters such as optimizer, dataloader, loss function, model and model structure. The best model is found by comparing ***best average cross validation loss***<br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/accuracy_of_training_and_validation_curve_1.png" width="70%" height="70%"> <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/accuracy_of_training_and_validation_curve_2.png" width="35%" height="35%"> <br><br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/loss_of_training_and_validation_curve_1.png" width="70%" height="70%"> <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/loss_of_training_and_validation_curve_2.png" width="35%" height="35%"> <br>

Compare the training and validation curve in tensorboard, I found that the ***best model*** performance ***(loss and accuracy)*** of training and validation are very close to each other. Hence, the model almost learned all the thing from training set.<br>

***Test set classification report***<br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/classification_report__test_set.png" width="50%" height="50%"> <br>
The f1-score of 'Not exited' is quite good but the recall of 'Exited' can not perform well <br><br>
***Test set confusion matrix***<br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/confusion_matrix_test_set_without_class_weight.png" width="50%" height="50%"> <br>
58.48% of "Exited" are predicted as "Not exited".  

***Training validation set classification report***<br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/classification_report__train_set.png" width="50%" height="50%"> <br><br>
As the cross validation result and performance of classification report are very similar, it can prove that the bad recall of 'Exited' is not caused by Overfitting and it should be caused by Underfitting. <br>
The Underfitting should be cause by imbalance dataset, lack of enough training data or the current features are not able to predict the result. <br> <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/dataset_detail.png" width="50%" height="50%"> <br>

### Assign class weight and oversampling ###
Using ***Max. f1 core in stead of Min. loss*** to find the best model <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/f1_of_training_and_validation_curve_1.png" width="100%" height="100%"> <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/f1_of_training_and_validation_curve_2.png" width="50%" height="50%"> <br> <br>
***Test set confusion matrix (with class weight)***<br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/classification_report__test_set_with_class_weight.png" width="80%" height="80%"> <br> <br>
***Test set classification report (with class weight)***<br>
The recall of 'Exited' is improved but the precision reduce. <br><br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/confusion_matrix_test_set_with_class_weight.png" width="80%" height="80%"> <br>
37.10% of "Exited" are predicted as "Not exited" (already reduce 21.38%)

### Conclusion ###
1) A greater number of data should be collected, and more useful features <br>
2) Use different parameters to tune model
    - model structure, optimizer, batch_size etc.
    - to reduce the impact of imbalance dataset: tune class weight and oversampling
3) In this business case, the recall of 'Exited' is much more important than precision of 'Exited'
    - Low precision of 'Not exited' will increase the promotion cost when we re-engage customer
    - But low recall of 'Exited' will loss the customer

## Model tuning ##
Allow to cross validate different kind of parameters easily in a dictionary, such as optimizer, dataloader, loss function, model and model structure <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/declare_tuning_parmas.png" width="40%" height="40%"> <br>

The cross validation training and validation curve will be stored and display on tensorboard ***instantly*** so you can stop a model if you find the trend is not good in any time <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/tsboard_demo.png" width="60%" height="60%"> <br>

Also, a df will show the CV performance of all parameter’s combinations.
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/cross_validation_performance.png" width="100%" height="100%"> <br>

## Technical term description ##
***Embeddings*** <br>
--> reduce the dimensionality of categorical variables and meaningfully represent categories in the transformed space <br>

***Batch normalization*** <br>
--> reduce training time and make trainable  (less Covariate Shift and less vanishing gradients)

***Dropout*** <br>
(ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random) <br>
--> avoid curbs the individual power of each neuron (prevent over-fitting) <br>

***Early stopping*** <br>
if not stop early, the model will overfit and generate a larger loss <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/train_valid_curve_expectation.png" width="50%" height="50%"> <br>

***TensorBoard*** <br>
is a tool for providing the (***instant***) measurements and visualizations needed during the machine learning workflow <br>

***Classification report*** <br>
1. Precision: tp / actual result = tp / (tp + fp)   <br>
    - How many selected items are relevant?
2. Recall: tp / predicted result = tp/(tp + fn)   <br>
    - How many relevant items are selected?
3. F1 = 2 * (precision * recall) / (precision + recall) <br>
4. Support is the number of samples of the true response <br>
5. Remarks: <br>
    - fp (False positive):  it is negative but predicted as positive <br>
    - fn (False negative):  it is positive but predicted as negative <br>
    - tp (True positive) <br>
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/precision_and_recall.png" width="20%" height="20%"> <br>

***StratifiedKFold*** <br>
As it is a imbalance dataset, using stratified k fold can have a fair validation and testing
<img src="https://github.com/Auyin111/churn-prediction/blob/master/readme%20photo/StratifiedKFold.png" width="40%" height="40%"> <br>

## Open TensorBoard through the command line ##
    tensorboard --logdir= YOUR PATH (Default: runs)

## Churn Modelling Dataset ##
[Kaggle dataset](https://www.kaggle.com/c/churn-modelling)


