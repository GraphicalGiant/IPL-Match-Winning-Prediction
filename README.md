# IPL-Match-Winning-Prediction
## Aim 
implementing various different models that can be used to predict winning IPL team

##DATASET
This data is from Oracle for educational use in 2018

## DESCRIPTION
The project aims to predict the future winning team in IPL matches using historical data and machine learning techniques.

## WORK DONE

The following tasks were completed as part of the project:

1. Data analysis was conducted, including identifying missing values and separating categorical and continuous data.
2. Unnecessary columns were removed from the dataset.
3. The winning column for a single match was prepared.
4. Feature engineering techniques were applied to transform the team1, team2, and venue data.
5. Exploratory Data Analysis (EDA) was performed, including creating count plots for umpires and Player of the Match.
6. The dataset was split into training and test data.
7. Various machine learning algorithms were trained using the training data, including Logistic Regression, Naive Bayes Classifier, Support Vector Classifier, KNN Classifier, Decision Tree Classifier, Random Forest Classifier, and XGBoost Classifier.
8. The performance of each model was evaluated, and the model with the highest accuracy was determined.

## MODELS USED

1. **Logistic Regression** - Logistic regression is a machine learning algorithm for classification. In this algorithm, the probabilities describing the possible outcomes of a single trial are modelled using a logistic function. It is most useful for understanding the influence of several independent variables on a single outcome variable.
2. **Naive Bayes Classifier** - Naive Bayes algorithm based on Bayes’ theorem with the assumption of independence between every pair of features. This algorithm requires a small amount of training data to estimate the necessary parameters. Naive Bayes classifiers are extremely fast compared to more sophisticated methods.
3. **Support Vector Classifier** - A support vector machine is a representation of the training data as points in space separated into categories by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall. It is effective in high dimensional spaces and uses a subset of training points in the decision function, so it is also memory efficient.
4. **K-nearest neighbors Classifier** - It is a simple algorithm to understand and can be used for classification analysis. Classification is computed from a simple majority vote of the K nearest neighbors of each point. This algorithm is simple to implement, robust to noisy training data, and effective if the training data is large.
5. **Decision Tree Classifier** - Given data of attributes together with its classes, a decision tree produces a sequence of rules that can be used to classify the data. Decision Tree is simple to understand and visualize, requires little data preparation, and can handle both numerical and categorical data.
6. **Random Forest Classifier** - Random forest classifier fits a number of decision trees on various sub-samples of datasets and uses average to improve the predictive accuracy of the model and controls over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement.  It results in a reduction in over-fitting and the random forest classifier is more accurate than decision trees in most cases.
7. **XGBoost Classifier** - XGBoost is a popular gradient-boosting library for GPU training, distributed computing, and parallelization. It’s precise, it adapts well to all types of data and problems, it has excellent documentation, and overall it’s very easy to use. 


## LIBRARIES USED

* Numpy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn



## ACCURACIES

| **Model** | Accuracy | 
| --- | --- |
|1. Logistic Regression | 52.466368 % | 
|2. Naive Bayes |47.085202 % |
|3. Support Vector|53.363229 %|
|4. K Nearest Neighbours|52.914798 % |
|5. Decision Tree|54.708520 % |
|6. Random Forest |55.156951 % |
|7. XGBoost |49.327354 % |

##CONCLUSION
The Random Forest Classifier achieved the highest accuracy among the evaluated algorithms, with KNN and Decision Tree following closely behind. This project provided an opportunity to gain hands-on experience in applying different classification algorithms. Although the accuracy may be slightly lower, it is important to note that winning in the IPL is influenced by numerous factors beyond statistical analysis.
