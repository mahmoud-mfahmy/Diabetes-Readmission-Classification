# Diabetes-Readmission-Classification
Using classification methods to predict hospital readmission for diabetic patients

## Summary:
* Dataset consists of 10 years of diabetic patients' data in the U.S., along with whether or they not they were readmitted to the hospital (Unbalanced classes with ~75% not readmitted)
* Data was cleaned and many of the features were recoded/engineered
* Multiple classification models were compared using R (Logistic Regression, SVM, Decision Tree, Random Forest) on the basis of AUC and K-Means Clustering was performed as well
* Logistic regression was found to be the best performing model (AUC = 0.615)

## References:
* [Research Paper from dataset creators](https://www.hindawi.com/journals/bmri/2014/781670/)

### Data Cleaning:
- The initial dataset has 101,766 rows and 50 attributes. First, we checked for missing values upon importing the data into R. 

![missing data](https://user-images.githubusercontent.com/76078425/102843070-6d417200-43d6-11eb-8235-166bee42d47c.jpg) 

- Weight, payer_code, and medical_specialty were dropped since they were missing > 40% of their data 
- Dropped only the missing rows for other columns 
- Dropped one row which had a value for gender = "Invalid"
- Kept only each patient's first hospital visit, dropped any follow-up visits

### Feature Engineering:

- Created buckets for many categorical features to reduce number of distinct values
- Combined 3 features (number of outpatient, inpatient, and emergency room visits) into 1 feature called "total visits"
- Dropped or recoded most features describing medication due to lack of variance
- Converted "age" variable from 10 age groups to just 3: {[0,30), [30,60), >60}
- Recoded target variable (Readmitted) from 3 values (>30 day readmission, <30 day readmission, No readmission) into a binary ("Readmission", "No Readmission")

### Logistic Regression:

- Undersampling used to balance data while maintaining independence of samples
``` R
backward <- step(lr_all, direction = 'backward', scope = list(lower=lr_base, 
                                                              upper = lr_all), trace = 1)                                                          
```                                                              





![LogReg_ROC](https://user-images.githubusercontent.com/76078425/102842982-41be8780-43d6-11eb-8871-f7eec3157d55.jpg)