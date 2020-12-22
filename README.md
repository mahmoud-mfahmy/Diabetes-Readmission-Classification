# Diabetes-Readmission-Classification
Using classification methods to predict hospital readmission for diabetic patients

## Summary:
* Dataset consists of 10 years of diabetic patients' data in the U.S., along with whether or they not they were readmitted to the hospital
* Data was cleaned and many of the features were recoded/engineered
* Multiple classification models were compared using R (Logistic Regression, SVM, Decision Tree, Random Forest) as well as K-Means Clustering
* Logistic regression was found to be the best model

## References:
* [Research Paper from dataset creators](https://www.hindawi.com/journals/bmri/2014/781670/)

### Data Cleaning:
- The initial dataset has 101,766 rows and 50 attributes. First, we checked for missing values upon importing the data into R. 

![missing data](https://user-images.githubusercontent.com/76078425/102843070-6d417200-43d6-11eb-8235-166bee42d47c.jpg) 

- Weight, payer_code, and medical_specialty were dropped since they were missing > 40% of their data 
- For the other columns, I only dropped the missing rows since they were so few (<2% of dataset)
- I dropped one row which had a value for gender = "Invalid"





