#### <div align="center"> <h1> Claims_NaiveBayes </h1> </div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/d5ba4bc4-2ce1-436e-bca6-281de7ca9dfc" width="350"/>
</p>

This project focuses on the application of the Naive Bayes algorithm to analyze and predict fraudulent insurance claims. The dataset used contains various attributes related to insurance claims, providing a comprehensive view for both exploratory data analysis and predictive modeling.


### Objectives:

The objective of this project is to identify key indicators of fraudulent insurance claims by analyzing the dataset and applying the Naive Bayes algorithm for classification. We aim to build an effective model to distinguish between fraudulent and legitimate claims, evaluate its performance using accuracy, precision, recall, and F1-score, and provide actionable insights to enhance fraud detection systems. Additionally, the project includes data cleaning, exploratory data analysis, feature engineering, and comprehensive documentation to ensure clarity and reproducibility.

### Key Findings and Insights

- **Fraudulent Patterns:** Certain incident causes, claim areas, and claim types are more frequently associated with fraudulent claims.

- **Police Reports:** Claims with a police report tend to have a higher likelihood of being legitimate.

- **Claim Amount:** Higher claim amounts are often scrutinized more and have a slightly higher rate of fraud detection.

- **Total Policy Claims:** Customers with a higher number of total policy claims have a different fraud probability compared to those with fewer claims.


## Dataset Description

The dataset consists of the following variables:

- **claim_id:** Unique identifier for each claim.

- **customer_id:** Unique identifier for each customer.

- **incident_cause:** The cause of the incident leading to the claim.

- **claim_date:** Date when the claim was filed.

- **claim_area:** The area where the claim incident occurred.

- **police_report:** Indicator of whether a police report was filed (Yes/No).

- **claim_type:** Type of claim (e.g., theft, accident, etc.).

- **claim_amount:** The monetary amount claimed.

- **total_policy_claims:** Total number of claims made by the policyholder.

- **fraudulent:** Indicator of whether the claim is fraudulent (Yes/No).

## Let's Get Started :

**Import all the necessary modules in the Jupyter Notebook.**

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt          

import datetime as dt 

import scipy.stats as stats

import statsmodels.formula.api as smf


**Import the dataset in the Jupyter Notebook**

     Claims = pd.read_csv('claims (1).csv')
     Claims.head()

![image](https://github.com/user-attachments/assets/0f4f22b3-89ab-472c-813d-79ea0e99722f)

**There are 1100 rows and 10 variables in our DataFrame.**

#### Checking the info of our DataFrame.

     Claims.info()

![image](https://github.com/user-attachments/assets/b0036fee-c5a5-47fe-b816-b29a92f5e85b)

**We need to convert the claim amount datatype from an object to a numerical format. So that we can continue with the further procedure.**

**Necessary Datatype Conversions**

     Claims['claim_amount']= Claims['claim_amount'].str.replace('$','').astype('float')

![image](https://github.com/user-attachments/assets/f6364ea8-bcb1-4f75-a3fc-f6544b6399ce)

**Converted the claim_amount datatype to numerical.**

**Here we need to check the minimum and maximum values of claim_amount and total_policy_claims. So that we can convert these variables to categorical variable by binning.**

     Claim_binned = pd.Series(pd.cut(Claims.claim_amount, range(0, 50000, 10000)))

     Claims_1['Claim_group'] = np.where(Claims_1.claim_amount<=10000, "Very Low",
                               np.where(Claims_1.claim_amount<=20000, "Low",
                               np.where(Claims_1.claim_amount<=30000, "Medium",
                               np.where(Claims_1.claim_amount<=40000, "High", "very High"))))

     Claims_1['total_policy_group'] = np.where(Claims_1.total_policy_claims<=3, "Low",
                                      np.where(Claims_1.total_policy_claims<=6, "Medium", "High"))


**Now we have converted the claim_amount, Claim_group and total_policy_claim datatype to categorical variables.**

**As we have created new columns so now we can drop the old columns.**


**We can also drop the irrelevant columns like claim_date, claim_id and customer_id from the DataFrame as these variables does not give that much impact to our model.**


    Claims_1 = Claims_1.drop(columns = ['claim_date'])
    Claims_1 = Claims_1.drop(columns = ['claim_id', 'customer_id'])
    Claims_1.head()

![image](https://github.com/user-attachments/assets/86289e25-4f0c-4c91-b448-9fcd76b34ebb)

**We can also convert the fraudulent variable to numerical datatype.**

      Claims_1['fraudulent'] = np.where(Claims_1['fraudulent'] == 'No',0,1)


![image](https://github.com/user-attachments/assets/09c1186d-c9aa-45a3-80d1-f658d8c5a4e3)


**Checking the missing values**

      Claims_1.isna().sum()

![image](https://github.com/user-attachments/assets/0419a81a-2970-4cf0-8b1e-406776d170ff)

**There are no missing values in the data.**


### Checking the number of frauds vs number of normal people by using bar charts.

![image](https://github.com/user-attachments/assets/32a2d2a9-9aba-4f45-bf2d-6bbbd3d86c2f)

So by seeing this bar chart we can say that the number of Normal Customers are more than the fraud customers.


### Use one-hot encoding to convert the variables into numerical columns

**As we have all the categorical variable in our DataFrame so we can do One-hot Encoding and convert all the categorical variables to numerical to build the model.**

         Claims_1 = pd.get_dummies(Claims_1, columns = ['incident_cause', 'claim_area', 'police_report', 'claim_type','Claim_group', 'total_policy_group'], dtype = int)
         Claims_1.head(10)

![image](https://github.com/user-attachments/assets/3b3ce122-0ec5-4eca-8a5d-35798fc366b2)



         Claims_1['claim_amount'] = Claims['claim_amount']
         Claims_1['total_policy_claims'] = Claims['total_policy_claims']



**Earlier we have dropped the claim_amount and total_policy_claims variables but again added because these two variable give a lot of impact to our model.**

         Claims_1.head(10)

![image](https://github.com/user-attachments/assets/6b92c4a2-8a43-47bf-910f-43a87466becb)


**Again Checking the missing values as the data has converted to numerical**


         Claims_1.isna().sum()

![image](https://github.com/user-attachments/assets/245e81cb-8e23-4563-8666-81b6ee0d2fbe)


**We are getting some missing values in the claim_amount and total_policy_claims columns, so we will just impute them with their respective medians.**


           Claims_1['claim_amount'] = Claims_1['claim_amount'].fillna(Claims_1['claim_amount'].median())
           Claims_1['total_policy_claims'] = Claims_1['total_policy_claims'].fillna(Claims_1['total_policy_claims'].median())
           Claims_1.head()

![image](https://github.com/user-attachments/assets/4a27e461-2d51-441c-81d9-0c944784a0be)


**We have filled the missing values by using median.**

## Outlier Treatment

           sns.boxplot(Claims_1.claim_amount)

![image](https://github.com/user-attachments/assets/8461c2ca-500a-4fc6-945a-043eb2550df3)


          sns.boxplot(Claims_1.total_policy_claims)

![image](https://github.com/user-attachments/assets/ffe3e3e1-d91d-45cf-887b-2b73a19617c4)


     def outliertreat_IQR(d):
         m = d.quantile(0.5)
         q1 = d.quantile(0.25)
         q3 = d.quantile(0.75)
         q_1p = d.quantile(0.01)
         q_99p = d.quantile(0.99)
         iqr = q3 - q1
         lc = q1 - 1.5*iqr
         uc = q3 + 1.5*iqr
         return lc,uc


## Feature Engineering: Correlation Analysis


    corr = Claims_3.corrwith(Claims_3['fraudulent']).abs().sort_values(ascending = False)
    corr

![image](https://github.com/user-attachments/assets/af92e5b0-d431-46d6-b0f0-7564c37ff0fe)


    v = corr[np.abs(corr)>0.03]


![image](https://github.com/user-attachments/assets/25512cc4-9fec-4958-b114-10f8ebe33b38)


**So here we have taken the cut off of less than 0.03 so that we can build the model.**


## Split train-test data

import sklearn

from sklearn.model_selection import train_test_split

**Importing the train_test_split module from sklearn library to validate the performance of the model.**


      x = Claims_3.drop(['fraudulent'], axis = 1)

      y = Claims_3['fraudulent']

**Dropping the fraudulent variable as it does not give that much impact to our model.**


## Building a Naive Bayes Model:

      from sklearn.naive_bayes import GaussianNB  # Gaussian distribution/ Normal distribution, so you don't have to do any transformation

      gnb = GaussianNB()

# fit the model

      gnb.fit(x_train, y_train)
 

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.

On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.


      y_train_pred = gnb.predict(x_train)
      y_train_pred = pd.Series(y_train_pred)
      Model_data_train = pd.DataFrame(y_train)
      Model_data_train['y_pred'] = y_train_pred
      Model_data_train

![image](https://github.com/user-attachments/assets/2eff4235-1810-47d8-bc2d-c9715ff51d8e)


**Predicting the y value for the training dataset.**


      pd.crosstab(Model_data_train.fraudulent,Model_data_train.y_pred, margins = True)

![image](https://github.com/user-attachments/assets/398b5f39-6152-4a4e-b02d-778bb7b7dee8)


     from sklearn.metrics import confusion_matrix
     data_table = confusion_matrix(y_train, y_train_pred)
     print('Confusion matrix\n\n', data_table)
     print('\nTrue Positives(TP) = ', data_table[0,0])
     print('\nTrue Negatives(TN) = ', data_table[1,1])
     print('\nFalse Positives(FP) = ', data_table[0,1])
     print('\nFalse Negatives(FN) = ', data_table[1,0])
     data_table.shape

![image](https://github.com/user-attachments/assets/a957e180-116d-4ad1-8603-d3cfa83028f8)


### Using the Heatmap

    matrix = pd.DataFrame(data=data_table, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')


![image](https://github.com/user-attachments/assets/9d33b407-1c47-4216-a491-774d6507a246)


     from sklearn.metrics import classification_report
     print(classification_report(y_train, y_train_pred))


![image](https://github.com/user-attachments/assets/9f0d8803-ae9d-4f23-ab73-d85f6a691685)


     from sklearn.metrics import accuracy_score
     print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_train_pred)))

![image](https://github.com/user-attachments/assets/48ffad35-7a8c-4ec9-a63b-24605253f851)

## Evaluate the model on the test data

     y_test_pred = gnb.predict(x_test)
     y_test_pred = pd.Series(y_test_pred)
     Model_data_test = pd.DataFrame(y_test)
     Model_data_test['y_pred'] = y_test_pred
     Model_data_test


![image](https://github.com/user-attachments/assets/3e0c4e07-dddc-4deb-b20a-373ab78f11e7)


     pd.crosstab(Model_data_test.fraudulent,Model_data_test.y_pred, margins = True)


![image](https://github.com/user-attachments/assets/969df912-903d-40b5-a7df-3e1823bb635a)


     from sklearn.metrics import classification_report
     print(classification_report(y_test, y_test_pred))

![image](https://github.com/user-attachments/assets/42a8bd31-7cec-47e6-810b-22ee1c95819f)

     from sklearn.metrics import accuracy_score
     print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_test_pred)))

![image](https://github.com/user-attachments/assets/f2e92ef1-6745-473b-8dc5-ed68399a6c1e)

**It is not an overfitting model. We have divided the training and testing dataset in 80-20 ratio and we got almost similar training and testing accuracy.**


**Conclusion :** In this project, we set out to improve the accuracy of our model, initially achieving a 73% accuracy rate. By doing some Outlier Treatment and Correlation Analysis the model's accuracy rate has increased but importantly by reintroducing the claim_amount variable into our analysis, we observed a notable increase in accuracy, reaching 77%. This suggests that in this dataset, claim_amount variable plays a significant role in predicting the outcomes of our model.












