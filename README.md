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





























































































































































































