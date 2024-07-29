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
