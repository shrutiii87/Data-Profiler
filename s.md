<img width="1400" height="420" alt="banner" src="https://github.com/user-attachments/assets/8ae1b87b-10ed-4591-8825-ca4f62bf5632" />

## 🎯 Objective 
Conduct **Data Preprocessing and Feature Engineering** on a real-world dataset.  
The aim is to **understand, clean, transform, and analyze** the dataset before it can be used for machine learning.  
This project emphasizes **data profiling, handling multiple formats, and performing EDA**.

---

## 📄 Problem Statement
You have been hired as a **Junior Data Analyst** by a consumer insights company.  
The dataset contains **customer purchase behavior** collected from multiple sources:
- CSV
- JSON
- SQL Database
- API  

Your goal is to **frame a machine learning problem** (predict customer churn) and perform **data preprocessing and profiling** to make the dataset ML-ready.  
You will apply concepts of **data analysis, tensors, data cleaning, and exploratory data analysis (EDA)** to extract insights.

---

## 🗂️ Project Files

| File / Folder                         | Description                                                     |
| -------------------------------------- | --------------------------------------------------------------- |
| 📁 `files_used/`                       | Contains all source files and reference materials used in the project |
| 📁 `outputs/`                          | Stores generated reports, visualizations, and profiling results |
| 📓 `data_profiler.ipynb`               | Main Jupyter Notebook for data preprocessing and profiling workflow |
| 📊 `retail_customer_transactions.csv`  | Raw dataset containing customer transaction records             |
| 📄 `customer_profiles.json`            | JSON dataset with customer profile information                  |
| 🗄️ `retail_customers.db`               | SQLite database containing structured customer data             |
| 📄 `Part_A_Fundamentals.pdf`           | Theoretical notes and mathematical derivations used in analysis |
| 🌐 `retail_profiling_report.html`      | Generated HTML report from ydata_profiling summarizing dataset insights |
| 📘 `README.md`                         | Project documentation and workflow guide                        |


---


## 🛠️ Tools Used

<div>
  
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite3-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/Requests-FF6F00?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-2C2D72?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Seaborn-4C9A2A?style=for-the-badge&logo=seaborn&logoColor=white"/>
<img src="https://img.shields.io/badge/ydata_profiling-FF4088?style=for-the-badge&logo=python&logoColor=white"/>

</div>

---

# 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1__WPrCETYab-M1IWBLWkbHsIcmElHnY4/view?usp=sharing)

📹 Click the badge above to watch the complete project demonstration.

---

### ♻️ Workflow :-

<img width="900" height="260" alt="623677736-04f1f10a-ea88-46e0-a13c-0cb251f93c85" src="https://github.com/user-attachments/assets/b789d565-a6e4-42e7-bbec-a3f0e804fd91" />

---

### 📥 Jupyter notebook

<img width="800" height="420" alt="ezgif-6f16a27d2d9f39e8" src="https://github.com/user-attachments/assets/4b503ecb-6a66-453a-afac-106e02a7a7d1" />

---



#### 🛠️ imported libraries

```python
import pandas as pd
import numpy as np
import requests
import json
import sqlite3
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

```

---

## 📊 Part A : Fundamentals 

## 📶 Part B : Data Acquisition 

### 5️⃣ import datasets from different sources :
- 📂 load CSV file using pandas .
- 🗂️ parse a JSON file .
- 🔗 Connect to a SQL table and fetch records .
- 🌐 fetch data from an API .

```python
import pandas as pd
df_csv = pd.read_csv("retail_customer_transactions.csv")
print(df_csv.shape)
df_csv.head()
---------------------------------------------------------
import json
with open("customer_profiles.json") as file_obj:
    profile_records = json.load(file_obj)
df_json = pd.json_normalize(profile_records)
df_json.info()
----------------------------------------------------------
import sqlite3
conn = sqlite3.connect("retail_customers.db")
cur = conn.cursor()
----------------------------------------------------------
import requests
resp = requests.get("https://randomuser.me/api/?results=20").json()
df_api = pd.json_normalize(resp["results"])
df_api[["gender", "email", "location.city"]].head()
```

---

### 🗂️ Merged the data :- 

```python
combined_df = pd.concat([df_csv, df_json, df_sql], ignore_index=True)

combined_df = combined_df.merge(
    df_api[["gender", "email", "location.city"]],
    left_index=True, right_index=True, how="left"
)

combined_df.info()
```
---

## 🧹 Part C : Data Understanding & Cleaning 

<img width="1100" height="420" alt="623677752-1f9d2e0b-9c93-4eb7-ae51-f6c417731cfd" src="https://github.com/user-attachments/assets/e8a7f564-9d26-4e13-87f9-8b5c8004ceec" />

### 6️⃣ Perform initial exploration :
- 📝 Use .head() , .info() , .describe() to explore
- 🔍 Identify missing values and duplicates 

```python
print(combined_df.head())
print(combined_df.info())
print(combined_df.describe(include="all"))
print(combined_df.isnull().sum())
print(combined_df.duplicated().sum())
```

### 7️⃣ Apply data cleaning.
- 🔍 Handle missing data.
- 🧠 Correct inconsistent data types. 
- 🗑️ Drop irrelevant columns.


```python
combined_df = combined_df.dropna(subset=["Age", "Annual_Income"], how="any")
combined_df["Total_Purchases"] = combined_df["Total_Purchases"].fillna(0).astype(int)
combined_df["Age"] = combined_df["Age"].astype(int)
combined_df["Annual_Income"] = combined_df["Annual_Income"].astype(int)
combined_df = combined_df.drop_duplicates()
combined_df = combined_df.drop(columns=["Unused_Col"], errors="ignore")

print("After cleaning:", combined_df.shape)
```

---

## 🧮 Part D : Exploratory Data Analysis (EDA)

<img width="1200" height="420" alt="623677768-69a78643-2f5e-48f7-a62b-ac7341af758f" src="https://github.com/user-attachments/assets/5f94e522-4891-499b-882c-5aadfe908515" />

### 8️⃣ Perform Univariate Analysis:
- Distribution plots of age,income, and purchases.

<img width="540" height="393" alt="age distribution" src="https://github.com/user-attachments/assets/42cdd1fd-a81b-4376-b560-1e904eee3494" />

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution Plot of Age
plt.figure(figsize=(6,4))
sns.histplot(combined_df["Age"], kde=True,color="#671839"  )
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```

##### 📊 Insight: Age is fairly evenly distributed between 18 and 70 years, with no dominant age group. 
##### ✅ Conclusion: The dataset includes customers from a wide range of ages. 

<img width="545" height="393" alt="annual income distribution" src="https://github.com/user-attachments/assets/88412274-559c-47d2-8e33-4b8c5c5b0f8d" />


```python
# Distribution Plot of Annual Income
plt.figure(figsize=(6,4))
sns.histplot(combined_df["Annual_Income"], kde=True,color="#561A8A")
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.show()
```

##### 📈 Annual income is spread across the full income range without a strong peak.
##### ✅ Customers have diverse income levels. 💵

<img width="540" height="393" alt="total purchases distribution" src="https://github.com/user-attachments/assets/f01fd6d0-a318-4103-9f3c-70fee7e17e75" />


```python
# Distribution Plot of Total Purchases
plt.figure(figsize=(6,4))
sns.histplot(combined_df["Total_Purchases"], kde=True,color="#092A48")
plt.title("Total Purchases Distribution")
plt.xlabel("Total Purchases")
plt.ylabel("Frequency")
plt.show()
```

##### 📦 Insight: Total purchases are distributed across the range with no single purchase level dominating. 
##### ✅ Conclusion: Customers show varied purchasing behavior. 

---

### 9️⃣ Perform Bivariate Analysis: 
- Relationship between Gender & Purchases.
- Relationship between Income & Churn.

<img width="563" height="455" alt="gender vs purchases" src="https://github.com/user-attachments/assets/229acbf2-21f1-4c03-97fb-2576e6f96e61" />


```python
sns.boxplot(x="Gender", y="Total_Purchases", data=combined_df , color="#F05D6C")
plt.title("Gender vs Purchases")
plt.show()
```

##### 📦 Insight: Male and female customers have a similar median of around 30 purchases, and both groups have a similar spread (about 1–60 purchases).
##### ✅ Conclusion: There is no clear difference in total purchases between male and female customers.

---

<img width="566" height="455" alt="income vs purchases by status" src="https://github.com/user-attachments/assets/2a6de1cd-74c1-4548-906e-651efc3a2d3e" />


```python
sns.scatterplot(x="Annual_Income", y="Total_Purchases", hue="Subscription_Status", data=combined_df)
plt.title("Income vs Purchases by Status")
plt.show()
```

##### 📈 Insight: Annual income ranges from about 20,000 to 120,000, while purchases range from 1 to 60, with points spread across the entire plot.
##### ✅ Conclusion: There is no clear relationship between annual income and total purchases from this scatter plot alone.

---

### 🔟 Perform Multivariate Analysis:
- Correlation heatmap of all numerical variables.
- Pair plots to identify feature interactions.

<img width="752" height="655" alt="correlation matrix" src="https://github.com/user-attachments/assets/dee9f867-de62-49a8-876a-24d7c2a66be6" />


```python
plt.figure(figsize=(8,6))
sns.heatmap(combined_df.corr(numeric_only=True), annot=True, cmap="magma")
plt.title("Correlation Matrix")
plt.show()
```

##### 📦 Insight: The strongest correlation visible is only ≈0.05 (Customer_ID vs Annual Income). Age vs Total Purchases is basically ≈−0.01, which is indistinguishable from zero.
##### ✅ Conclusion: Since every value sits between −0.04 and 0.05, the heatmap proves there is no linear relationship between customer attributes and buying behavior.


---

<img width="741" height="741" alt="pairplot" src="https://github.com/user-attachments/assets/5909066e-4fc6-4eb1-b1cd-8cda121b8e98" />


```python
sns.pairplot(combined_df[["Age", "Annual_Income", "Total_Purchases"]])
plt.show()
```

##### 📦 Insight: The scatter plots are just square clouds of dots. For example, Age (20–70) vs Annual Income ($20k–$120k) shows a perfectly even spread, and Purchases scatter evenly across the axis.
##### ✅ Conclusion: The uniform scatter proves independence — Age, Income, and Purchases don’t cluster or trend together; they’re spread randomly across their ranges.

---


## 🧮 Part E : Data Profiling

### 1️⃣1️⃣ Generate a Pandas Profiling Report that summarizes:
- Missing Values.
- Descriptive stat.
- Correlations.
- Warnings on potential data quality issues.


```python
from ydata_profiling import ProfileReport
report = ProfileReport(combined_df, title="Retail Customer Profiling Report", explorative=True)
report.to_file("retail_profiling_report.html")
print("Report saved.")
```

---

## 📂 Project Workflow
1. **Data Acquisition** → Import data from CSV, JSON, SQL, and API  
2. **Data Cleaning** → Handle missing values, duplicates, and inconsistencies  
3. **Data Profiling** → Generate profiling reports for better understanding  
4. **Feature Engineering** → Create new features for ML readiness  
5. **EDA** → Visualize distributions, correlations, and trends  
6. **Problem Framing** → Define ML task (customer churn prediction)

---

---

## 📈 Results & Insights
- ✅ Cleaned dataset with consistent data types and no duplicates  
- ✅ Exploratory analysis showing **no strong correlations** between age, income, and purchases  
- ✅ Profiling report highlighting **missing values and data quality issues**  
- ✅ Clear framing of ML problem: **Customer Churn Prediction**

---

## 📌 Expected Outcomes
- A **structured dataset** ready for machine learning models  
- **EDA visualizations** (histograms, scatter plots, heatmaps, pair plots)  
- **Profiling report** generated via `ydata_profiling`  
- Documentation of **data cleaning and preprocessing steps**  

---

## ⚙️ Installation & Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/data-preprocessing-project.git
cd data-preprocessing-project
pip install -r requirements.txt
```

---

## 🙏 Thank You
Thank you for taking the time to explore this project!  
Your feedback, suggestions, and contributions are always welcome.  

⭐ If you found this project helpful, don’t forget to **star the repository** and share it with others.  
