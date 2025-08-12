🚢 Titanic Survival Prediction with Random Forest

This project predicts passenger survival on the Titanic using machine learning. 
It involves: data cleaning, feature engineering, exploratory data analysis (EDA), and model building with a Random Forest Classifier.

## 📌 Overview

The Titanic dataset is one of the most famous beginner-friendly datasets in data science.  
In this project, we:

- Clean missing values and handle categorical variables.
- Explore survival patterns with visualizations.
- Create new features (`FamilySize`, `IsAlone`).
- Train and evaluate a Random Forest model.

---

## 📂 Dataset

The dataset is loaded directly from GitHub:  
```

[https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

```

### Key Columns:
- **Survived**: 0 = No, 1 = Yes  
- **Pclass**: Passenger class (1st, 2nd, 3rd)  
- **Age, SibSp, Parch, Fare**: Numerical features  
- **Sex, Embarked**: Categorical features  

---

## 🔍 Steps in the Project

### 1️⃣ Data Cleaning
- Fill missing `Age` with the median.
- Fill missing `Embarked` with the mode.
- Drop `Cabin` due to high missingness.
- Remove unnecessary columns: `Name`, `Ticket`, `PassengerId`.

### 2️⃣ Feature Engineering
- **One-hot encoding** for categorical variables (`Sex`, `Embarked`).
- **FamilySize** = `SibSp` + `Parch` + 1.
- **IsAlone** = 1 if `FamilySize` == 1, else 0.

### 3️⃣ Exploratory Data Analysis (EDA)
- Survival distribution plots (`countplot`).
- Age vs Survival (`boxplot`).
- Passenger class vs Survival.
- Correlation heatmap.

### 4️⃣ Model Building
- **Split**: 80% training / 20% testing.
- **Scaling**: StandardScaler for numeric features.
- **Algorithm**: RandomForestClassifier.
- **Evaluation**: Accuracy score & classification report.

---

## 📊 Visualizations
Some of the visual insights include:
- Survival rate differences between genders.
- Younger passengers had a slightly higher survival rate.
- First-class passengers had higher survival chances.
- Heatmap showing correlations between features.

---

## ⚙️ Model Performance
Example output:
```

Accuracy: 0.82
precision    recall  f1-score   support
0       0.85      0.89      0.87       105
1       0.77      0.71      0.74        74

````

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Random Forest Classifier**
- **Jupyter Notebook / Python Script**

---

## 📈 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies
pip install -r requirements.txt

# Run the script / notebook
python titanic_model.py
````

---

## 📜 License

This project is licensed under the MIT License.
