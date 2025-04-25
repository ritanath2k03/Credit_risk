# 📊 Credit Risk Clustering with KMeans

This project clusters credit applicants based on financial and demographic features using unsupervised machine learning. It helps identify groups such as low-risk, medium-risk, and high-risk applicants without labeled target data.

# video demonstration : 
[![Video Demo](https://example.com/image.jpg)](https://drive.google.com/file/d/1mbnHpjZX49RAbGmWp4QcZjax83NXjjbp/view)
---

## 🚀 Project Summary

- **Goal**: Segment customers using clustering to better understand risk profiles.
- **Algorithm**: KMeans clustering with PCA for visualization.
- **Dataset**: German Credit Data

---

## 🧠 Features Used

| Feature            | Description                             |
|--------------------|-----------------------------------------|
| Age                | Applicant's age                         |
| Sex                | Gender                                  |
| Job                | Job status (0 = unemployed to 3 = skilled) |
| Housing            | Type of housing                         |
| Saving accounts    | Savings level                           |
| Checking account   | Checking account balance                |
| Credit amount      | Total credit amount requested           |
| Duration           | Duration of the credit in months        |
| Purpose            | Purpose of the credit                   |

---

## 🛠️ Model Pipeline

1. **Data Preprocessing**
   - Handle missing values
   - Label encode categorical variables
   - Standardize numeric features

2. **Clustering**
   - Apply KMeans with 3 clusters
   - Use PCA to reduce dimensionality for visualization

3. **Evaluation**
   - Evaluate clusters using Silhouette Score

---

## 📈 Evaluation Metric

Since this is **unsupervised**, we use:

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to others.  
  **Score Achieved**: `~0.40`

  > This score shows moderately well-separated and meaningful clustering.

---

## ✅ Why KMeans is Optimal

- Works well with numeric + encoded data
- Efficient with large datasets
- Easy to interpret and deploy
- Consistent results across runs
- Silhouette Score proved it was more stable than DBSCAN or GMM for this problem

---

## 📦 Files Included

| File                       | Purpose                                |
|----------------------------|----------------------------------------|
| `german_credit_data.csv`   | Raw input dataset                      |
| `training_model.py`        | Full training script (preprocess + model saving) |
| `kmeans_model.pkl`         | Trained clustering model               |
| `scaler.pkl`               | StandardScaler object                  |
| `encoders.pkl`             | All LabelEncoders used in preprocessing |
| `clustered_credit_data.csv`| Output with cluster labels             |
| `streamlit_app.py`         | Streamlit UI for prediction            |
| `gradio_app.py`            | Gradio interface version               |
| `README.md`                | Project documentation (this file)      |

---

## 💡 Use Cases

- Customer segmentation in banking and finance
- Risk categorization for credit approvals
- Personalized marketing or fraud detection

---

## 📬 How to Run

### 🔷 Streamlit App

```bash
pip install gradio
python gradio_app.py

