# Customer Segmentation Using KMeans Clustering

## Overview

This project applies **unsupervised learning** to segment mall customers into distinct groups based on their demographic and spending characteristics. Unlike supervised learning tasks, there is no target variable here — the goal is to discover natural groupings within the customer base using clustering.

The project uses the **Mall Customer Segmentation** dataset from Kaggle:
[kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

The dataset contains the following columns:

- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## Objective

The goal is to segment mall customers into meaningful groups using KMeans clustering, and to describe those groups in a way that could support targeted marketing decisions. The workflow includes:

- Loading and exploring the dataset
- Exploratory Data Analysis (EDA)
- Selecting and scaling appropriate features
- Applying KMeans with different values of K
- Using the Elbow Method to select the final number of clusters
- Profiling and interpreting the resulting segments
- Visualizing clusters in 2D using PCA
- Saving the final model and scaler
- Testing the saved artifacts on sample new customers

## Feature Selection

The KMeans model was trained using only three numerical features:

- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

**`CustomerID`** was excluded because it is only an identifier and carries no meaningful information for clustering.

**`Gender`** was explored during EDA to understand the overall customer base, but it was **not** used as a training feature. The project focuses on numerical demographic and spending behavior, and since KMeans is a distance-based algorithm, only continuous numerical features were used for training.

Before clustering, the three selected features were standardized using `StandardScaler()`. This ensures that features with larger numeric ranges (such as income) do not dominate the distance calculations used by KMeans. The scaled features were then passed to the clustering algorithm.

## Choosing the Number of Clusters

KMeans was first tested with a few candidate values:

| Number of Clusters | Inertia |
|---|---|
| 3 | 295.21 |
| 4 | 205.23 |
| 5 | 168.25 |

To make a more informed choice, a broader Elbow Method analysis was performed across K = 2 through K = 10:

| K | Inertia |
|---|---|
| 2 | 389.39 |
| 3 | 295.21 |
| 4 | 205.23 |
| 5 | 168.25 |
| 6 | 133.87 |
| 7 | 117.01 |
| 8 | 103.87 |
| 9 | 93.09 |
| 10 | 82.39 |

The final model uses **K = 5**. This value was chosen because the rate of decrease in inertia slows down noticeably beyond K = 5, indicating diminishing returns from adding further clusters. K = 5 offers a reasonable balance between cluster compactness, simplicity, and interpretability. This is not presented as the only mathematically valid choice — it is a reasonable, defensible one based on the Elbow Method and the interpretability of the resulting segments.

## Cluster Profiles

The final KMeans model (K = 5) produced the following segments:

| Cluster | Avg. Age | Avg. Income (k$) | Avg. Spending Score | Customer Count |
|---|---|---|---|---|
| 0 | 46.25 | 26.75 | 18.35 | 20 |
| 1 | 25.19 | 41.09 | 62.24 | 54 |
| 2 | 32.88 | 86.10 | 81.53 | 40 |
| 3 | 39.87 | 86.10 | 19.36 | 39 |
| 4 | 55.64 | 54.38 | 48.85 | 47 |

### Interpretation

**Cluster 0 — Low-Income, Low-Spending Customers**
Average age 46.25, average income 26.75k, average spending score 18.35 (20 customers). This segment represents relatively low-income, low-spending customers.

**Cluster 1 — Young, High-Spending Customers**
Average age 25.19, average income 41.09k, average spending score 62.24 (54 customers). A younger segment with moderate income but relatively high spending behavior.

**Cluster 2 — High-Value Customers**
Average age 32.88, average income 86.10k, average spending score 81.53 (40 customers). The strongest segment overall, combining high income with high spending.

**Cluster 3 — High-Income, Low-Spending Customers**
Average age 39.87, average income 86.10k, average spending score 19.36 (39 customers). High income but low spending, making this an interesting segment for targeted marketing and promotional strategies.

**Cluster 4 — Mature, Moderate-Spending Customers**
Average age 55.64, average income 54.38k, average spending score 48.85 (47 customers). The oldest segment on average, with relatively balanced income and spending behavior.

### Key Observation

Clusters 2 and 3 share nearly the same average annual income (~86.10k) but have very different spending scores (81.53 vs. 19.36). This demonstrates that income alone does not determine spending behavior, reinforcing the value of considering multiple customer characteristics — rather than income alone — for segmentation.

## PCA Visualization

PCA was used **after** clustering, purely for visualization purposes:

- KMeans was trained on the three standardized original features (`Age`, `Annual Income (k$)`, `Spending Score (1-100)`).
- PCA was **not** used to train KMeans.
- PCA was applied only to reduce the feature space to two dimensions so the five resulting clusters could be visualized on a 2D scatter plot.

## Exploratory Data Analysis

The notebook includes EDA covering:

- Age distribution
- Annual Income distribution
- Spending Score distribution
- Gender distribution
- Annual Income vs. Spending Score
- Age vs. Spending Score

The purpose of this EDA was to understand the structure and distribution of the customer data before applying clustering.

## Saved Artifacts

The project saves two artifacts:

- `models/kmeans_model.pkl` — the trained KMeans model
- `models/scaler.pkl` — the fitted `StandardScaler`

Both are needed for the model to be usable on new data. The scaler must be applied to any new customer data first, so that it is transformed using the exact same scaling process used during training. The scaled data is then passed to the saved KMeans model, which assigns the new customer to one of the five learned clusters.

The notebook also reloads both saved artifacts and tests them on sample new customers to verify that they work correctly outside of the original training session.

## Project Structure

```
customer-segmentation-clustering/
│
├── data/
│   └── Mall_Customers.csv
│
├── models/
│   ├── kmeans_model.pkl
│   └── scaler.pkl
│
├── results/
│   ├── feature_distributions.png
│   ├── income_vs_spending_eda.png
│   ├── income_vs_spending_clusters.png
│   ├── elbow_method.png
│   ├── pca_clusters.png
│   └── cluster_profiles.csv
│
├── customer_segmentation.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

- **`data/`** — contains the raw dataset used for the project.
- **`models/`** — contains the saved KMeans model and scaler.
- **`results/`** — contains generated plots and the exported cluster profile summary.
- **`customer_segmentation.ipynb`** — the main notebook containing the full workflow, from data loading to model evaluation.
- **`requirements.txt`** — lists the Python dependencies required to run the project.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

Clone the repository and navigate into the project folder:

```bash
git clone <repository-url>
cd customer-segmentation-clustering
```

Create and activate a virtual environment (Windows):

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open `customer_segmentation.ipynb` and run the notebook from beginning to end.

## Results

- K = 5 was selected as the final number of clusters using the Elbow Method.
- Five distinct customer segments were identified and profiled.
- Each cluster was interpreted in plain terms based on age, income, and spending behavior.
- PCA was used to visualize the five clusters in two dimensions.
- The final KMeans model and the `StandardScaler` were saved for reuse.
- The saved artifacts were tested on sample new customers to confirm they work correctly.

A key business insight from this analysis is that **high income does not necessarily translate to high spending** — clearly illustrated by Clusters 2 and 3, which have nearly identical average income but very different spending behavior.

## Conclusion

KMeans clustering was used to identify five distinct customer segments based on age, income, and spending behavior. These segments provide a clearer picture of the mall's customer base than raw, unsegmented data would allow, and highlight groups — such as high-income, low-spending customers — that could be prioritized for targeted marketing and promotional strategies. This kind of segmentation can help guide more informed, data-driven marketing decisions.