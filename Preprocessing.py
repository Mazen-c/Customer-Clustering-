import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Wholesale customers data.csv")

print("Columns:", df.shape[1])
print("Rows:", df.shape[0])
df.info()
print(df.isnull().sum())
print(df.duplicated().sum())

# Select numerical columns
num_cols = df.columns.drop(['Channel', 'Region'])
df_processed = df.copy()
df_processed[num_cols] = np.log1p(df_processed[num_cols])   # Step 1: log
scaler = StandardScaler()
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])  # Step 2: scale

# Outlier detection
plt.figure(figsize=(10,6))
sns.boxplot(data=df_processed[num_cols])
plt.xticks(rotation=45)
plt.show()

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_processed[num_cols])

df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
print(pca.explained_variance_ratio_)
print("Total variance:", sum(pca.explained_variance_ratio_))

# PCA Results Visualization
plt.figure(figsize=(8,6))
plt.scatter(df_pca['PC1'], df_pca['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.show()

# Explanation
# PCA was applied to reduce the dimensionality of the dataset.
# Since several numerical features are correlated, PCA transforms them into
# a smaller set of uncorrelated components while preserving most of the variance.
# The number of components was selected based on the explained variance ratio
# to retain the majority of the information.