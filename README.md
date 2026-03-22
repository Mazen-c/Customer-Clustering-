# Customer Segmentation on Wholesale Data

This project applies unsupervised learning to the Wholesale Customers dataset to discover natural customer groups based on annual spending behavior.

The workflow includes:
- Data loading and quality checks
- Feature preprocessing (log transform + standardization)
- Dimensionality reduction with PCA
- Cluster analysis with multiple algorithms
- Model selection with Elbow, Silhouette, and GMM criteria

## Dataset

- Source file: `Data/Wholesale customers data.csv`
- Records: 440
- Columns: 8
- Target-style metadata columns: `Channel`, `Region`
- Spending features used for clustering:
	- `Fresh`
	- `Milk`
	- `Grocery`
	- `Frozen`
	- `Detergents_Paper`
	- `Delicassen`

## Project Structure

```text
Project 1/
|- Data/
|  |- Wholesale customers data.csv
|- Notebook/
|  |- Clustering.ipynb
|- Preprocessing.py
|- README.md
```

## Methods Used

1. Preprocessing
- Log transform with `np.log1p` to reduce skewness.
- Feature scaling with `StandardScaler`.

2. Dimensionality Reduction
- `PCA(n_components=2)` for visualization and clustering in 2D.
- First two PCs retain a large portion of variance (about 71.27% from script output).

3. Clustering Models
- K-Means
- Spectral Clustering
- Gaussian Mixture Model (GMM)

4. Cluster Selection
- Elbow method (inertia trend)
- Silhouette score (cluster separation quality)
- AIC/BIC for GMM component selection

## Environment Setup

Create and activate your virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

## How To Run

### Option 1: Python Script

```powershell
.\.venv\Scripts\python.exe Preprocessing.py
```

This runs:
- Data checks
- Preprocessing
- PCA
- Visual diagnostics

### Option 2: Jupyter Notebook

```powershell
jupyter notebook
```

Then open `Notebook/Clustering.ipynb` and run cells top to bottom.

## Key Outputs

- Dataset health summary (nulls, duplicates, schema)
- Distribution and correlation visualizations
- PCA projection plot (`PC1` vs `PC2`)
- Elbow curve for K-Means
- Silhouette-based cluster quality check
- Spectral and GMM clustering visual comparisons
- AIC/BIC curves for GMM model complexity

## Notes

- Inertia always decreases as `k` increases, so it should not be used alone to pick the final cluster count.
- Prefer combining Elbow shape with Silhouette score and interpretability.
- If Elbow and Silhouette disagree, Silhouette is often a stronger indicator of separation quality.

## Future Improvements

- Save final model outputs and labels to files.
- Add a `requirements.txt` for reproducible installs.
- Add quantitative comparison table across clustering methods.
- Include cluster profiling report for business interpretation.
