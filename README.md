PyMicroML: A Machine Learning-based Toolkit for Identifying Key Microbes Linked to Fermentation Metabolites

**PyMicroML** is a machine learning-based toolkit designed to identify key microbes (e.g., ASVs) linked to fermentation metabolites. It supports multiple feature selection algorithms and model integration analysis, making it suitable for microbial and metabolite interaction research in fermented food, environmental samples, and other related fields.

------

## Data Preparation

The following data is required for analysis:

1. **Fermentation Physicochemical Parameter Data (Optional)**
   - Matrix data including parameters such as temperature, pH, etc.
2. **Fermentation Metabolite Data**
   - Quantitative information on target metabolite concentrations.
3. **Microbiome Data (ASV or OTU Table)**
   - Common formats include `.csv` / `.tsv` where rows represent samples and columns represent microbial features.

------

## Environment Requirements

- Python ≥ 3.7

### Python Dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `shap`

------

##  🚀 Analysis Workflow

2. **Correlation Network Analysis**

   - Calculate the correlation coefficients (Spearman/Pearson) between microbes and metabolites.
   - Build networks and select key microbes (ASVs).

2. **Ranking of ASV feature importance**

   Three models are applied to evaluate the importance of microbial features:

   - **PLSR VIP** (Partial Least Squares Regression Variable Importance in Projection)
   - **Random Forest Feature Importance**
   - **XGBoost SHAP** (Shapley Additive Explanations)

3. **Feature Selection: Machine Learning Algorithms**

- Supported algorithms include:
  - Bagging
  - Random Forest
  - XGBoost
  - AdaBoost
  - Extra Trees
- Select the optimal number of ASV features and fermentation parameters combinations

4. **Constructing a predictive model of flavor metabolites**

   Train regression models using selected microbial features and physicochemical variables

   Evaluate model performance using cross-validation

   Visualize predicted vs. actual metabolite concentrations

------

## Quick Start

```
git clone https://github.com/yourname/PyMicroML.git
cd PyMicroML

# Create the environment
conda env create -f environment.yml
conda activate pymicroml

# Run the jupyter notebook
jupyter notebook **.ipynb
```