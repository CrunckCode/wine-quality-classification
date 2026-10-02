# Red Wine Quality Classification

Supervised and unsupervised study of the UCI Red Wine Quality dataset (1,599 samples, 11 physicochemical features) for the Rutgers MQF Data Mining course. Wine is labelled Good (quality 7 or higher, 13.6% of samples) or Not Good, which makes it a mild class-imbalance problem. The imbalance, threshold and AUC work is the same toolkit used in credit default (PD) modeling.

**Note:** a course group submission. The notebook is the code I wrote. The presentation deck and script are not included.

## Files
- `Final_wine_quality_project.ipynb` (79 cells)
- `winequality-red.csv`: the public dataset.

## Part 1: classification pipeline
- **EDA:** summary statistics, no missing values, correlation heatmap (alcohol +0.48, volatile acidity -0.39, sulphates +0.25, citric acid +0.23), per-quality boxplots.
- **Preprocessing:** target binarization, removal of 240 duplicate rows (1,359 remain), 80/20 stratified split, Z-score scaling fit on the training set only to avoid leakage.
- **Models:** Decision Tree, Random Forest, KNN and Gaussian Naive Bayes compared by 5-fold CV; Random Forest (0.880) and KNN (0.858) lead.
- **Tuning:** `GridSearchCV` with F1 scoring (preferred over accuracy under imbalance). KNN: k=3, Manhattan distance. Random Forest: max_depth 10, max_features sqrt, min_samples_leaf 5, 100 trees.
- **Evaluation:** training versus test error, confusion matrices, feature importances, KNN bias-variance curve. Baseline test results: Random Forest accuracy 0.893 with Good-class recall 0.35; KNN accuracy 0.868 with recall 0.41.

## Part 2: improvements
1. **Class imbalance:** SMOTE (train only), class weighting and threshold tuning (threshold chosen on training data). Random Forest Good-class results: baseline recall 0.35 and F1 0.47; **class weighting recall 0.62 and F1 0.57 (best)**; SMOTE 0.57 and 0.48; threshold 0.40 gives 0.46 and 0.54.
2. **Naive Bayes assumption check:** six feature pairs with |r| above 0.5 (fixed acidity and pH -0.69, fixed acidity and density +0.67, free and total SO2 +0.66), which breaks conditional independence and is consistent with NB ranking last.
3. **ROC-AUC:** Random Forest about 0.88 versus KNN about 0.75, showing F1 understated the Random Forest's ranking advantage.

## Part 3: clustering
K-Means (elbow and silhouette, peak about 0.21 at K=2), DBSCAN (degenerate across the eps sweep) and Ward agglomerative clustering with a dendrogram, validated against the true labels with ARI, NMI and silhouette. Every ARI is below 0.10, so the chemistry does not cluster into Good and Not Good; supervised models work by learning the labelled boundary. A 2D PCA projection (PC1 28%, PC2 17%) is used for visualization.

## Known shortcut
`GridSearchCV` is tuned once rather than re-tuned for each imbalance variant.
