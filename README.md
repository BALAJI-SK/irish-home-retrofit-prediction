# Irish Home Retrofit Prediction

Machine learning project to predict Irish home energy ratings (BER) based on physical and efficiency drivers from the SEAI dataset (1.35 million rows).

## Documentation

This repository contains comprehensive documentation covering the entire data engineering and exploratory analysis pipeline:

1. **[Data Cleaning & Feature Selection](data_cleaning.md)**: 
   Outlines the Phase 1 and Phase 2 approaches used to reduce the raw dataset from ~215 columns to a curated set of 45 physical and efficiency drivers (mitigating target leakage and sparsity).

2. **[Row Imputation & Statistical Analysis](imputation_analysis_report.md)**: 
   Documents the methodology for handling missing data, categorizing NMAR vs. MAR blocks, applying Contextual Regression Imputation vs Mode, and using the Interquartile Range (IQR) for outlier suppression to preserve distribution integrity.

3. **[PCA, LDA, MCA & FAMD Dimensionality Reduction](pca_lda_report.md)**: 
   The capstone exploratory data analysis report. It uses four distinct dimensionality reduction methods across continuous and categorical data types to extract the final feature sets. Includes a rigorous VIF/Pearson multicollinearity audit resulting in 35 final features suitable for multiple modelling pathways.

4. **[Initial Row Cleaning Discovery](row_cleaning_balaji_14_mar.md)**: 
   Raw notes, statistical outcomes, and lecture-grounded justifications (DCU MSc) regarding the decision to drop `PredominantRoofType` and other heavily missing blocks.
