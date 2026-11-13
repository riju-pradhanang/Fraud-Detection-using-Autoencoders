# Report outline (≈6-8 pages)

1. **Abstract**: problem, approach (AE / RDA vs IF / OCSVM), headline PR-AUC numbers, one-sentence takeaway.
2. **Introduction**: card fraud, extreme imbalance, why unsupervised, contributions (what *you* did).
3. **Related work / background**: autoencoders for anomaly detection; Zhou & Paffenroth (KDD 2017); IF, OCSVM.
4. **Method**: AE architecture; RDA objective `‖L_D − D(E(L_D))‖ + λ‖S‖`, alternating optimisation; scoring; deviations from the paper
   (mini-batch, quantile λ, no ADMM dual); baselines and their hyper-parameters.
5. **Experimental protocol**: datasets (rows, fraud rate), preprocessing, splits (stratified / time), no-leakage rules, threshold chosen on validation,
   5 seeds, metrics (why PR-AUC primary; random baseline = fraud rate).
6. **Results**: main table (ULB), second table (IEEE-CIS), PR curves, score histograms, confusion matrices at the validation threshold.
7. **Ablations**: bottleneck size; λ / contamination; AE on normals vs contaminated vs RDA.
8. **Discussion**: when/why AE-based methods win or lose; failure cases (false positives on large amounts?); effect of the contamination in training.
9. **Limitations**: PCA-anonymised features, random split, few frauds → variance, no cost-sensitive analysis, IEEE-CIS feature handling.
10. **Conclusion + future work** (VAE, sequence models, cost-based thresholding).
11. **References**: Zhou & Paffenroth 2017; Liu et al. 2008 (Isolation Forest); Schölkopf et al. 2001 (OCSVM); Dal Pozzolo et al. 2015 (ULB dataset).
