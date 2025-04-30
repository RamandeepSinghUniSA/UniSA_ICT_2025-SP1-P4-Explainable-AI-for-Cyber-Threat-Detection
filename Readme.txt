Main:
Datasets:
Full-Dataset
One-hot Encoded Data - correlation encoder (see CorrelationEncoderTests in tools).
T-001 - Test impact of different parameters on shap calculations.
T-002A - Test sampled Explainer approximation to full data.
T-002B - Test sampled Explainer for sampled Neural Network.
T-003 - Test sampling methods for Neural Network (after modification).
T-004 - SHAPselector class (incomplete).
SMOTErf - Test sampling techniques on Random Forest.
PrimaryBinary - Saved SMOTE model for ensemble.
SecondaryMulti - Secondary RandomForest model for ensemble (will be changed).
Ensembler - Combining the two models for final prediction.

Models:
models - Contains the dynamic Neural Network model.
saved_models: saved models

Reports: Old reports from capstone 1.

Tools: 
encoder - for encoding data using correlation
data_visualiser - A tool for visualising within label data.

Archive: Older tests kept for historic purpose.