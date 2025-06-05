# UniSA_ICT_2025-SP1-P4-Explainable-AI-for-Cyber-Threat-Detection
UniSA_ICT_2025-SP1-P4-Explainable-AI-for-Cyber-Threat-Detection

# Has Sampling and Encoding techniques and Model implementations into SHAP.

Authors and Collaborators: Adi Selak, Mathew Coleman, Scott Chandler, Ramandeep Singh, Edrick Laitly

Datasets:
USNW-NB15 Dataset.

Tests:
T-001 - Test impact of different parameters on shap calculations.
T-002A - Test sampled Explainer approximation to full data.
T-002B - Test sampled Explainer for sampled Neural Network.
T-003 - Test sampling methods for Neural Network (after modification).
T-004 - Original Correlation Encoder used for dataset in Neural Network Model.
T-005 - Kmeans and DBScan tests on Normal labels in dataset.
T-006 - Optimisation and Kfold validation on Neural Network model.

Models:
models - Contains the dynamic Neural Network model.
saved_models: saved models used for testing, user guides, and evaluations.

Reports: Evaluations using SHAP on SMOTE Model and Multiclass classifier Tree Model.

Tools: 
encoder - for encoding data using correlation
data_visualiser - A tool for visualising within label data.
shapmanager - Used to steamline shap operations with additional grouping functions.

UserGuides: Example code of SMOTE and multi-class Tree with the shap manager class.
