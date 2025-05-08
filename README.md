# UniSA_ICT_2025-SP1-P4-Explainable-AI-for-Cyber-Threat-Detection
UniSA_ICT_2025-SP1-P4-Explainable-AI-for-Cyber-Threat-Detection

# The main folder contains recent work and project deliverables. The archive contains previous tests and implementations.

# Has Sampling and Encoding techniques and Model implementations into SHAP.

Author: Adi Selak

Main:
SMOTE NN Model.

Datasets:
Full-Dataset
One-hot Encoded Data - correlation encoder (see CorrelationEncoderTests in tools).

Tests:
T-001 - Test impact of different parameters on shap calculations.
T-002A - Test sampled Explainer approximation to full data.
T-002B - Test sampled Explainer for sampled Neural Network.
T-003 - Test sampling methods for Neural Network (after modification).
T-004 - SHAPselector class (incomplete).
T-005 - SMOTE upsampling on hard to predict labels.

Models:
models - Contains the dynamic Neural Network model.
saved_models: saved models

Reports: Old reports from capstone 1.

Tools: 
encoder - for encoding data using correlation
data_visualiser - A tool for visualising within label data.

Archive: Older tests kept for historic purpose.