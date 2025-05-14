import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import LabelEncoder 
class SHAPbinary:
    """
    Will add more docstrings later...

    Initialisation:
        - shap_values (numpy) The shap value arrays in two dimensions (values, features).
        - explainer (explainer) The explainer object created by re-running the explainer onto the test_data (xp(data))
        - data (pd.DataFrame, tensor) Either the background data from the Tree Model (pd.DataFrame) or Neural Network model (tensor). If using a tensor the feature_names parameter must be given with X_test.columns.
        - feature_names (list) A list from the feature name columns if using a tensor.
    
    NOTE: Tensors are automatically converted to a DataFrame object with the feature name mappings.

    """
    def __init__(self, shap_values, explainer, data, feature_names=None):
        self.explainer = explainer
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            self.data_main = pd.DataFrame(data, columns=feature_names)
            self.shap_values = shap_values.squeeze()
            self.data = self.data_main.copy()
        else:
            self.shap_values = shap_values
            self.data_main = data
            self.data = self.data_main.copy()
    
    def string_encode(self, encoder, categories):
        """
        string_encode: Use the feature encoder (le_features) to add strings back to the encoded categories.

        Parameters:
            - encoder (list) A list containing encoders for each encoded feature.
            - categories (list) A list of the categorical features encoded.
        """


        i = 0
        for c in categories:
            le = encoder[i]
            self.data[c] = le.inverse_transform(self.data[c])
            i += 1

    def custom_group(self, type_of, category, limit=None, calculation=None):
        """
        custom_group: Convert a group of categories based on top 30 for label encoded categories.

        Parameters:
            - type_of (string) For 'label-encoded-specific' provide a list of values to encode and change the rest to 'Other'. For 'label-encoded' provide a limit and calculation to calculate top 30.
            - category (string) The name of the category column.
            - limit (int) The total number of values to convert (rest are converted to other).
            - calculation (string) Calculation to select the top given by limit. For 'average' the absolute average is used. For 'sum' the absolute sum is used.
        """
        # Values below 30 are not a problem but we cant always guarantee this
        def convert(x):
            if x in category_values:
                return x
            else:
                return 'Other'

        if type_of == 'label_encoded-specific':
            data = self.data.copy()
            cat, values = category
            category_values = set(values)
            data[cat] = data[cat].apply(convert)
            self.data = data

        if type_of == 'label_encoded':
            data = self.data.copy()
            cat_idx = data.columns.get_loc(category)
            shap_values = np.abs(self.shap_values[:, cat_idx])
            calculated = {}
            for value in data[category].unique():
                subset = shap_values[data[category] == value]
                if calculation == 'average':
                    calculated[value] = subset.mean()
                elif calculation == 'sum':
                    calculated[value] = subset.sum()
            # We can update this to have top or bottom limit.
            sorted_values = sorted(calculated.items(), key=lambda x: x[1], reverse=True)
            top_values = sorted_values[:limit]
            category_values = []

            for value in top_values:
                category_values.append(value[0])

            self.data[category] = self.data[category].apply(convert)

    
    def stacked_group(self, category, limit, calculation):
        """
        stacked_group: Stack multiple one hot encoded variables from the same group together.

        Parameters:
            - category (string) The category name of the one hot encoded variables
            - limit (int) The limit of variables to combine
            - calculation (string) Calculation to select the top given by limit. For 'average' the absolute average is used. For 'sum' the absolute sum is used.
        """
        data = self.data.copy()

        calculated = {}
        cat_idx = []
        shap_values = self.shap_values.squeeze()

        for i in range(len(data.columns)):
            if data.columns[i].startswith(f"{category}_"):
                cat_idx.append(i)

        for i in cat_idx:
            shap_col = np.abs(shap_values[:, i])
            column_name = data.columns[i]
            if calculation == 'average':
                calculated[column_name] = shap_col.mean()
            elif calculation == 'sum':
                calculated[column_name] = shap_col.sum()

        sorted_values = sorted(calculated.items(), key=lambda x: x[1], reverse=True)
        top_values = sorted_values[:limit]

        stacked_list = []
        for i, _ in top_values:
            values = data[i].reset_index(drop=True)
            temp = pd.DataFrame({
                category: [i] * len(values)
            })
            stacked_list.append(temp)
            stacked_cat = pd.concat(stacked_list, axis=0, ignore_index=True)

        shap_list = []
        for i, _ in top_values:
            idx = data.columns.get_loc(i)
            shap_col = self.shap_values[:, idx]
            shap_list.extend(shap_col)

        stacked_shap_cat = np.array(shap_list)
        stacked_shap_cat = stacked_shap_cat.reshape(-1, 1)

        idx_list = []
        for i, _ in top_values:
            idx = data.columns.get_loc(i)
            idx_list.append(idx)

        stacked_shap_data = np.delete(shap_values, idx_list, axis=1)
        stacked_shap_data = np.concatenate([stacked_shap_data] * len(top_values), axis=0)
        self.shap_values = np.concatenate([stacked_shap_data, stacked_shap_cat], axis=1)

        data = data.drop(data.columns[idx_list], axis=1)
        data = pd.concat([data] * len(top_values), ignore_index=True)
        self.data = pd.concat([data, stacked_cat], axis=1)
    
    
    def plot_summary(self, type_of, max_features, colour):
        if type_of == 'summary':
            if colour == None:
                shap.summary_plot(self.shap_values, self.data, max_display=max_features)
            else:
                shap.summary_plot(self.shap_values, self.data, cmap=colour, max_display=max_features)
        
    def plot_cohorts(self, n_cohorts):
        shap.plots.bar(self.explainer.cohorts(n_cohorts).abs.mean(0))
    
    def plot_bar(self, max_features):
        shap.bar_plot(abs(self.shap_values).mean(0), self.data, max_display=max_features)

    def plot_dependence(self, variable, interaction):
        if interaction == None:
            idx = self.data.columns.get_loc(variable)
            shap.dependence_plot(idx, self.shap_values, self.data)
        
    def restore_data(self):
        self.data = self.data_main


class SHAPmulti:
    def __init__(self, shap_values, explainer, data, feature_names=None):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            self.data_main = pd.DataFrame(data, columns=feature_names)
            self.shap_values = shap_values.squeeze()
            self.data = self.data_main.copy()
        else:
            self.shap_values = shap_values
            self.explainer = explainer
            self.data_main = data
            self.data = self.data_main.copy()


class SHAPmanager(SHAPbinary, SHAPmulti):
    def __init__(self, shap_values, explainer, data, feature_names=None, label_type=None):
        if label_type == 'binary':
            # Initialize the SHAPbinary part
            SHAPbinary.__init__(self, shap_values, explainer, data, feature_names)
        else:
            # Initialize the SHAPmulti part
            SHAPmulti.__init__(self, shap_values, explainer, data, feature_names)