import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class SHAPmanager:
    """
    SHAPmanager: A wrapper that handles multiple shap evaluation tasks. Automatically links the variable names to the correct shap array, handles differing parameters for
    visualisations, has grouping functions for ordinal and one hot encoded features, filtering of features, selecting labels, and restoring the data to the original form. 
    Makes managing tasks easy without the large overhead of coding indexes.

    Known Errors:
        - After restoring data make sure to set the label in multiclass using change_label function.
    
    NOTE: This is a working progress..

    """
    def __init__(self, shap_values, explainer, data, label_type=None, feature_encoder=None, label_encoder=None, feature_names=None, set_label=None):
        """
        Initiaisation: Handles different types of data. If using a tensor automatically adds the feature names and converts to a DataFrame. Select either the binary
        or multi-class version for evaluation.

        Parameters:
            - shap_values (numpy) The shap values array for either binary or multiclass predictions.
            - explainer (explainer) The explainer object from shap.
            - data (tensor, pd.DataFrame) The data to evaluate that is linked to the shap values.
            - label_type (string) Either 'multi' for multi class or 'binary' for binary class data.
            - feature_encoder (list) The list that holds the label encoders for category variables.
            - label_encoder (LabelEncoder) The LabelEncoder object used for the labels.
            - feature_names (list) The list of feature names if using a tensor from data.columns.
            - set_label (string) Preset to a given label if multi-class uses the LabelEncoder to map the corresponding shap values and squeeze the dimension.

        Attributes:
            - explainer (explainer) The explainer object.
            - xp_flag (boolean) Used for some explainer plot types (currently unused)
            - data (pd.DataFrame) The dataset that is used for all operations.
            - data_main (pd.DataFrame) The original dataset used to restore the working dataset.
            - shap_values (numpy) The shap values used for all operations (squeezed to two dimensions)
            - shap_values_main (numpy) The original shap values in either 2D or 3D form used to restore the working shap values.
            - label_encoder (LabelEncoder) The encoder object used to switch and set the label.
            - feature_encoder (list) The list of feature encoders used to set string values back to categories.

        """

        if label_type == 'binary':
            if isinstance(data, torch.Tensor):
                explainer.feature_names = feature_names
                self.explainer = explainer
                self.xp_flag = True
                data = data.numpy()
                self.data_main = pd.DataFrame(data.copy(), columns=feature_names)
                self.shap_values = shap_values.squeeze()
                self.shap_values_main = shap_values.copy().squeeze()
                self.data = self.data_main.copy()

            else:
                explainer
                self.explainer = explainer
                self.shap_values = shap_values
                self.shap_values_main = shap_values.copy()
                self.data_main = data.copy()
                self.data = self.data_main.copy()
        if label_type == 'multi':
            self.explainer  = explainer
            self.data_main = data
            self.data = self.data_main.copy()
            self.shap_values_main = shap_values
            self.shap_values = self.shap_values_main.copy()
            idx = list(label_encoder.classes_).index(set_label)
            self.shap_values = self.shap_values[:, :, idx]
            self.label_encoder = label_encoder
            self.feature_encoder = feature_encoder

    def custom_group(self, type_of, category, limit=None, calculation=None):
        """
        custom_group: Convert a group of categories based on top 30 for label encoded categories. This helps to handle large cardinality categories when displaying them in
        shap visualisations.

        Parameters:
            - type_of (string) For 'label-encoded-specific' provide a list of values to encode and change the rest to 'Other'. For 'label-encoded' provide a limit and calculation to calculate top 30.
            - category (string) The name of the category column.
            - limit (integer) The total number of values to convert (rest are converted to 'Other').
            - calculation (string) Calculation to select the top values given by limit. For 'average' the absolute average is used. For 'sum' the absolute sum is used.

        """
        def convert(x):
            """
            convert: The inner apply function to map the string name into the column or 'Other' if below the limit.

            Parameters:
                - x (numpy) The column array to apply the function to.

            Returns:
                - x | 'Other' (string) The name of the category column or 'Other' if below the limit.

            """
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
        stacked_group: Stack multiple one hot encoded variables from the same group together. This helps to visualise multiple one hot encoded features on the same
        visualisation. Stacks the remaining shap values for mapping.

        Parameters:
            - category (string) The category name of the one hot encoded variables
            - limit (integer) The limit of variables to combine
            - calculation (string) Calculation to select the top features by limit. For 'average' the absolute average is used. For 'sum' the absolute sum is used.

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
        """
        plot_summary: Plots the standard summary plot for shap.

        Parameters:
            - type_of (string) Currently just 'summary' is available but will be expanded to include others such as bar or waterfall.
            - max_features (integer) The maximum number of features to display.
            - colour (string) Option to change the colour using any cmap colouring schema.

        """
        if type_of == 'summary':
            if colour == None:
                shap.summary_plot(self.shap_values, self.data, max_display=max_features)
            else:
                shap.summary_plot(self.shap_values, self.data, cmap=colour, max_display=max_features)
        
    def plot_cohorts(self, n_cohorts):
        """
        plot_cohorts: Plots the cohorts given by the explainer object.
        
        Parameters:
            - n_cohorts (integer): The number of cohorts to display (setting this to more than 4 usually results in bad plots).

        """
        shap.plots.bar(self.explainer[..., 0].cohorts(n_cohorts).abs.mean(0))
    
    def plot_bar(self, max_features):
        """
        plot_bar: Plots the features in a bar plot based on the absolute average shap value.
        
        Parameters:
            - max_features (integer) The maximum number of features.

        """
        shap.bar_plot(abs(self.shap_values).mean(0), self.data, max_display=max_features)

    def plot_dependence(self, variable, n_interaction):
        var = self.data.columns.get_loc(variable)
        idx = shap.approximate_interactions(var, self.shap_values, self.data)
        if (n_interaction == 1 or n_interaction % 2 == 0):
            if n_interaction == 1:
                shap.dependence_plot(var, self.shap_values, self.data, interaction_index=idx[0])
            else:
                n_rows = n_interaction / 2
                fig, axes = plt.subplots(nrows=int(n_rows), ncols=2, figsize=(14, 6 * n_rows))
                axes = axes.flatten()

                for i in range(n_interaction):
                    shap.dependence_plot(var, self.shap_values, self.data,
                                         interaction_index=idx[i], ax=axes[i], show=False)
                          
                plt.tight_layout()
                plt.show()
            plt.rcdefaults()
        else:
            print('Error: Interaction Index must either be 1 or a multiple of 3.')
     
    def change_label(self, label):
        """
        change_label: Option to change the label if using multi-class data. Squeezes the shap values to the corresponding label.

        Parameters:
            - label (string) The label to change the data to.

        NOTE: This must be used after restore data.

        """
        shap_values = self.shap_values_main.copy()
        idx = list(self.label_encoder.classes_).index(label)
        self.shap_values = shap_values[:, :, idx]

    def string_encode(self, categories):
        """
        string_encode: Uses the feature encoder to add strings back to the label encoded categories.

        Parameters:
            - categories (list) A list of the categorical features encoded.

        """

        i = 0
        for c in categories:
            le = self.feature_encoder[i]
            self.data[c] = le.inverse_transform(self.data[c])
            i += 1

    def restore_data(self):
        """
        restore_data: Restore the data to its original form.

        NOTE: If using multi-class data the change_label function must be used after restoring the data.

        """
        self.data = self.data_main.copy()
        self.shap_values = self.shap_values_main.copy()

    def remove_features(self, features):
        """
        remove_features: Remove a set of features from the data and shap values.

        - Parameters:
            - features (list) The list of features to remove.
 
        """
        feature_idx = []
        for i in features:
            idx = self.data.columns.get_loc(i)
            feature_idx.append(idx)

        self.shap_values = np.delete(self.shap_values, feature_idx, axis=1)
        self.data = self.data.drop(columns=features, axis=1)