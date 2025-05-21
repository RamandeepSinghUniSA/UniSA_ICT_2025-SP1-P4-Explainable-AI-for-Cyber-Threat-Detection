import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

class SHAPmanager:
    """
    SHAPmanager: A wrapper that handles multiple shap evaluation tasks. Automatically links the variable names to the correct shap array, handles differing parameters for
    visualisations, has grouping functions for ordinal and one hot encoded features, filtering of features, selecting labels, and restoring the data to the original form. 
    Makes managing tasks easy without the large overhead of coding indexes.

    Known Errors:
        - After restoring data make sure to set the label in multiclass using change_label function.
    
    NOTE: This is a working progress..

    """
    def __init__(self, explainer, label_type=None, feature_encoder=None, label_encoder=None, feature_names=None, set_label=None):
        """
        Initiaisation: Handles different types of data. If using a tensor automatically adds the feature names and converts to a DataFrame. Select either the binary
        or multi-class version for evaluation.

        Parameters:
            - explainer (shap.Explainer) The explainer object which stores the shap values and corresponding data.
            - label_type (string) Either 'multi' for multi class or 'binary' for binary class data.
            - feature_encoder (list) The list that holds the label encoders for category variables used for Tree Models.
            - label_encoder (LabelEncoder) The LabelEncoder object used for the labels in multi-class Tree Models.
            - feature_names (list) The list of feature names if using a tensor from data.columns.
            - set_label (string) Preset to a given label if multi-class uses the LabelEncoder to map the corresponding shap values and squeeze the dimension.

        Attributes:
            - binary (boolean) The binary flag used for some transform operations.
            - tree_flag (boolean) Used when setting the label in binary Tree Models.
            - explainer (shap.Explainer) The explainer object used for operations.
            - explainer_data (pd.Dataframe) The extra dataset used in the restore data function.
            - explainer_values (numpy) The shap values used in the restore data function.
            - label (string) The label used in visualisation functions.
            - feature_encoder (LabelEncoder) The label encoder used to convert label encoded features to strings and other mapping functions.
            - label_encoder (LabelEncoder) The label encoder used to switch target labels for multi-class Tree Models.

        """

        if label_type == 'binary':
            if isinstance(explainer.data, torch.Tensor):
                self.binary = True
                self.tree_flag = False
                explainer.data = explainer.data.numpy()
                explainer.data = pd.DataFrame(explainer.data, columns=feature_names)
                self.explainer = explainer
                self.explainer_values= explainer.values.copy()
                self.explainer_data = explainer.data.copy()
                self.label = 0
            else:
                self.binary = True
                self.tree_flag = True
                explainer.data = pd.DataFrame(explainer.data, columns=explainer.feature_names)
                self.explainer = explainer
                self.explainer_values = explainer.values.copy()
                self.explainer_data = explainer.data.copy()
                if feature_encoder == None:
                    print('Error: Must provide feature_encoder for binary Tree Models.')
                self.feature_encoder = feature_encoder
                if set_label == 'Normal':
                    self.label = 0
                else:
                    self.label = 1

        if label_type == 'multi':
            self.binary = False
            explainer.data = pd.DataFrame(explainer.data, columns=explainer.feature_names)
            self.explainer = explainer
            self.explainer_values = explainer.values.copy()
            self.explainer_data = explainer.data.copy()
            if feature_encoder == None or label_encoder == None or set_label == None:
                print('Error: Must provide feature_encoder, label_encoder, and set_label for multi-class.')
                return
            self.feature_encoder = feature_encoder
            self.label_encoder = label_encoder
            self.label = (list(label_encoder.classes_).index(set_label))
        plt.style.use('default')    


    def custom_group(self, category, type_of='label-encoded', limit=30, calculation='average'):
        """
        custom_group: Convert a group of categories based on top 30 for label encoded categories. This helps to handle large cardinality categories when displaying them in
        shap visualisations. The conversion occurs just for the explainer.data and is automatically mapped to the shap values in the visualisations.

        Parameters:
            - category (string) The name of the category column.
            - type_of (string) For 'label-encoded-specific' provide a list of values to encode. For 'label-encoded' provide a limit and calculation to encode.
            - limit (integer) The total number of values to convert (rest are converted to 'Other').
            - calculation (string) Calculation to select the top values given by limit. For 'average' the absolute average is used. For 'sum' the absolute sum is used.

        """
        def convert(x):
            if x in category_values:
                return x
            else:
                return 'Other'

        if type_of == 'label-encoded-specific':
            data = self.explainer.data.copy()
            cat, values = category
            category_values = set(values)
            data[cat] = data[cat].apply(convert)
            self.explainer.data = data

        if type_of == 'label-encoded':
            data = self.explainer.data.copy()
            cat_idx = data.columns.get_loc(category)
            shap_values = np.abs(self.explainer.values[:, cat_idx, self.label])
            calculated = {}
            unique_vals = data[category].unique()
            for i in range(len(unique_vals)):
                value = unique_vals[i]
                subset = shap_values[data[category] == value]
                if calculation == 'average':
                    calculated[value] = subset.mean()
                elif calculation == 'sum':
                    calculated[value] = subset.sum()

            sorted_values = sorted(calculated.items(), key=lambda x: x[1], reverse=True)
            top_values = sorted_values[:limit]
            category_values = []

            for i in range(len(top_values)):
                category_values.append(top_values[i][0])

            self.explainer.data[category] = self.explainer.data[category].apply(convert)

    def stacked_group(self, category, limit=30, calculation='average'):
        """
        stacked_group: Stack multiple one-hot encoded variables from the same group together. This helps to visualise multiple one hot encoded features on the same
        visualisation. Stacks the remaining shap values and data for mapping in visualisations. The whole data is multiplied by the given limit so it can be computationally
        expensive.

        Parameters:
            - category (string) The category name of the one hot encoded variables (provide just the preceding name).
            - limit (integer) The limit of variables to stack.
            - calculation (string) Calculation to select the top features by limit. For 'average' the absolute average is used. For 'sum' the absolute sum is used.

        """
        data = self.explainer.data.copy()
        shap_values = self.explainer.values.squeeze()

        cat_idx = []
        for i in range(len(data.columns)):
            if data.columns[i].startswith(f"{category}_"):
                cat_idx.append(i)
        calculated = {}
        for i in cat_idx:
            shap_col = np.abs(shap_values[:, i])
            if calculation == 'average':
                calculated[i] = shap_col.mean()
            elif calculation == 'sum':
                calculated[i] = shap_col.sum()

        sorted_values = sorted(calculated.items(), key=lambda x: x[1], reverse=True)
        top_values = sorted_values[:limit]

        stacked_list = []
        for i, _ in top_values:
            values = data.iloc[:, i].reset_index(drop=True)
            temp = pd.DataFrame({
            category: [data.columns[i]] * len(values)
            })
            stacked_list.append(temp)

        stacked_cat = pd.concat(stacked_list, axis=0, ignore_index=True)

        shap_list = []
        for i, _ in top_values:
            shap_col = shap_values[:, i]
            for val in shap_col:
                shap_list.append(val)

        stacked_shap_cat = np.array(shap_list).reshape(-1, 1)

        idx_list = []
        for i, _ in top_values:
            idx_list.append(i)
    
        stacked_shap_data = np.delete(shap_values, idx_list, axis=1)

        stacked_shap_data = np.concatenate([stacked_shap_data] * len(top_values), axis=0)
        stacked_shap_data = np.concatenate([stacked_shap_data, stacked_shap_cat], axis=1)

        self.explainer.values = stacked_shap_data.reshape(-1, stacked_shap_data.shape[1], 1)
        data = data.drop(data.columns[idx_list], axis=1)
        data = pd.concat([data] * len(top_values), ignore_index=True)
        self.explainer.data = pd.concat([data, stacked_cat], axis=1)
        
    def plot_dependence(self, variable, n_interaction=None, colour=None, specific_interaction=None,  x_jitter=0,
                        xmin=None, xmax=None, ymin=None, ymax=None, alpha=1):
        """
        plot_dependence: A wrapper for the shap dependency plots which includes the original parameters with some additional ones. The purpose of this wrapper
        is to simplify the process of creating visualisations by taking care of indexing and sub-plots along with other functions.

        Parameters:
            - variable (string) The name of the variable to draw interactions to for the current label.
            - n_interaction (int) Finds a number of interactions based on approximate interactions. Can be set to either 1 or a multiple of 2. Creates subplots if multiple.
            - colour (string) Add a cmap colour can be provided to customise the plots.
            - specific_interaction (string) Select a specific variable as the interaction index.
            - x_jitter (float) Adds jitter to the X axis values which sometimes helps visualise tight clusters. Useful for discreet features. A float value between (0.0 - 1.0).
            - xmin (percentile(int)) A cutoff point for outliers based on the left sided percentile on the X axis given as percentile(int).
            - xmax (percentile(int))  A cutoff point for outliers based on the right sided percentile on the X axis given as percentile(int).
            - ymin (percentile(int)) A cutoff point for outliers based on the left sided percentile on the Y axis interaction given as percentile(int).
            - ymax (percentile(int)) A cutoff point for outliers based on the right sided percentile on the Y axis interaction given as percentile(int).
            - alpha (float) The amount of transparency for the points which can help visualise tight clusters. A float value between (0.0 - 1.0).

        """
        var = self.explainer.data.columns.get_loc(variable)
        if specific_interaction: 
            si = self.explainer.data.columns.get_loc(specific_interaction)
            if colour:
                shap.dependence_plot(var, self.explainer.values[:, :, self.label], self.explainer.data, interaction_index=si,
                cmap=plt.colormaps[colour], show=True, x_jitter=x_jitter, xmin=xmin, xmax=xmax, ymin=ymin,
                ymax=ymax, display_features=None, alpha=alpha)
                return
            else:
                shap.dependence_plot(var, self.explainer.values[:, :, self.label], self.explainer.data, interaction_index=si,
                show=True, x_jitter=x_jitter, xmin=xmin, xmax=xmax, ymin=ymin,
                ymax=ymax, display_features=None, alpha=alpha)
                return

        if (n_interaction == 1 or n_interaction % 2 == 0):
            idx = shap.approximate_interactions(self.label, self.explainer.values[:, :, self.label], self.explainer.data)
            if n_interaction == 1:
                if colour:
                    shap.dependence_plot(var, self.explainer.values[:, :, self.label], self.explainer.data, interaction_index=idx[0], 
                    cmap=plt.colormaps[colour], x_jitter=x_jitter, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, display_features=None, alpha=alpha)
                else:
                    shap.dependence_plot(var, self.explainer.values[:, :, self.label], self.explainer.data, interaction_index=idx[0], 
                    x_jitter=x_jitter, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, display_features=None, alpha=alpha)
            else:
                n_rows = n_interaction / 2
                fig, axes = plt.subplots(nrows=int(n_rows), ncols=2, figsize=(14, 6 * n_rows))
                axes = axes.flatten()

                for i in range(n_interaction):
                    if colour:
                        shap.dependence_plot(var, self.explainer.values[:, :, self.label], self.explainer.data,
                        interaction_index=idx[i], ax=axes[i], show=False, cmap=plt.colormaps[colour], 
                        x_jitter=x_jitter, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, display_features=None, alpha=alpha)
                    else:
                        shap.dependence_plot(var, self.explainer.values[:, :, self.label], self.explainer.data,
                        interaction_index=idx[i], ax=axes[i], show=False, x_jitter=x_jitter, 
                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, display_features=None, alpha=alpha)
                          
                plt.tight_layout()
                plt.show()
            plt.rcdefaults()

    def plot_summary(self, max_features=None, colour=None, multi=None, features=None):
        """
        plot_summary: Plots the standard summary plot for shap. Alternatively use the multi function to compare more than one label using absolute averages. 

        Parameters:
            - max_features (int) The maximum number of features to plot.
            - colour (string) Option to change the colour using any cmap colours.
            - multi (string) Used in multi-class to plot more than one label. Either 'All' for all labels or a list with the label names to plot. Automatically plots the current label.
            - features (list) A list of specific features to plot.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if multi is not None:
                if multi == 'All':
                    shap.summary_plot(self.explainer, color=plt.colormaps['tab10'], class_names=self.label_encoder.classes_)
                    return
                else:
                    names = []
                    names.append(self.label_encoder.classes_[self.label])
                    labels = []
                    labels.append(self.label)
                    for l in multi:
                        idx = list(self.label_encoder.classes_).index(l)
                        labels.append(idx)
                        names.append(l)
                    shap.summary_plot(self.explainer, class_inds=labels, color=plt.colormaps['tab10'], class_names=self.label_encoder.classes_)
                    return
            if features is not None:
                idx = []
                for i in features:
                    c = self.explainer.data.columns.get_loc(i)
                    idx.append(c)

                if colour is None:
                    shap.summary_plot(self.explainer.values[:, idx, self.label], self.explainer.data[features], max_display=max_features)
                    return
                else:
                    shap.summary_plot(self.explainer.values[:, idx, self.label], self.explainer.data[features], cmap=colour, max_display=max_features)
                    return
            if colour is None:
                shap.summary_plot(self.explainer.values[:, :, self.label], self.explainer.data, max_display=max_features)
            else:
                shap.summary_plot(self.explainer.values[:, :, self.label], self.explainer.data, cmap=colour, max_display=max_features)

    def string_encode(self, categories):
        """
        string_encode: Uses the feature encoder to add strings back to the label encoded categories for Tree Models.

        Parameters:
            - categories (list) A list of the categorical features that were used in the LabelEncoder.

        """

        i = 0
        for c in categories:
            le = self.feature_encoder[i]
            self.explainer.data[c] = le.inverse_transform(self.explainer.data[c].astype(int))
            i += 1

    def restore_data(self):
        """
        restore_data: Restore the data to its original form.

        NOTE: If using multi-class data the change_label function must be used after restoring the data.

        """
        self.explainer.values = self.explainer_values.copy()
        self.explainer.data = self.explainer_data.copy()

    def remove_features(self, features):
        """
        remove_features: Remove a set of features from the data and shap values along with it.

        - Parameters:
            - features (list) The list of features to remove.
 
        """
        feature_idx = []
        for i in features:
            if i in self.explainer.data.columns:
                idx = self.explainer.data.columns.get_loc(i)
                feature_idx.append(idx)
    
        self.explainer.values = np.delete(self.explainer.values, feature_idx, axis=1)
        self.explainer.data = self.explainer.data.drop(columns=features)

    def set_label(self, label):
        """
        set_label: Primarily used for multi-class Tree Models to change the label to evaluate. Can also be used with the binary Tree Model however the plots will just show the
        inverse and don't reveal much more information.

        Parameters:
            - label (string) For multi-class provide a label name to change it to. For binary Tree Models provide either 'Threat' or 'Normal'.

        NOTE: This function must be  used after the restore data function in multi-class Tree Models.
        
        """
        if self.binary == True:
            if self.tree_flag == True:
                if label == 'Threat':
                    self.label = 1
                if label == 'Normal':
                    self.label = 0
                return
        if self.binary == False:
            self.label = list(self.label_encoder.classes_).index(label)