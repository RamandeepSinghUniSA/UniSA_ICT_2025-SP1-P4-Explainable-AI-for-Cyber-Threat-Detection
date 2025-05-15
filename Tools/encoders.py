import pandas as pd
import numpy as np
import tensorflow as tf


class CorrOnehotEncoder:
    """
    CorrOnehotEncoder: Encodes the given column by creating one-hot encoded columns for categories that have
    a correlation higher than a threshold with the target column.

    """
    def __init__(self, column, target):
        """
        Parameters:
            - column (pd.Series) The feature column to encode.
            - target (pd.Series) The target column.

        """
        # Force to string for groups.
        self.column = column.astype(str)
        # Convert to float32 precision to minimise memory load.
        self.target = target.astype(np.float32)

    def corr(self, x, y):
        """
        Calculate the Pearson correlation coefficient (Phi).
        
        Parameters:
            - x (tensor - float32) The first variable.
            - y (tensor - float32) The target to draw correlation to.
        
        Returns:
            - r (float32) The Pearson correlation coefficient (Phi).

        """
        mean_x = tf.reduce_mean(x)
        mean_y = tf.reduce_mean(y)
        covariance = tf.reduce_sum((x - mean_x) * (y - mean_y))
        std_x = tf.sqrt(tf.reduce_sum((x - mean_x) ** 2))
        std_y = tf.sqrt(tf.reduce_sum((y - mean_y) ** 2))
        r = covariance / (std_x * std_y)
        return r

    def encode(self, sparse_n, threshold, max_encoded):
        """
        Encode the feature column by creating one-hot encoded columns for categories that have
        a correlation higher than a threshold with the target.
        
        Parameters:
            - sparse_n (int) Minimum number of occurrences (1's) for a category in the column.
            - threshold (float) The correlation threshold.
            - max_encoded (int) The maximum number of encoded features.
        
        Returns:
            - ohe_df (pd.DataFrame) One-hot encoded columns that meet the correlation threshold.

        """
        # Convert to numpy for tensors.
        column_np = self.column.to_numpy()
        target_np = self.target.to_numpy()

        # Store results.
        ohe_list = []    
        column_names = []
        correlations = []
        # Iterate through each unique category in the column.
        for c in np.unique(column_np):
            # Convert to binary - float32 minimises memory issues.
            corr_column = (column_np == c).astype(np.float32)
            # If the category count is below sparse_n, skip encoding.
            if np.sum(corr_column) < sparse_n:
                continue
            # Convert to tensors for the correlation calculation.
            correlation = self.corr(tf.convert_to_tensor(corr_column, dtype=tf.float32), 
                                    tf.convert_to_tensor(target_np, dtype=tf.float32))
            # If the absolute correlation is greater than the threshold, add to the list.
            if abs(correlation.numpy()) > threshold:
                ohe_list.append(corr_column)
                column_names.append(f"{self.column.name}_{c}")
                # Store correlations to sort.
                correlations.append(abs(correlation.numpy()))

        # Sort the columns by their correlation with the target.
        sorted_indices = np.argsort(correlations)[::-1]
        sorted_ohe_list = []
        sorted_column_names = []
        for i in sorted_indices:
            sorted_ohe_list.append(ohe_list[i])
            sorted_column_names.append(column_names[i])

        # Limit the number of variables to max_encoded.
        if len(sorted_ohe_list) > max_encoded:
            sorted_ohe_list = sorted_ohe_list[:max_encoded]
            sorted_column_names = sorted_column_names[:max_encoded]
        # Add the encoded data to a dataframe.
        ohe_df = pd.DataFrame(np.column_stack(sorted_ohe_list), columns=sorted_column_names)
        
        if ohe_df.empty:
            print("No correlations exceed the threshold.")
            return pd.DataFrame()
        
        return ohe_df
    

class CorrVarEncoder:
    """
    CorrThresholdEncoder: Encodes a given column based on a correlation threshold. All values within the variable that fall below the threshold are
    converted to a given string name. For Example: LowThreshold.
    
    NOTE: It is recommended to include the threshold used in the new value name.

    """
    def __init__(self, column, target):
        """
        Parameters:
            - data (pd.Series) The column that contains the feature.
            - target (pd.Series) The target column to draw correlation to.

        """
        self.column = column.astype(str)
        self.target = target.astype(np.float32)

    def corr(self, x, y):
        """
        Calculate the Pearson correlation coefficient (Phi).
        
        Parameters:
            - x (tensor - float32) The first variable.
            - y (tensor - float32) The target to draw correlation to.
        
        Returns:
            - r (float32) The Pearson correlation coefficient (Phi).

        """
        mean_x = tf.reduce_mean(x)
        mean_y = tf.reduce_mean(y)
        covariance = tf.reduce_sum((x - mean_x) * (y - mean_y))
        std_x = tf.sqrt(tf.reduce_sum((x - mean_x) ** 2))
        std_y = tf.sqrt(tf.reduce_sum((y - mean_y) ** 2))
        r = covariance / (std_x * std_y)
        return r

    def encode(self, threshold, value_name, sparse_n, max_encoded):
        """
        encode: Takes the column to encode and computes the correlation with the target column. If the correlation is below the threshold, 
        the category value is replaced with a specified value name. It also filters categories based on the sparse_n condition.
        
        Parameters:
            - threshold (float) The threshold for correlation. The function creates onehot encoded columns of all variables that have correlation
              higher than the threshold to the target label.
            - value_name (str) The value to replace categories with low correlation.
            - sparse_n (int) The minimum number of occurrences for a category to be considered.
            - max_encoded (int) The maximum number of categories.
        
        Returns:
            - column (pd.Series) The converted column.
        """
        corr_dict = {}

        # Go through each unique category in the column.
        for c in self.column.unique():
            corr_column = (self.column == c).astype(np.float32)
            num_ones = corr_column.sum()
            # Set category to value name if below sparse_n.
            if num_ones < sparse_n:
                self.column[self.column == c] = value_name
                continue

            # Convert to tensors to minimise memory allocation.
            corr_column_tensor = tf.convert_to_tensor(corr_column, dtype=tf.float32)
            target_tensor = tf.convert_to_tensor(self.target, dtype=tf.float32)
            # Calculate the correlation with the target label.
            correlation = self.corr(corr_column_tensor, target_tensor)
            # Only add to the dictionary if the correlation is above the threshold.
            if abs(correlation.numpy()) >= threshold:
                corr_dict[c] = correlation.numpy()
            else:
                # If correlation is below threshold, mark as low correlation.
                self.column[self.column == c] = value_name

        # Sort categories for max_encoded.
        sorted_corr_dict = sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        limited_categories = []
        for item in sorted_corr_dict[:min(max_encoded-1, len(sorted_corr_dict))]:
            limited_categories.append(item[0])

        # Replace values that are not in the top 'max_encoded' categories.
        for c in self.column.unique():
            if c not in limited_categories:
                self.column[self.column == c] = value_name

        return self.column

    
class CorrBinEncoder:
    """
    CorrBinEncoder: Encodes a variable based on the correlation drawn to the given label based on the number of categories provided. The variable is binarised
    using the correlations with pd.cut.

    NOTE: pd.cut creates relative borders when binarising which means that at times even a value that is labelled as High might still only be low correlation (0.1).

    """
    def __init__(self, column, target):
        """
        Parameters:
            - data (pd.Series) The column that contains the feature.
            - target (pd.Series) The target column to draw correlation to.
            
        """
        self.column = column.astype(str)
        self.target = target.astype(np.float32)

    def corr(self, x, y):
        """
        Calculate the Pearson correlation (Phi).
        
        Parameters:
            - x (tensor - float32) The first variable.
            - y (tensor - float32) The target to draw correlation to.
        
        Returns:
            - r (float32) The Pearson correlation coefficient (Phi).
        """
        mean_x = tf.reduce_mean(x)
        mean_y = tf.reduce_mean(y)
        covariance = tf.reduce_sum((x - mean_x) * (y - mean_y))
        std_x = tf.sqrt(tf.reduce_sum((x - mean_x) ** 2))
        std_y = tf.sqrt(tf.reduce_sum((y - mean_y) ** 2))
        r = covariance / (std_x * std_y)
        return r

    def encode(self, bin_cut, bin_labels):
        """
        encode: Select a number of bins and corresponding labels to binarize the variable based on correlation to the label.

        Parameters:
            - column (string) The column to encode.
            - bin_cut (int) The number of bins to create based on pd.cut.
            - bin_labels (list) A list of strings to name the new values (High, Medium, Low) - must match the same number of bins.

        Returns:
            - encoded_column (pd.Series) The encoded column with correlation binned by categories.
        """
        corr_dict = {}

        # Go through each unique category in the column.
        for c in self.column.unique():
            # Create a binary column for each category.
            corr_column = (self.column == c).astype(np.float32)

            # Convert to tensors to minimise memory for corr calculation.
            corr_column_tensor = tf.convert_to_tensor(corr_column, dtype=tf.float32)
            target_tensor = tf.convert_to_tensor(self.target, dtype=tf.float32)
            correlation = self.corr(corr_column_tensor, target_tensor)
            corr_dict[c] = correlation.numpy()

        # Create a DataFrame for cut.
        corr_df = pd.DataFrame(list(corr_dict.items()), columns=['category', 'corr'])
        corr_df['abs_corr'] = corr_df['corr'].abs()
        # Binarise using pd.cut.
        corr_df['binned'] = pd.cut(
            corr_df['abs_corr'],
            bins=bin_cut,
            labels=bin_labels,
            include_lowest=True
        )
        # Map each category to its corresponding bin.
        category_to_bin = dict(zip(corr_df['category'], corr_df['binned']))
        encoded_column = self.column.map(category_to_bin)

        return encoded_column