import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch.nn as nn
import torch.optim as optim
import os
from imblearn.over_sampling import SMOTE
import joblib
import time
from tqdm import tqdm
import torch

class NNsmote(nn.Module):
    """
    NNsmote: A Neural Network which uses SMOTE upsampling from imb-learn in the training pass to handle imbalanced classes. The process upsamples all minority classes
    to the majority class by synthetically imputing values using a given number of neighbours. The number of neighbors influences the position of the synthetically 
    created samples of the minority class as SMOTE estimates new points based on existing samples and their nearest neighbors.

    Initialisation:
        - n_features (int) The number of features used in the data.
        - fc_size (int) The size of the dense layer which directly impacts complexity into shap calculations (weights | parameters).
        - device (torch.device) The device to calculate the tensor processes (currently only tested with cpu).
        - save_dir (string) The path to save the NN object. Saves all the attributes and parameters for a given epoch.
        
    Attributes:
        - criterion (nn.BCEWithLogitsLoss) Applies a probability conversion for the output and includes sigmoid activation.
        - device (torch.device) The given device for the tensor calculations.
        - save_dir (string) The given path used to save the model.
        - fc1 (int) The size of the expanded dense layer.
        - fc2 (int) The contracted layer (1) for the sigmoid calculation.
        - activation1 (nn.ReLU) The activation function for the expanded dense layer (deactivates all negatives to 0).
        - to (self.device) The device that is binded to the NN.
        - train_loss (list) The training loss stored for each epoch.
        - train_accuracy (list) The training accuracy stored for each epoch.
        - train_f1 (list) The training macro f1 score stored for each epoch.
        - test_loss (float) The test loss stored for the test run.
        - test_accuracy (float) The test accuracy stored for the test run.
        - test_f1 (float) The test macro f1 score stored for the test run.
        - test_predicted (list) The test predicted output stored as a list.
        - epoch_time (list) The time taken for each epoch using time module.

    NOTE: Additional attributes can be added, For Example: model.fc3 = nn.Linear. This also translates to the forward function by defining a forward function
        and using model.forward = forward.__get__(model).

    """
    def __init__(self, n_features, fc_size, device, save_dir):
        super(NNsmote, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.save_dir = save_dir
        self.fc1 = nn.Linear(n_features, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)
        self.activation1 = nn.ReLU()
        self.to(self.device)
        self.train_loss = []
        self.train_accuracy = []
        self.train_f1 = []
        self.test_loss = None
        self.test_accuracy = None
        self.test_f1 = None
        self.test_predicted = []
        self.epoch_time = []

    def forward(self, x):
        """
        forward: The forward pass. A basic dense layer specified by the size given in init. Returns a binary sigmoid which calculates the probability
        that it is either a 0 or a 1 (0.5 > = 1).

        Parameters:
            - x (tensor) The batch size and the number of features.

        Returns:
            - x (tensor) A float value that is converted by the sigmoid output of either 0 or 1.
        
        """
        x = self.activation1(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_accuracy_f1(self, predictions, labels):
        """
        calculate_accuracy_f1: A function that is used in the run method which calculates and returns overall accuracy and macro f1 score for the training and
        test run at the end of each epoch.

        Parameters:
            - predictions (tensor) A tensor array of the current predictions for the epoch.
            - labels (tensor) The labels that are passed and collected from each batch at the end of the epoch.
        
        Returns:
            - accuracy (float) The overall accuracy.
            - f1 (float) The macro f1 score.

        """
        preds = (predictions > 0.5).float()
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        return accuracy, f1

    def run(self, train_loader, learning_rate, epochs, save_factor, n_neighbours=2):
        """
        run: The run function that trains the model. Includes the SMOTE upsampling for each batch. If the upsampler fails to find more than one class in the batch,
        the process fails and defaults to the standard sample and prints an error message. This becomes more likely as the batch size is decreased. The training run
        uses shuffling from a DataLoader which gives a different result for each epoch meaning that even when the model converges, it is still able to get a different 
        result. Each epoch is saved based on the given save factor and the model can be loaded for each saved epoch to evaluate the result with the test set. The model 
        uses nn.BCEWithLogitsLoss as the criterion and optim.Adam with the given starting learning rate which is adjusted dynamically by the class. Prints the accuracy 
        and f1 score for each epoch along with tqm progress bars.

        Parameters:
            - train_loader (DataLoader) A DataLoader object which batches and shuffles the data and feeds it into the network.
            - learning_rate (float) The learning rate for the Adam optimiser. A starting learning rate of between 0.05-0.1 has been found to be optimal so far.
            - epochs (int) The number of epochs to train the model on (with a learning rate of 0.1 the model tend to converge on the first epoch).
            - save_factor (int) The factor which decides the interval epoch to save.
            - n_neighbours (int) The number of neighbours to use for SMOTE upsampling.
        
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            start_time = time.time()
            self.train()
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_f1 = 0
            num_batches = len(train_loader)

            for _, (train_seq, train_label) in tqdm(
                enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):

                train_seq, train_label = train_seq.to(self.device), train_label.to(self.device)
                batch_size = len(train_seq)
                smote_neighbors = min(batch_size, n_neighbours)

                if batch_size > 1:
                    try:
                        smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
                        train_smote, train_label_smote = smote.fit_resample(
                            train_seq.cpu().numpy(), train_label.cpu().numpy())
                        train_smote = torch.tensor(train_smote, dtype=torch.float32).to(self.device)
                        train_label_smote = torch.tensor(train_label_smote, dtype=torch.float32).to(self.device)
                    except ValueError:
                        print('Error running SMOTE.')
                        train_smote, train_label_smote = train_seq, train_label
                else:
                    train_smote, train_label_smote = train_seq, train_label

                optimizer.zero_grad()
                outputs = self(train_smote)
                loss = self.criterion(outputs.squeeze(), train_label_smote)
                loss.backward()
                optimizer.step()

                acc, f1 = self.calculate_accuracy_f1(outputs, train_label_smote)
                epoch_loss += loss.item()
                epoch_accuracy += acc
                epoch_f1 += f1

            self.train_loss.append(epoch_loss / num_batches)
            self.train_accuracy.append(epoch_accuracy / num_batches)
            self.train_f1.append(epoch_f1 / num_batches)
            self.epoch_time.append(time.time() - start_time)

            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {self.train_loss[epoch]:.4f}, Train Acc: {self.train_accuracy[epoch]:.4f}, Train F1: {self.train_f1[epoch]:.4f}")

            if (epoch + 1) % save_factor == 0:
                self.save_model(epoch + 1)

    def test(self, test_loader):
        """
        test: The testing run for the model. After a model has been trained call the test function to see how well it can predict the test set. Prints the
        test accuracy and macro f1 at the end of the run and stores them as attributes.

        Parameters:
            - test_loader (DataLoader) The DataLoader test set which is not shuffled or upsampled.

        """
        self.eval()
        self.test_predicted = []
        test_loss = 0
        test_accuracy = 0
        test_f1 = 0
        num_batches = len(test_loader)

        with torch.no_grad():
            for test_seq, test_label in tqdm(test_loader, total=num_batches, desc="Testing", unit="batch"):
                test_seq, test_label = test_seq.to(self.device), test_label.to(self.device)
                outputs = self(test_seq)
                loss = self.criterion(outputs.squeeze(), test_label)
                accuracy, f1 = self.calculate_accuracy_f1(outputs, test_label)
                test_loss += loss.item()
                test_accuracy += accuracy
                test_f1 += f1
                preds = (outputs > 0.5).float()
                self.test_predicted.extend(preds.cpu().numpy())

        self.test_loss = test_loss / num_batches
        self.test_accuracy = test_accuracy / num_batches
        self.test_f1 = test_f1 / num_batches

        print(f"Test Loss: {self.test_loss:.4f}, Test Accuracy: {self.test_accuracy:.4f}, Test F1: {self.test_f1:.4f}")

    def save_model(self, epoch):
        """
        save_model: Saves the models at each epoch given by the factor in the run method. Saves the whole object including all current attributes using
        joblib library.

        Parameters:
            - epoch (int) The current epoch which is being saved.

        """
        path = os.path.join(self.save_dir, f"PB_epoch_{epoch}.joblib")
        joblib.dump(self, path)

    def load_model(self, path):
        """
        load_model: Load a previously saved model for a given epoch. The epoch is based on the save file name. For Example: PB_epoch_1. To load a model
        the function requires using an existing or new NNsmote class.

        Parameters:
            - path (string) The path of the saved model to restore.

        Returns:
            - (NNsmote) The restored trained model.

        """
        model = joblib.load(path)
        self.to(self.device)
        return model
    
def plot_confusion_matrix(actual_labels, predicted_labels):
    """
    Plot confusion matrix: Plot a confusion matrix using the results of the NNsmote class.
    
    Parameter:
    - actual_labels (tensor) The true labels of the data.
    - predicted_labels (tensor) The predicted labels from model.predicted_labels.

    """
    cm = confusion_matrix(actual_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_metrics(metric, epochs, title):
    """
    Plot the training metrics for the NNsmote class.

    Parameter:
    - metric (list) Training metric (e.g. model.train_loss).
    - epochs (int) The number of epochs.
    - title (string) The title of the plot (imputes the category with type of plot).
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, epochs + 1), metric, color=sns.color_palette('pastel')[0])
    plt.xlabel('Epochs')
    plt.ylabel(f"{title}")
    plt.title(f"Training {title}")
    plt.ylim(0, 1)
    plt.show()