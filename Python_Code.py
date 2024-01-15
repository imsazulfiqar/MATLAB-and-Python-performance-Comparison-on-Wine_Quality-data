#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from time import time
from datetime import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn import svm


# In[2]:


# Load the dataset through the URL link 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, delimiter=";")




# In[3]:


df.head()


# In[40]:


df.describe()


# In[4]:


sns.countplot(x='quality', hue='quality', data=df, palette ="inferno")
plt.show()


# The original dataset has 7 classes from 3-9, the distribution plot shows how classes 6 and 5 combined account for a majority of the classes, naturally this will cause some imbalances in the dataset if this was left to be the way it is. 

# In[5]:


sns.boxplot(x= "quality", y="alcohol", hue="quality", data=df, palette="inferno", width=0.5)
plt.show()


# The boxplot above shows how several outliers exist in class 4, 5 and 8. However, this is a negligible amount and no transformations will be made to remove any effects of outliers in the distribution of the target class.

# In[6]:


# the correlation matrix is defined below, where the Pearson coefficient will be used
corr_matrix = pd.melt(df.corr(method='pearson').reset_index(),id_vars=['index'])


# In[7]:


# importing altair to create the correlation matrix 
import altair as alt


# In[8]:


base = alt.Chart(corr_matrix).encode(
    x=alt.X('index:N',scale=alt.Scale(paddingInner=0),sort=alt.EncodingSortField('index', order='ascending')),
    y=alt.Y('variable:N',scale=alt.Scale(paddingInner=0),sort=alt.EncodingSortField('index', order='ascending')),
)
# the colors of the heatmap will be scaled according to the values between -1 and 1, which are the Pearson coefficients of each tile
heatmap = base.mark_rect().encode(
    color=alt.Color('value:Q', scale=alt.Scale(domain=[-1,1],scheme='greenblue')),
    tooltip=['value','index','variable']
    )

# text will be the annotated values of the Pearson coefficient on each tile in the matrix
text = base.mark_text(baseline='middle').encode(
    alt.Text('value',format='.2f'),
    color=alt.value('white'),
    tooltip=['value','index','variable']
)

# combines the heatmap and text layers to create the final chart.
(heatmap + text).properties(width=360,height=360).configure_axis(title=None)


# In[9]:


print(df.dtypes)


# The data types of each column is printed, to understand which are floats and which are integers. The quality column is an integer as the values are between 3-9 as expected, but to make sure no complications occur in the modelling, it will be converted to a float, just like the rest.

# In[10]:


df['quality'] = df['quality'].astype('float64')


# In[11]:


print(df.dtypes)


# In[12]:


# another heatmap of the feature variables and the target variable is created 
fig, ax = plt.subplots (figsize=(30, 9))
corr = df.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, cmap=sns.cubehelix_palette(as_cmap=True), annot = True)
ax.xaxis.tick_top()
plt.show ()


# from the heatmap above, it is observed that some features have a high co-correlation with each other, however since we are doing a classification task and not a regression task, those features will not be excluded from the model. 

# In[13]:


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))

for i, column in enumerate(df.columns):
    ax = axes[i//4][i%4]
    sns.histplot(df[column], ax=ax, kde=True, color = 'teal')
    ax.set_xlabel(column)

fig.tight_layout()
plt.show()


# In[14]:


df[df.quality == '0'].shape[0]


# In[15]:


# Split the quality of the wine into groups of 2
df["quality"] = df["quality"].apply(lambda x: 1 if x > 5 else 0)


# No 0's in the quality column, dataset is cleaned and ready for creating some models.

# In[16]:


sns.countplot(x=df["quality"], palette = "inferno")


# In[17]:


# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1].values)


# In[18]:


from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df.quality==1]
df_minority = df[df.quality==0]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


# In[19]:


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_upsampled.iloc[:, :-1], df_upsampled['quality'], test_size=0.2)
# Print the shapes of the train and test sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[20]:


print(y_test.value_counts())


# In[21]:


print(y_train.value_counts())


# After grouping them into 2 classes, 

# In[22]:


# Define the MLP model using PyTorch
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super(MLP, self).__init__()

        # Define input layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layer_sizes[0])])

        # Define hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            self.layers.append(nn.Tanh())

        # Define output layer
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Define the SVM model using scikit-learn
class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0):
        self.clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)






# In[23]:


import time


# Here I import time to time the performance of the models in respect of the training and test speeds

# In[24]:


X_train.shape


# In[25]:


y_test.shape


# In[27]:


X_test = X_test.to_numpy()
y_test = y_test.to_numpy()


# The above code is to convert the data in X_test and y_test into numpy arrays, this is to avoid complications and errors in the cell below when the model is being ran. It only needs to be run once, an error message will occur if the kernel is not restarted but the cell above is ran again, for whatever reason. If an error occurs, simply restart the kernel and run all cells again.

# In[43]:


# Define the input and output sizes
input_size = X_train.shape[1]
output_size = 1

# Set the hyperparameters for the MLP model
lr = 0.00967008660641566
momentum = 0.7886278169208891
batch_size = 64
num_epochs = 200
patience = 50 # number of epochs to wait before early stopping
best_loss = np.inf # initialize the best loss to infinity
best_model = None # initialize the best model to None
early_stop_count = 0 # initialize the early stopping counter to 0
weight_decay = 0.008214754160560066 # add weight decay parameter

# Create the MLP model
mlp = MLP(input_size, output_size=1, hidden_layer_sizes=[34, 63, 25])

# Define the loss function for the MLP
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer for the MLP model
optimizer = optim.SGD(mlp.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay) # add weight decay parameter

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Split the training set into training and validation sets
X_train_, X_val, y_train_, y_val_ = train_test_split(X_train.values, y_train.values, test_size=0.2, random_state=42)

# Initialize the lists to store the training and testing losses, validation and testing accuracies
train_losses = []
test_losses = []
val_accuracies = []
test_accuracies = []

# Record the start time
start_time = time.time()

# Set the epoch interval to 10
epoch_interval = 10 

# Loop through the number of epochs
for epoch in range(num_epochs):
    # Put the MLP model in training mode
    mlp.train()

    # Reset the optimizer gradients
    optimizer.zero_grad()

    # Loop through the training set in batches
    for i in range(0, X_train_.shape[0], batch_size):
        batch_X = torch.tensor(X_train_[i:i+batch_size], dtype=torch.float32)
        batch_y = torch.tensor(y_train_[i:i+batch_size], dtype=torch.float32)

        # Forward pass through the MLP
        outputs = mlp(batch_X)

        # Calculate the loss
        loss = criterion(outputs.view(-1), batch_y)

        # Backward pass through the MLP
        loss.backward()

        # Update the MLP parameters
        optimizer.step()

        # Reset the optimizer gradients
        optimizer.zero_grad()

        # Store the training loss in the list
        train_losses.append(loss.item())

    # Put the MLP model in evaluation mode
    mlp.eval()

    # Turn off gradient tracking
    with torch.no_grad():
        # Forward pass through the MLP with the validation set
        outputs = mlp(torch.tensor(X_val, dtype=torch.float32))

        # Calculate the validation loss
        val_loss = criterion(outputs.view(-1), torch.tensor(y_val_, dtype=torch.float32))

        # Append the validation accuracy to the list
        val_accuracies.append(accuracy_score(y_val_, (outputs.view(-1) > 0).float().numpy()))

        # Adjust the learning rate based on the validation loss
        scheduler.step(val_loss)

        # Early stopping
        # Check if the validation loss has improved, and save the model parameters if it has
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model = mlp.state_dict()
            early_stop_count = 0
        else:
            # If the validation loss has not improved, increment the early stopping counter
            early_stop_count += 1
            # If the counter has reached the patience threshold, print a message and break out of the training loop
            if early_stop_count >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Stopping early...')
                break

        # Compute the loss and accuracy on the test set using the current model parameters
        outputs = mlp(torch.tensor(X_test, dtype=torch.float32))
        test_loss = criterion(outputs.view(-1), torch.tensor(y_test, dtype=torch.float32))
        test_losses.append(test_loss.item())
        y_pred = (outputs.view(-1) > 0).float().numpy()
        test_accuracy = accuracy_score(y_test, y_pred)
        test_accuracies.append(test_accuracy)

        # Only print summary after epoch_interval epochs or on the last epoch
        if (epoch+1) % epoch_interval == 0 or epoch == num_epochs-1:
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}, Val Accuracy={val_accuracies[-1]:.4f}, Test Accuracy={test_accuracy:.4f}, Time={epoch_time:.2f}s')

# Evaluate the best model on the test set
# Load the model parameters that achieved the best validation loss
mlp.load_state_dict(best_model)
# Set the model to evaluation mode, which disables dropout and other stochastic operations
mlp.eval()
# Use the model to make predictions on the test set
with torch.no_grad():
    outputs = mlp(torch.tensor(X_test, dtype=torch.float32))
    y_pred_mlp = (outputs.view(-1) > 0).float().numpy()
    # Compute the accuracy of the predictions and print it
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'Test Accuracy={accuracy_mlp:.4f}')
        


# In[29]:


import numpy as np

unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))


# In[30]:


unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))


# In[31]:


from sklearn.metrics import confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred_mlp)


# In[32]:


plt.figure(figsize=(12, 5))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.xlim(0, 100)
plt.ylabel('Loss')
plt.ylim(0.4, 0.8)

# Confusion matrix
plt.subplot(1, 2, 2)
conf_matrix = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()


# In[33]:


# using subplot function and creating plot one
plt.subplot(1, 2, 1)  # row 1, column 2, count 1
plt.plot(test_accuracies, 'g')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
 
# using subplot function and creating plot two
# row 1, column 2, count 2
plt.subplot(1, 2, 2)
 
# r is for red color
plt.plot(train_losses, 'r')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.xlim(0,150)
plt.ylabel('Training Loss')
plt.ylim(0.55,0.9)
 
# space between the plots
plt.tight_layout()
 
# show plot
plt.show()


# In[34]:


from sklearn.metrics import confusion_matrix, classification_report

# Classification report
class_report = classification_report(y_test, y_pred_mlp)
print('Classification Report:')
print(class_report)


# TO DO:
# Remove epochs from SVM model, they don't use epochs, they use iterations. 
# Ask what is degree, gamma, coef0 and C
# 

# In[35]:


C = 10
kernel = 'rbf'
degree = 3
gamma = 'auto'
coef0 = 0.0

svm_model = SVM(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
start_time = time.time() # start time

X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train, y_train, test_size=0.2) # split training set into training and validation sets
best_val_accuracy = 0.0 # initialize best validation accuracy to 0
patience = 5 # set the number of epochs to wait before stopping if validation accuracy does not improve
wait = 0 # initialize the wait counter to 0

while wait < patience:
    svm_model.fit(X_train_, y_train_)
    y_pred_val = svm_model.predict(X_val_)
    val_accuracy = accuracy_score(y_val_, y_pred_val)
    epoch_time = time.time() - start_time
    print(f'Training time={epoch_time:.2f}s, Validation Accuracy={val_accuracy:.4f}')
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_svm_model = svm_model
        wait = 0 # reset the wait counter if validation accuracy improves
    else:
        wait += 1 # increment the wait counter if validation accuracy does not improve
    
# Evaluate the best model on the test set
y_pred_svm = best_svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Test Accuracy={accuracy_svm:.4f}')


# In[36]:


from sklearn.metrics import roc_curve, auc, confusion_matrix


# In[37]:


# Plot ROC curve
svm_preds = best_svm_model.predict(X_val_)
svm_probs = best_svm_model.predict_proba(X_val_)[:, 1] if hasattr(best_svm_model, 'predict_proba') else svm_preds
fpr, tpr, thresholds = roc_curve(y_val_, svm_probs)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[38]:


# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_svm)
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[39]:


from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_svm)
print('Confusion Matrix:')
print(conf_mat)

# Classification report
class_report = classification_report(y_test, y_pred_svm)
print('Classification Report:')
print(class_report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




