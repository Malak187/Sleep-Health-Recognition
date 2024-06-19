import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import linear_sum_assignment

# Load your dataset, assuming 'data' is your DataFrame
data = pd.read_csv('processingData\processed_dataset.csv')

# Separate features (X) and target variable (y)
x = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

lda = LinearDiscriminantAnalysis(n_components=2)  
# Specify the number of components (optional)

X_lda = lda.fit_transform(x, y)


# Split the data into training and testing sets after LDA transformation
X_train_lda, X_test_lda, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)



# Colors for classes, assuming there are three classes (for example)
colors = ['red', 'green', 'blue']
target_names = ['Class 0', 'Class 1', 'Class 2']

plt.figure(1)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], alpha=0.8, color=color, label=target_name)

plt.title('Scatter plot of the LDA-transformed data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best', shadow=False, scatterpoints=1)






# Train SVM on the LDA-transformed data
print("Performance Evaluation for SVM Algorithm: ")
print()
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train_lda, y_train)

# Predictions
y_pred = svm_clf.predict(X_test_lda)

# Evaluation
print()
print("First, Classification Report:")
print()
print(classification_report(y_test, y_pred))
print()
print("Second, Confusion Matrix:")
print()
print(confusion_matrix(y_test, y_pred))



plt.figure(figsize=(10, 12))  # Adjust the size as needed

plt.subplot(1, 2, 1) 

# Create meshgrid
x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict on the mesh to get the decision boundaries
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plot decision boundary
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
scatter = plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='o', s=50)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('SVM on LDA-transformed Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Add a legend
plt.legend(*scatter.legend_elements(), title="Classes")









# Apply K-means clustering on the LDA-transformed data
kmeans = KMeans(n_clusters=3,random_state=23)
kmeans.fit(X_lda)
clusters = kmeans.labels_

print()
print("Performance Evaluation for K-mean Algorithm: ")
print()
ari = adjusted_rand_score(y, clusters)
print(f"Accuracy of k-mean algorithm: {ari:.4f}")
print()




initial_cm = confusion_matrix(y, clusters)
row_ind, col_ind = linear_sum_assignment(-initial_cm)

mapped_labels = np.zeros_like(clusters)
for i in range(len(col_ind)):
    mapped_labels[clusters == row_ind[i]] = col_ind[i]

new_cm = confusion_matrix(y, mapped_labels)
print("Re-mapped Confusion Matrix:")
print(new_cm)


plt.subplot(1, 2, 2)
#Visualize the clusters
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=clusters, cmap = cmap_bold , marker='o')
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering on LDA-transformed Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()