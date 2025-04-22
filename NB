from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the Digits dataset
digits = load_digits()

# Step 2: Prepare the data
X = digits.data  # Features (8x8 pixel values flattened into 64 features)
y = digits.target  # Labels (0-9)

# Step 3: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = nb.predict(X_test)

# Step 6: Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Naive Bayes classifier: {accuracy * 100:.2f}%')

# Step 7: Visualize some predictions (optional)
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')  # Reshape back to 8x8 image
    ax.set_title(f'Pred: {y_pred[i]}')
    ax.axis('off')
plt.show()
