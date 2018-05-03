import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import time
import seaborn as sns
from ANN_class import ANN

df = pd.read_csv('./train.csv')
X = np.asmatrix(df.drop('label', axis=1)) / 255
y = np.asarray(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

ann = ANN(2, [64, 64])

ALPHA = 0.1
EPOCH = 20
start_time = time.time()
ann.fit(X_train, y_train, ALPHA, EPOCH)
print("--- %s seconds ---" % (time.time() - start_time))

res = ann.predict(X_test).reshape(1,-1)
y = y_test.reshape(1,-1)
print(f'Test Accuracy: {np.sum(res == y) / len(y_test)}')

# plot some number
num = plt.imshow(X_train[1000,:].reshape(28,28))


# plot the Training cost and Validation cost
epoches = np.arange(EPOCH)
plt.plot(epoches, ann.errors, color='darkorange', label="cost")
plt.xlabel('# epochs')
plt.ylabel('Cost')
legend = plt.legend(loc='best', shadow=True)

# plot the Training accuracy and Validation accuracy
plt.plot(epoches, ann.accuracies, color='darkorange', label="accuracy")
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
legend = plt.legend(loc='best', shadow=True)
