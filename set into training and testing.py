data = np.load('files/mnist_data.npy').astype(float)

train_y = data[0:5000, 0]
train_x = data[0:5000, 1:]

test_y = data[0:1000, 0]
test_x = data[0:1000, 1:]

RBF_CLASSIFIER = RBF(train_x, train_y, test_x, test_y, num_of_classes=10,
                     k=500, std_from_clusters=False)

RBF_CLASSIFIER.fit()
