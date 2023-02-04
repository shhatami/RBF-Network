def fit(self):

    self.centroids, self.std_list = kmeans(self.X, self.k, max_iters=1000)

    if not self.std_from_clusters:
        dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
        self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

    RBF_X = self.rbf_list(self.X, self.centroids, self.std_list)

    self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.convert_to_one_hot(self.y, self.number_of_classes)

    RBF_list_tst = self.rbf_list(self.tX, self.centroids, self.std_list)

    self.pred_ty = RBF_list_tst @ self.w

    self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])

    diff = self.pred_ty - self.ty

    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
