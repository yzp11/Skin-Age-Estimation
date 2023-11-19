from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class MYSVR:
    def __init__(self):
        self.sc_X = StandardScaler()
        self.sc_Y = StandardScaler()
        self.regressor = SVR(kernel='rbf')

    def train(self,X,Y):
        # Feature Scaling
        X = self.sc_X.fit_transform(X)
        Y = self.sc_Y.fit_transform(Y)
        # Fitting the SVR model to the dataset
        Y=Y.flatten()
        self.regressor.fit(X, Y)

    def forward(self,X_in):
        # Predicting a new result with the Polynomial Regression
        Y_Pred = self.sc_Y.inverse_transform(
            self.regressor.predict(self.sc_X.transform(X_in)).reshape(-1, 1))
        return Y_Pred






