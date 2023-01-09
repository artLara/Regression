import numpy as np

class Regressor():
    def __init__(self):
        self.x = None
        self.y = None


    def setValues(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def lineal(self):
        x = self.x
        y = self.y

        N = len(x)
        sumX = np.sum(x)
        sumY = np.sum(y)
        sumXY = np.sum(x*y)
        sumX2 = np.sum(x**2)

        m = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX**2)
        b = (sumX2*sumY - sumX*sumXY) / (N*sumX2 - sumX**2)
        y_pred = [m*i + b for i in x]
        mse = np.mean((y-y_pred)**2)
        return y_pred, mse

    def exp_line(self):
        x = self.x.copy()
        y = self.y.copy()
        N = len(x)
        x_log = np.log(x)
        y_log = np.log(y)
        x = np.log(x)
        y = np.log(y)

        sumX = np.sum(x_log)
        sumY = np.sum(y_log)
        sumXY = np.sum(x_log*y_log)
        sumX2 = np.sum(x_log**2)

        m = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX**2)
        b = (sumX2*sumY - sumX*sumXY) / (N*sumX2 - sumX**2)
        y_pred = [m*i + b for i in x_log]
        mse = np.mean((y-y_pred)**2)
        return y_pred, mse

    def exp_log(self):
        x = self.x.copy()
        y = self.y.copy()
        N = len(x)
        x_log = np.log(x)
        y_log = np.log(y)
        x = np.log(x)
        y = np.log(y)

        sumX = np.sum(x_log)
        sumY = np.sum(y_log)
        sumXY = np.sum(x_log*y_log)
        sumX2 = np.sum(x_log**2)

        m = (N*sumXY - sumX*sumY) / (N*sumX2 - sumX**2)
        b = (sumX2*sumY - sumX*sumXY) / (N*sumX2 - sumX**2)
        A = np.exp(b)
        B = m
        x = self.x.copy()
        x.sort()
        y_pred = [A*(i**B) for i in x]
        return y_pred

    def powerlaw_log(self):
        x = self.x.copy()
        y = self.y.copy()
