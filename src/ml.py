import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class RegressorFramework:
    def __init__(self, model, cv_folds=3):
        """
        初始化模型与参数
        :param model: sklearn回归模型对象
        :param cv_folds: 交叉验证的折数
        """
        self.model = model
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        拟合模型
        :param X: 输入特征（未归一化）
        :param y: 标签
        """
        # 归一化
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def cross_validate(self, X, y):
        """
        使用交叉验证评估模型
        :return: R2 和 RMSE 的平均值
        """
        X_scaled = self.scaler.fit_transform(X)
        r2_scores = cross_val_score(self.model, X_scaled, y, cv=self.cv_folds, scoring='r2')
        neg_mse_scores = cross_val_score(self.model, X_scaled, y, cv=self.cv_folds, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-neg_mse_scores)
        return r2_scores.mean(), rmse_scores.mean()

    def predict(self, X_test):
        """
        对单独的测试集进行预测
        :param X_test: 测试数据（未归一化）
        :return: 预测值
        """
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """
        在测试集上评估模型
        :param X_test: 特征
        :param y_test: 标签
        :return: R2 和 RMSE
        """
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse, y_pred
