# -*- coding: utf-8 -*-]
import lightgbm as lgb
import os
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from scipy import stats
import xgboost as xgb
import catboost as cgb

# Sequentialのインポート
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
import tensorflow as tf
# Dense、LSTMのインポート
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau
from keras import optimizers

tf.random.set_seed(0)


class ModelETR(object):
    # モデルをこの変数に読み込む
    models = {}

    @ classmethod
    def create_model(cls, train_X, train_y, label):
        """
        Args:
            train_X (pandas dataframe)
            train_y (pandas dataframe)
            label (list[str]): target label name
        Returns:
            ExtraTreesRegressor
        """
        # モデル作成
        if label == "label_high_20":
            cls.models[label] = ExtraTreesRegressor(
                max_depth=7, min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.1, n_estimators=700, random_state=0)
        else:
            cls.models[label] = ExtraTreesRegressor(
                max_depth=7, min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.1, n_estimators=700, random_state=0)

        cls.models[label].fit(train_X.values, train_y.values)

        return cls.models[label]

    @classmethod
    def get_model(cls, label, model_path="../model"):
        m = os.path.join(model_path, f"my_extraTree_{label}.pkl")
        with open(m, "rb") as f:
            # pickle形式で保存されているモデルを読み込み
            cls.models[label] = pickle.load(f)

    @classmethod
    def predict(cls, label, test_X):
        """Predict method

        Args:
            label (str): target label name
            test_X (df): test data
        Returns:
            ndarray: Results of Predict.
        """
        print("model et predict")
        # 予測実施
        predict = cls.models[label].predict(test_X)
        # 出力対象列に追加
        return predict


class ModelLGB(object):
    # モデルをこの変数に読み込む
    models = {}

    @ classmethod
    def create_model(cls, train_X, train_y, label):
        """
        Args:
            train_X (pandas dataframe)
            train_y (pandas dataframe)
            label (list[str]): target label name
        Returns:
            ExtraTreesRegressor
        """
        # モデル作成
        if label == "label_high_20":
            cls.models[label] = lgb.LGBMRegressor(
                max_depth=5,learning_rate=0.05, num_leaves=100,
                n_estimators=100, random_state=0)
        else:
            cls.models[label] = lgb.LGBMRegressor(
                max_depth=7, learning_rate=0.05, num_leaves=100,
                n_estimators=100, random_state=0)

        cls.models[label].fit(train_X.values, train_y.values)

        return cls.models[label]

    @classmethod
    def get_model(cls, label, model_path="../model"):

        m = os.path.join(model_path, f"my_lgb_{label}.pkl")
        with open(m, "rb") as f:
            # pickle形式で保存されているモデルを読み込み
            cls.models[label] = pickle.load(f)

    @classmethod
    def predict(cls, label, test_X):
        """Predict method

        Args:
            label (str): target label name
            test_X (df): test data
        Returns:
            ndarray: Results of Predict.
        """
        print("lgb model predict")
        # 予測実施
        predict = cls.models[label].predict(test_X)
        # 出力対象列に追加
        return predict


class ModelXGB(object):
    # モデルをこの変数に読み込む
    models = {}

    @ classmethod
    def create_model(cls, train_X, train_y, label):
        """
        Args:
            train_X (pandas dataframe)
            train_y (pandas dataframe)
            label (list[str]): target label name
        Returns:
            ExtraTreesRegressor
        """
        # モデル作成
        if label == "label_high_20":
            cls.models[label] = xgb.XGBRegressor(
                alpha=0, colsample_bytree=0.5, gamma=1, learning_rate=0.01,
                max_depth=7, min_child_weight=3, n_estimators=700, eta=0.1,
                objective="reg:pseudohubererror", random_state=0, subsample=1)
        else:
            cls.models[label] = xgb.XGBRegressor(
                alpha=0, colsample_bytree=0.5, gamma=0.1, learning_rate=0.01,
                max_depth=7, min_child_weight=3, n_estimators=700, eta=0.1,
                objective="reg:pseudohubererror", random_state=0, subsample=1)

        cls.models[label].fit(train_X.values, train_y.values)

        return cls.models[label]

    @classmethod
    def get_model(cls, label, model_path="../model"):

        m = os.path.join(model_path, f"my_xgb_{label}.pkl")
        with open(m, "rb") as f:
            # pickle形式で保存されているモデルを読み込み
            cls.models[label] = pickle.load(f)

    @classmethod
    def predict(cls, label, test_X):
        """Predict method

        Args:
            label (str): target label name
            test_X (df): test data
        Returns:
            ndarray: Results of Predict.
        """
        # 予測実施
        predict = cls.models[label].predict(test_X)
        # 出力対象列に追加
        return predict


class ModelCGB(object):
    # モデルをこの変数に読み込む
    models = {}

    @ classmethod
    def create_model(cls, train_X, train_y, label):
        """
        Args:
            train_X (pandas dataframe)
            train_y (pandas dataframe)
            label (list[str]): target label name
        Returns:
            ExtraTreesRegressor
        """
        # モデル作成
        if label == "label_high_20":
            cls.models[label] = cgb.CatBoostRegressor(
                bagging_temperature=0.01, depth=5, iterations=100,
                learning_rate=0.18831273426065617, od_type='Iter', od_wait=5,
                random_strength=33, random_seed=0)
        else:
            cls.models[label] = cgb.CatBoostRegressor(
                bagging_temperature=0.01, depth=5, iterations=222,
                learning_rate=0.2, od_type='Iter', od_wait=5,
                random_strength=33, random_seed=0)

        cls.models[label].fit(train_X.values, train_y.values)

        return cls.models[label]

    @classmethod
    def get_model(cls, label, model_path="../model"):

        m = os.path.join(model_path, f"my_cgb_{label}.pkl")
        with open(m, "rb") as f:
            # pickle形式で保存されているモデルを読み込み
            cls.models[label] = pickle.load(f)

    @classmethod
    def predict(cls, label, test_X):
        """Predict method

        Args:
            label (str): target label name
            test_X (df): test data
        Returns:
            ndarray: Results of Predict.
        """
        # 予測実施
        predict = cls.models[label].predict(test_X)
        # 出力対象列に追加
        return predict


class ModelNN(object):
    # モデルをこの変数に読み込む
    models = {}

    @classmethod
    def create_model(cls, train_X, train_y, val_X, val_y, label):
        """
        Args:
            train_X (df): training data
            train_y (df): training variable
            val_X (df): validation data
            val_y (df): validation variable
            label (list[str]): target label name

        Returns:
            Neural Net model
        """
        # モデル作成
        train_X = train_X
        train_X = stats.zscore(train_X)
        train_X = train_X.reshape(
            (train_X.shape[0], 1, train_X.shape[1]))
        val_X = val_X
        val_X = stats.zscore(val_X)
        val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

        model = Sequential()
        model.add(LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(BatchNormalization())
        model.add(Dropout(.2))

        model.add(Dense(256))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(.1))

        model.add(Dense(256))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(.1))

        model.add(Dense(128))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(.05))

        model.add(Dense(64))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(.05))

        model.add(Dense(32))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(.05))

        model.add(Dense(16))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(.05))

        model.add(Dense(1))

        # ネットワークのコンパイル
        model.compile(loss='mse', optimizer=optimizers.Adam(0.001),
                      metrics=['mse'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                              verbose=1, epsilon=1e-4, mode='min')
        ]

        model.fit(x=train_X, y=train_y, epochs=80,
                  validation_data=(val_X, val_y), callbacks=[callbacks])
        cls.models[label] = model

        return model

    @classmethod
    def get_model(cls, label, model_path="../model"):
        m = os.path.join(model_path, f"my_nn_{label}")
        cls.models[label] = tf.keras.models.load_model(m)

    @classmethod
    def predict(cls, label, test_X):
        """Predict method

        Args:
            label (list[str]): target label name
            test_X (dict[str]): paths to the dataset files
        Returns:
            str: Inference for the given input.
        """
        test_X = stats.zscore(test_X)
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # 予測実施
        predict = cls.models[label].predict(test_X)
        # 出力対象列に追加
        return predict


