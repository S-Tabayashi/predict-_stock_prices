# -*- coding: utf-8 -*-
from sklearn.ensemble import ExtraTreesRegressor
from scipy import stats

# Sequentialのインポート
from keras.models import Sequential
# Dense、LSTMのインポート
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers

class ModelETR(object):
    def __init__(self, feature_columns):
        self.models = None
        self.feature_columns = feature_columns

    def create_model(self, train_X, train_y, label):
        """
        Args:
            train_X (pandas dataframe)
            train_y (pandas dataframe)
        Returns:
            ExtraTreesRegressor
        """
        # モデル作成
        self.model[label] = ExtraTreesRegressor(max_depth=5,
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                min_weight_fraction_leaf=0.1,
                                                n_estimators=700,
                                                random_state=0)

        self.model[label].fit(train_X[self.feature_columns].values, train_y.values)

        return self.model

    def get_model(self):


    def predict(self, label, test_X):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """
        # 予測実施
        df[label] = self.models[label].predict(test_X.values)
            # 出力対象列に追加
        output_columns.append(label)


class ModelNN(object):
    def __init__(self, feature_columns):
        self.model = None
        self.feature_columns = feature_columns

    def create_model(self, train_X, train_y, val_X, val_y):
        """
        Args:
            train_X (pandas dataframe)
            train_y (pandas dataframe)
        Returns:
            ExtraTreesRegressor
        """
        # モデル作成
        train_X = train_X[self.feature_columns]
        train_X = stats.zscore(train_X)
        train_X = train_X.reshape(
            (train_X.shape[0], 1, train_X.shape[1]))
        val_X = val_X[self.feature_columns]
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


    def predict(self, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """
        if label == 'label_high_20':
            feature_columns = cls.get_feature_columns(
                cls.dfs, feats, column_group='selected_columns')
        elif label == 'label_low_20':
            feature_columns = cls.get_feature_columns(
                cls.dfs, feats, column_group='selected_columns')
        else:
            feature_columns = cls.get_feature_columns(
                cls.dfs, feats, column_group='selected_columns')
        test_X = stats.zscore(feats[feature_columns])
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # 予測実施
        df[label] = cls.models[label].predict(test_X)
        # 出力対象列に追加
        output_columns.append(label)



