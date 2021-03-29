# -*- coding: utf-8 -*-
import io
import os

import numpy as np
import pickle

import const
from models import ModelETR as cls_etr
from models import ModelLGB as cls_lgb
from models import ModelXGB as cls_xgb
from models import ModelCGB as cls_cgb
from models import ModelNN as cls_nn
from utility import Utility as utility

np.random.seed(0)

class ScoringService(object):

    models = {
        "extraTree": cls_etr,
        "lgb": cls_lgb,
        "xgb": cls_xgb,
        "cgb": cls_cgb,
        "nn": cls_nn
    }

    # データをこの変数に読み込む
    dfs = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = utility.get_inputs(dataset_dir)
        return inputs

    @classmethod
    def save_model(cls, model, label, model_name, model_path="../model"):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_name (str): prediction model name
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        if model_name == 'nn':
            os.makedirs(model_path, exist_ok=True)
            model.save(os.path.join(model_path, f"my_{model_name}_{label}"))
        else:
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, f"my_{model_name}_{label}.pkl"),
                      "wb") as f:
                # モデルをpickle形式で保存
                pickle.dump(model, f)

    @classmethod
    def get_model(cls, labels=None):
        """Get model method

        Args:
            labels (arrayt): list of prediction target labels
        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if labels is None:
            labels = const.TARGET_LABELS

        try:
            for label in labels:
                for model in cls.models.keys():
                    print(model)
                    cls.models[model].get_model(label=label)
            return True
        except Exception as e:
            print(e)
            return False

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path="../model"
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.dfs = utility.get_dataset(inputs)
            cls.codes = utility.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = const.TARGET_LABELS
        for label in labels:
            print(label)
            train_X, train_y, val_X, val_y = \
                utility.create_train_val_from_feature(cls.dfs, codes, label)

            for model in cls.models.keys():
                print(model)
                if model == 'nn':
                    if label == "label_high_20":
                        feature_columns = utility.get_feature_columns(
                            cls.dfs, train_X,
                            column_group='selected_high_columns')
                    else:
                        feature_columns = utility.get_feature_columns(
                            cls.dfs, train_X,
                            column_group='selected_low_columns')

                    created_model = cls.models[model].create_model(
                        train_X[feature_columns], train_y,
                        val_X[feature_columns],val_y, label)
                    cls.save_model(created_model, label, model,
                                   model_path=model_path)

                else:
                    if label == "label_high_20":
                        feature_columns = utility.get_feature_columns(
                            cls.dfs, train_X,
                            column_group='selected_high_columns')
                    else:
                        feature_columns = utility.get_feature_columns(
                            cls.dfs, train_X,
                            column_group='selected_low_columns')
                    created_model = cls.models[model].create_model(
                        train_X[feature_columns], train_y, label)
                    cls.save_model(created_model, label, model,
                                   model_path=model_path)

    @classmethod
    def predict(cls, inputs, model_names=None,
                labels=None, codes=None, start_dt=const.TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            model_names (list[str]): predict models
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """
        result_models = {}

        # データ読み込み
        if cls.dfs is None:
            cls.dfs = utility.get_dataset(inputs)
            utility.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if labels is None:
            labels = const.TARGET_LABELS

        # 特徴量を作成
        #buff = []
        #for code in codes:
        #    buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        #feats = pd.concat(buff)
        feats_path = os.path.join(os.path.dirname("__file__"),
                                  "../../feature/")
        data_feats = os.path.join(feats_path, "high_low_feature.pkl")
        with open(data_feats, "rb") as f:
            feats = pickle.load(f)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + \
                            df.loc[:, "code"].astype(str)

        # 出力対象列を定義
        output_columns = ["code"]

        # 目的変数毎に予測
        for label in labels:
            if label == "label_high_20":
                feature_columns = utility.get_feature_columns(
                    cls.dfs, feats, column_group='selected_high_columns')
            else:
                feature_columns = utility.get_feature_columns(
                    cls.dfs, feats, column_group='selected_low_columns')
            test_X = feats[feature_columns].values

            for model in cls.models.keys():
                print(model)
                if model == 'nn':
                    cls_nn.get_model(label=label)
                    nn_result = cls_nn.predict(label, test_X)
                    result = nn_result.reshape(nn_result.shape[0])
                else:
                    cls.models[model].get_model(label)
                    result = cls.models[model].predict(label, test_X)

                result_models[model] = result

            # モデル毎の予測値の平均を求める
            result_model = utility.calculate_results(result_models, label)
            df[label] = result_model

            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()

