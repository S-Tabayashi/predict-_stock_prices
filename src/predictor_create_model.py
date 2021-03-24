# -*- coding: utf-8 -*-
import io
import os
import pickle

from models import ModelETR, ModelNN
import numpy as np
import pandas as pd
from scipy import stats

# プログレスパーの表示
from tqdm.auto import tqdm


class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    VAL_START = "2019-02-01"
    # 評価期間終了日
    VAL_END = "2019-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    ALPHA = 0.25

    SELECT_FIN_DATA_COLUMNS = ['Result_FinancialStatement FiscalYear',
                               'Result_FinancialStatement NetSales',
                               'Result_FinancialStatement OperatingIncome',
                               'Result_FinancialStatement OrdinaryIncome',
                               'Result_FinancialStatement NetIncome',
                               'Result_FinancialStatement TotalAssets',
                               'Result_FinancialStatement NetAssets',
                               'Result_FinancialStatement '
                               'CashFlowsFromOperatingActivities',
                               'Result_FinancialStatement '
                               'CashFlowsFromFinancingActivities',
                               'Result_FinancialStatement '
                               'CashFlowsFromInvestingActivities',
                               'Forecast_FinancialStatement FiscalYear',
                               'Forecast_FinancialStatement NetSales',
                               'Forecast_FinancialStatement OperatingIncome',
                               'Forecast_FinancialStatement OrdinaryIncome',
                               'Forecast_FinancialStatement NetIncome',
                               'Result_Dividend FiscalYear',
                               'Result_Dividend QuarterlyDividendPerShare',
                               'Result_Dividend AnnualDividendPerShare',
                               'Forecast_Dividend FiscalYear',
                               'Forecast_Dividend QuarterlyDividendPerShare',
                               'Forecast_Dividend AnnualDividendPerShare',
                               'IssuedShareEquityQuote IssuedShare',
                               'Section/Products', '33 Sector(Code)',
                               '17 Sector(Code)']

    SECTION_PRODUCTS = {
        "First Section (Domestic)": 1,
        "JASDAQ(Standard / Domestic)": 2,
        "Second Section(Domestic)": 3,
        "Mothers (Domestic)": 4,
        "JASDAQ(Growth/Domestic)": 5
    }

    FEATURES = ['MA_gap_2month',
                'MA_gap_3month',
                'volatility_2month',
                'volatility_3month',
                'Result_Dividend FiscalYear',
                'return_3month',
                'Forecast_Dividend FiscalYear',
                'volatility_1month',
                'Forecast_FinancialStatement FiscalYear',
                'MA_gap_1month',
                'pbr',
                'Result_FinancialStatement FiscalYear',
                'return_1month',
                'ema_12',
                'Result_FinancialStatement TotalAssets',
                'signal',
                'Previous_FinancialStatement NetIncome',
                'per',
                'Result_FinancialStatement CashFlowsFromOperatingActivities',
                'Result_FinancialStatement CashFlowsFromInvestingActivities',
                'ema_10']

    MODEL_NAMES = ['ExtraTreesRegressor', 'NeuralNet']


    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
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
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            if k != 'stock_fin_price':
                cls.dfs[k] = pd.read_csv(v)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"].copy()
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]
            # 日付列をpd.Timestamp型に変換してindexに設定
            stock_labels["datetime"] = pd.to_datetime(stock_labels["base_date"])
            stock_labels.set_index("datetime", inplace=True)

            # 特定の目的変数に絞る
            labels = stock_labels[label]
            # nanを削除
            labels.dropna(inplace=True)

            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END].copy()
                _val_X = feats[cls.VAL_START : cls.VAL_END].copy()
                _test_X = feats[cls.TEST_START :].copy()

                _train_y = labels[: cls.TRAIN_END].copy()
                _val_y = labels[cls.VAL_START : cls.VAL_END].copy()
                _test_y = labels[cls.TEST_START :].copy()

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y

    @classmethod
    def calculate_glossary_of_financial_analysis(cls, row):
        operating_profit_margin = 0
        ordinary_profit_margin = 0
        net_profit_margin = 0
        total_asset_turnover = 0
        net_sales_growth_rate = 0
        ordinary_income_growth_rate = 0
        operationg_income_growth_rate = 0
        total_assets_growth_rate = 0
        net_assets_growth_rate = 0
        eps = 0
        bps = 0
        roe = 0

        # 売上高営業利益率 売上高営業利益率（％）＝営業利益÷売上高×100
        if row['Result_FinancialStatement NetSales'] != 0:
            operating_profit_margin = \
                row['Result_FinancialStatement OperatingIncome'] / \
                row['Result_FinancialStatement NetSales'] * 100
        # 売上高経常利益率　売上高経常利益率（％）＝経常利益÷売上高×100
        if row['Result_FinancialStatement NetSales'] != 0:
            ordinary_profit_margin = \
                row['Result_FinancialStatement OrdinaryIncome'] / \
                row['Result_FinancialStatement NetSales'] * 100
        # 売上高純履歴率　売上高純利益率（％）＝当期純利益÷売上高×100
        if row['Result_FinancialStatement NetSales'] != 0:
            net_profit_margin = row['Result_FinancialStatement NetIncome'] / \
                                row['Result_FinancialStatement NetSales'] * 100
        # 総資本回転率 総資本回転率（％）＝売上高÷総資本（自己資本＋他人資本）×100
        if row['Result_FinancialStatement NetAssets'] != 0:
            total_asset_turnover = row['Result_FinancialStatement NetSales'] / \
                                row['Result_FinancialStatement NetAssets'] * 100
        # 売上高増加率
        if row['Previous_FinancialStatement NetSales'] != 0:
            net_sales_growth_rate = \
                (row['Result_FinancialStatement NetSales'] -
                row['Previous_FinancialStatement NetSales']) / \
                row['Previous_FinancialStatement NetSales'] * 100
        # 経常利益増加率
        if row['Previous_FinancialStatement OrdinaryIncome'] != 0:
            ordinary_income_growth_rate = \
                (row['Result_FinancialStatement OrdinaryIncome'] -
                row['Previous_FinancialStatement OrdinaryIncome']) / \
                row['Previous_FinancialStatement OrdinaryIncome'] * 100

        # 営業利益増加率
        if row['Previous_FinancialStatement OperatingIncome'] != 0:
            operationg_income_growth_rate = \
                (row['Result_FinancialStatement OperatingIncome'] -
                row['Previous_FinancialStatement OperatingIncome']) / \
                row['Previous_FinancialStatement OperatingIncome'] * 100
        # 総資本増加率
        if row['Previous_FinancialStatement TotalAssets'] != 0:
            total_assets_growth_rate = \
                (row['Result_FinancialStatement TotalAssets'] -
                row['Previous_FinancialStatement TotalAssets']) / \
                row['Previous_FinancialStatement TotalAssets'] * 100
        # 純資本増加率
        if row['Previous_FinancialStatement NetAssets'] != 0:
            net_assets_growth_rate = \
                (row['Result_FinancialStatement NetAssets'] -
                row['Previous_FinancialStatement NetAssets']) / \
                row['Previous_FinancialStatement NetAssets'] * 100
        # 一株当たり当期純利益（EPS）
        if row['IssuedShareEquityQuote IssuedShare'] != 0:
            eps = row['Result_FinancialStatement NetIncome'] / \
                  row['IssuedShareEquityQuote IssuedShare']
            # BPS 一株当たり純資産（円） ＝ 純資産 ÷ 発行済株式総数
            bps = row['Result_FinancialStatement NetAssets'] / \
                  row['IssuedShareEquityQuote IssuedShare']
            # ROE EPS（一株当たり利益）÷ BPS（一株当たり純資産）× 100
            if bps > 0:
                roe = eps / bps * 100
        return pd.Series(
            [operating_profit_margin, ordinary_profit_margin,
             net_profit_margin, total_asset_turnover,
             net_sales_growth_rate, ordinary_income_growth_rate,
             operationg_income_growth_rate, total_assets_growth_rate,
             net_assets_growth_rate, eps, bps, roe])

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_finデータを読み込み
        stock_fin = dfs["stock_fin"].copy()

        stock_list = dfs["stock_list"].copy()
        stock_fin = pd.merge(stock_fin, stock_list, on=["Local Code"])

        # 特定の銘柄コードのデータに絞る
        fin_data = stock_fin[stock_fin["Local Code"] == code].copy()
        # 日付列をpd.Timestamp型に変換してindexに設定
        fin_data["datetime"] = pd.to_datetime(fin_data["base_date"])
        fin_data.set_index("datetime", inplace=True)
        # fin_dataの特定のカラムを取得
        fin_data = fin_data[cls.SELECT_FIN_DATA_COLUMNS]

        # 特徴量追加
        fin_data = fin_data.join(fin_data
        [['Result_FinancialStatement NetSales',
          'Result_FinancialStatement OperatingIncome',
          'Result_FinancialStatement OrdinaryIncome',
          'Result_FinancialStatement NetIncome',
          'Result_FinancialStatement TotalAssets',
          'Result_FinancialStatement NetAssets',
          'Result_FinancialStatement CashFlowsFromOperatingActivities',
          'Result_FinancialStatement CashFlowsFromFinancingActivities',
          'Result_FinancialStatement CashFlowsFromInvestingActivities']].rename(
            columns=
            {
                'Result_FinancialStatement NetSales':
                    'Previous_FinancialStatement NetSales',
                'Result_FinancialStatement OperatingIncome':
                    'Previous_FinancialStatement OperatingIncome',
                'Result_FinancialStatement OrdinaryIncome':
                    'Previous_FinancialStatement OrdinaryIncome',
                'Result_FinancialStatement NetIncome':
                    'Previous_FinancialStatement NetIncome',
                'Result_FinancialStatement TotalAssets':
                    'Previous_FinancialStatement TotalAssets',
                'Result_FinancialStatement NetAssets':
                    'Previous_FinancialStatement NetAssets',
                'Result_FinancialStatement CashFlowsFromOperatingActivities':
                    'Previous_FinancialStatement '
                    'CashFlowsFromOperatingActivities',
                'Result_FinancialStatement CashFlowsFromFinancingActivities':
                    'Previous_FinancialStatement '
                    'CashFlowsFromFinancingActivities',
                'Result_FinancialStatement CashFlowsFromInvestingActivities':
                    'Previous_FinancialStatement '
                    'CashFlowsFromInvestingActivities'}).shift(-1))
        fin_data[['operating_profit_margin', 'ordinary_profit_margin',
                  'net_profit_margin', 'total_asset_turnover',
                  'net_sales_growth_rate', 'ordinary_income_growth_rate',
                  'operationg_income_growth_rate',
                  'total_assets_growth_rate',
                  'net_assets_growth_rate','eps', 'bps', 'roe']] = \
            fin_data.apply(cls.calculate_glossary_of_financial_analysis, axis=1)

        # 欠損値処理
        fin_feats = fin_data.fillna(0)

        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        fin_feats = fin_feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n):]

        # stock_priceデータを読み込む
        price = dfs["stock_price"].copy()
        # 特定の銘柄コードのデータに絞る
        price_data = price[price["Local Code"] == code].copy()
        # 日付列をpd.Timestamp型に変換してindexに設定
        price_data["datetime"] = \
            pd.to_datetime(price_data["EndOfDayQuote Date"])
        price_data.set_index("datetime", inplace=True)
        # 終値のみに絞る
        feats = price_data[["EndOfDayQuote ExchangeOfficialClose"]].copy()
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]

        # 終値の20営業日リターン
        feats["return_1month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(20)
        # 終値の40営業日リターン
        feats["return_2month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(40)
        # 終値の60営業日リターン
        feats["return_3month"] = feats[
            "EndOfDayQuote ExchangeOfficialClose"
        ].pct_change(60)
        # 終値の20営業日ボラティリティ
        feats["volatility_1month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(20)
            .std()
        )
        # 終値の40営業日ボラティリティ
        feats["volatility_2month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(40)
            .std()
        )
        # 終値の60営業日ボラティリティ
        feats["volatility_3month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"])
            .diff()
            .rolling(60)
            .std()
        )
        # 終値と20営業日の単純移動平均線の乖離
        feats["MA_gap_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"] \
                                 / (feats["EndOfDayQuote ExchangeOfficialClose"]
                                    .rolling(20).mean())
        # 終値と40営業日の単純移動平均線の乖離
        feats["MA_gap_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"] \
                                 / (feats["EndOfDayQuote ExchangeOfficialClose"]
                                    .rolling(40).mean())
        # 終値と60営業日の単純移動平均線の乖離
        feats["MA_gap_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"] \
                                 / (feats["EndOfDayQuote ExchangeOfficialClose"]
                                    .rolling(60).mean())

        # 特徴量追加
        # EWMA
        feats['EWMA'] = feats['EndOfDayQuote ExchangeOfficialClose']

        for t in zip(feats.index, feats.index[1:]):
            feats.loc[t[1], 'EWMA'] = cls.ALPHA * feats.loc[
                t[1], 'EndOfDayQuote ExchangeOfficialClose'] + (1 - cls.ALPHA) \
                                      * feats.loc[t[0], 'EWMA']

        # EMA 10日
        feats["ema_10"] = feats["EndOfDayQuote ExchangeOfficialClose"].ewm(
            span=10).mean()

        # MACD
        # EMA12
        feats["ema_12"] = feats["EndOfDayQuote ExchangeOfficialClose"].ewm(
            span=12).mean()
        # EMA 26
        feats["ema_26"] = feats["EndOfDayQuote ExchangeOfficialClose"].ewm(
            span=26).mean()
        feats["macd"] = feats["ema_12"] - feats["ema_26"]
        feats["signal"] = feats["macd"].ewm(span=9).mean()

        # PBR 株価 ÷ BPS（1株あたり純資産）
        feats["pbr"] = feats["EndOfDayQuote ExchangeOfficialClose"] \
                       / fin_data["bps"]
        # PER 株価 ÷ 1株当たり利益（EPS）
        feats["per"] = feats["EndOfDayQuote ExchangeOfficialClose"] \
                       / fin_data["eps"]

        # 欠損値処理
        feats = feats.fillna(0)
        # 元データのカラムを削除
        feats = feats.drop(["EndOfDayQuote ExchangeOfficialClose"], axis=1)

        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        feats = feats.loc[feats.index.isin(fin_feats.index)]
        fin_feats = fin_feats.loc[fin_feats.index.isin(feats.index)]

        # データを結合
        feats = pd.concat([feats, fin_feats], axis=1).dropna()

        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], 0)

        # 市場・商品区分を数値に変換
        feats["Section/Products"] = cls.SECTION_PRODUCTS[feats
        ["Section/Products"][0]]

        # 銘柄コードを設定
        feats["code"] = code

        # 生成対象日以降の特徴量に絞る
        feats = feats.loc[pd.Timestamp(start_dt):]

        return feats

    @classmethod
    def get_feature_columns(cls, dfs, train_X,
                            column_group="fundamental+technical"):
        # 特徴量グループを定義
        # ファンダメンタル
        fundamental_cols = dfs["stock_fin"].select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[
            fundamental_cols != "Result_Dividend DividendPayableDate"
            ]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"]
        # 価格変化率
        returns_cols = [x for x in train_X.columns if "return" in x]
        # テクニカル
        technical_cols = [
            x for x in train_X.columns if
            (x not in fundamental_cols) and (x != "code")
        ]
        columns = {
            "fundamental_only": fundamental_cols,
            "return_only": returns_cols,
            "technical_only": technical_cols,
            "fundamental+technical": list(fundamental_cols) + list(
                technical_cols),
            "selected_columns": cls.FEATURES,
        }
        return columns[column_group]

    @classmethod
    def create_train_val(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        # 特徴量を取得
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, _, _ = cls.get_features_and_label(
            dfs, codes, feature, label
        )

        return train_X, train_y, val_X, val_y

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
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_{model_name}_{label}.pkl"),
                  "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_model(cls, model_path="../model", labels=None, model_names=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels
            model_names (arrayt): list of prediction models

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        if model_names is None:
            model_names  = cls.MODEL_NAMES
        try:
            for label in labels:
                for model_name in model_names:
                    m = os.path.join(model_path, f"my_{model_name}_{label}.pkl")
                    with open(m, "rb") as f:
                        # pickle形式で保存されているモデルを読み込み
                        cls.models[label][model_name] = pickle.load(f)
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
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            print(label)
            train_X, train_y, val_X, val_y = cls.create_train_val(
                cls.dfs, codes, label)

            for model_name in cls.MODEL_NAMES:
                print(model)
                if model_name == 'ExtraTreesRegressor':
                    feature_columns = cls.get_feature_columns(
                        cls.dfs, train_X, column_group='selected_columns')

                    etr_model = ModelETR(feature_columns)
                    model = etr_model.create_model(train_X,train_y)
                    cls.save_model(model, label, model_name,
                                   model_path=model_path)

                else:
                    nn_model = ModelNN(cls.FEATURES)
                    nn_model.create_model(train_X, train_y, val_X, val_y)
                    cls.save_model(model, label, model_name,
                                   model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """
        etr_model = ModelETR()

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        buff = []
        for code in codes:
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats = pd.concat(buff)

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
            feature_columns = cls.get_feature_columns(
                cls.dfs, feats, column_group='selected_columns')

            for model in models:


                # 予測実施
                df[label] = etr_model.predict(feats[feature_columns].values)
                # 出力対象列に追加

            # モデル毎の予測値の平均を求める


            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()
