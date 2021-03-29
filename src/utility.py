import numpy as np
import os
import pandas as pd
import pickle

# プログレスパーの表示
from tqdm.auto import tqdm

import const

np.random.seed(0)


class Utility(object):
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
        datasets = {}
        for k, v in inputs.items():
            if k != 'stock_fin_price':
                datasets[k] = pd.read_csv(v)
        return datasets


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
                _train_X = feats[: const.TRAIN_END].copy()
                _val_X = feats[const.VAL_START: const.VAL_END].copy()
                _test_X = feats[const.TEST_START:].copy()

                _train_y = labels[: const.TRAIN_END].copy()
                _val_y = labels[const.VAL_START: const.VAL_END].copy()
                _test_y = labels[const.TEST_START:].copy()

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
            total_asset_turnover = \
                row['Result_FinancialStatement NetSales'] / \
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
        fin_data = fin_data[const.SELECT_FIN_DATA_COLUMNS]

        # 特徴量追加
        fin_data = fin_data.join(
            fin_data[['Result_FinancialStatement NetSales',
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
                  'net_assets_growth_rate', 'eps', 'bps', 'roe']] = \
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
        feats = price_data[["EndOfDayQuote ExchangeOfficialClose",
                            "EndOfDayQuote High", "EndOfDayQuote Low"]].copy()
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n):]

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

        # 最高値
        # 最高値の20営業日リターン
        feats["return_1month_high"] = feats["EndOfDayQuote High"].pct_change(20)
        # 最高値の40営業日リターン
        feats["return_2month_high"] = feats["EndOfDayQuote High"].pct_change(40)
        # 最高値の60営業日リターン
        feats["return_3month_high"] = feats["EndOfDayQuote High"].pct_change(60)
        # 最高値の20営業日ボラティリティ
        feats["volatility_1month_high"] = (
            np.log(feats["EndOfDayQuote High"]).diff().rolling(20).std()
        )
        # 最高値の40営業日ボラティリティ
        feats["volatility_2month_high"] = (
            np.log(feats["EndOfDayQuote High"]).diff().rolling(40).std()
        )
        # 最高値の60営業日ボラティリティ
        feats["volatility_3month_high"] = (
            np.log(feats["EndOfDayQuote High"]).diff().rolling(60).std()
        )
        # 終値と20営業日の単純移動平均線の乖離
        feats["MA_gap_1month_high"] = \
            feats["EndOfDayQuote ExchangeOfficialClose"] / (
                feats["EndOfDayQuote High"].rolling(20).mean()
            )
        # 終値と40営業日の単純移動平均線の乖離
        feats["MA_gap_2month_high"] = \
            feats["EndOfDayQuote ExchangeOfficialClose"] / (
                feats["EndOfDayQuote High"].rolling(40).mean()
            )
        # 終値と60営業日の単純移動平均線の乖離
        feats["MA_gap_3month_high"] = \
            feats["EndOfDayQuote ExchangeOfficialClose"] / (
                feats["EndOfDayQuote High"].rolling(60).mean()
            )

        # 最安値
        # 最安値の20営業日リターン
        feats["return_1month_low"] = feats["EndOfDayQuote Low"].pct_change(20)
        # 最安値の40営業日リターン
        feats["return_2month_low"] = feats["EndOfDayQuote Low"].pct_change(40)
        # 最安値の60営業日リターン
        feats["return_3month_low"] = feats["EndOfDayQuote Low"].pct_change(60)
        # 最安値の20営業日ボラティリティ
        feats["volatility_1month_low"] = (
            np.log(feats["EndOfDayQuote Low"]).diff().rolling(20).std()
        )
        # 最安値の40営業日ボラティリティ
        feats["volatility_2month_low"] = (
            np.log(feats["EndOfDayQuote Low"]).diff().rolling(40).std()
        )
        # 最安値の60営業日ボラティリティ
        feats["volatility_3month_low"] = (
            np.log(feats["EndOfDayQuote Low"]).diff().rolling(60).std()
        )
        # 終値と20営業日の単純移動平均線の乖離
        feats["MA_gap_1month_low"] = \
            feats["EndOfDayQuote ExchangeOfficialClose"] / (
                feats["EndOfDayQuote Low"].rolling(20).mean()
            )
        # 終値と40営業日の単純移動平均線の乖離
        feats["MA_gap_2month_low"] = \
            feats["EndOfDayQuote ExchangeOfficialClose"] / (
                feats["EndOfDayQuote Low"].rolling(40).mean()
            )
        # 終値と60営業日の単純移動平均線の乖離
        feats["MA_gap_3month_low"] = \
            feats["EndOfDayQuote ExchangeOfficialClose"] / \
            (feats["EndOfDayQuote Low"].rolling(60).mean()
             )

        # 特徴量追加
        # EWMA
        feats['EWMA'] = feats['EndOfDayQuote ExchangeOfficialClose']

        for t in zip(feats.index, feats.index[1:]):
            feats.loc[t[1], 'EWMA'] = \
                const.ALPHA * feats.loc[
                    t[1], 'EndOfDayQuote ExchangeOfficialClose'] + (
                        1 - const.ALPHA) * feats.loc[t[0], 'EWMA']

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
        feats["Section/Products"] = const.SECTION_PRODUCTS[feats
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
            "selected_columns": const.FEATURES,
            "selected_high_columns": const.FEATURES_HIGH,
            "selected_low_columns": const.FEATURES_LOW,
        }
        return columns[column_group]

    @classmethod
    def create_train_val_from_pkl(cls, label):
        """
        Args:
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        train_X_path = os.path.join(os.path.dirname("__file__"),
                                    "../../high_low_datas/train_X")
        train_y_path = os.path.join(os.path.dirname("__file__"),
                                    "../../high_low_datas/train_y")
        val_X_path = os.path.join(os.path.dirname("__file__"),
                                  "../../high_low_datas/val_X")
        val_y_path = os.path.join(os.path.dirname("__file__"),
                                  "../../high_low_datas/val_y")

        data_X = os.path.join(train_X_path, f"train_X_{label}.pkl")
        with open(data_X, "rb") as f:
            train_X = pickle.load(f)
        data_y = os.path.join(train_y_path, f"train_y_{label}.pkl")
        with open(data_y, "rb") as f:
            train_y = pickle.load(f)
        data_X = os.path.join(val_X_path, f"val_X_{label}.pkl")
        with open(data_X, "rb") as f:
            val_X = pickle.load(f)
        data_y = os.path.join(val_y_path, f"val_y_{label}.pkl")
        with open(data_y, "rb") as f:
            val_y = pickle.load(f)

        return train_X, train_y, val_X, val_y

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
            buff.append(cls.get_features_for_predict(dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, _, _ = cls.get_features_and_label(
            dfs, codes, feature, label
        )

        return train_X, train_y, val_X, val_y

    @classmethod
    def create_train_val_from_feature(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        # 特徴量を取得
        # buff = []
        # for code in codes:
        #     buff.append(cls.get_features_for_predict(dfs, code))
        # feature = pd.concat(buff)
        feature_path = os.path.join(os.path.dirname("__file__"),
                                    "../../feature")
        data_feature = os.path.join(feature_path, f"high_low_feature.pkl")
        with open(data_feature, "rb") as f:
            feature = pickle.load(f)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, _, _ = cls.get_features_and_label(
            dfs, codes, feature, label
        )

        return train_X, train_y, val_X, val_y

    @classmethod
    def calculate_results(cls, results, label) -> np.array:
        """
        Args:
            results (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            Returns:
                    calculated_results (np.ndarray)
    """

        #calculated_result = results['nn']

        calculated_result = (results['extraTree'] +
         #                    results['nn'] +
                             results['lgb'] +
         #                    results['xgb'] +
                             results['cgb']) / 3
        return calculated_result

