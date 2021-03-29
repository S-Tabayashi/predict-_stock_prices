# 訓練期間終了日
TRAIN_END = "2021-03-25"
# 評価期間開始日
VAL_START = "2020-02-01"
# 評価期間終了日
VAL_END = "2021-12-01"
# テスト期間開始日
TEST_START = "2021-01-01"
# 目的変数
TARGET_LABELS = ["label_high_20",
                 "label_low_20",
                 "label_high_5",
                 "label_high_10",
                 "label_low_5",
                 "label_low_10"]

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

FEATURES_HIGH = ['MA_gap_2month_high',
                 'MA_gap_3month_high',
                 'volatility_2month_high',
                 'volatility_3month_high',
                 'Result_Dividend FiscalYear',
                 'return_3month_high',
                 'Forecast_Dividend FiscalYear',
                 'volatility_1month_high',
                 'Forecast_FinancialStatement FiscalYear',
                 'MA_gap_1month_high',
                 'pbr',
                 'Result_FinancialStatement FiscalYear',
                 'return_1month_high',
                 'ema_12',
                 'Result_FinancialStatement TotalAssets',
                 'signal',
                 'Previous_FinancialStatement NetIncome',
                 'per',
                 'Result_FinancialStatement CashFlowsFromOperatingActivities',
                 'Result_FinancialStatement CashFlowsFromInvestingActivities',
                 'ema_10']

FEATURES_LOW = ['MA_gap_2month_low',
                'MA_gap_3month_low',
                'volatility_2month_low',
                'volatility_3month_low',
                'Result_Dividend FiscalYear',
                'return_3month_low',
                'Forecast_Dividend FiscalYear',
                'volatility_1month_low',
                'Forecast_FinancialStatement FiscalYear',
                'MA_gap_1month_low',
                'pbr',
                'Result_FinancialStatement FiscalYear',
                'return_1month_low',
                'ema_12',
                'Result_FinancialStatement TotalAssets',
                'signal',
                'Previous_FinancialStatement NetIncome',
                'per',
                'Result_FinancialStatement CashFlowsFromOperatingActivities',
                'Result_FinancialStatement CashFlowsFromInvestingActivities',
                'ema_10']



