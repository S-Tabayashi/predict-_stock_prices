from predictor import ScoringService as cls

# データセットをダウンロードして解凍したファイルを配置した場所を定義します。
# データ保存先ディレクトリ
DATASET_DIR= "../../data_dir"

# 読み込むファイルを定義します。
inputs = cls.get_inputs(DATASET_DIR)

print(inputs)
cls.train_and_save_model(inputs, model_path="model")