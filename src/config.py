from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    model_folder: str 
    data_folder: str 
    tensor_log_dir: str 
    log_dir: str 
    version: int
    vectorizer_path: str
    base_data_path: str
    val_data_path: str 
    test_data_path: str = "" 
    vectorizer_file: str = "vectorizer_combined_text.joblib"
    train_data_file: str = "data_train.csv"
    test_data_file: str = "data_test.csv"
    
    def model_path(self, acc):
        return f"{self.model_folder}/model_v-{self.version}_acc-{acc:.3}.joblib"
    
    def feature_names_path(self, acc):
        return f"{self.model_folder}/feature_names_v-{self.version}_acc-{acc:.3}.joblib.gz"


def get_default_config(version = 1, root="./") -> ModelConfig:
    model_folder = Path(f"{root}../output")
    # data_folder = Path("../data_input")
    data_folder = Path(f"{root}../../data_prep/kic_dataprep_ausw/_2data_final")
    output_folder = Path(f"{root}../../data_prep/output")
    val_data_path = str(Path(f"{root}data_output") / "val_data.csv")
    
    cfg = ModelConfig(
        version=version,
        model_folder=str(model_folder),
        test_data_path = Path(f"{root}../../data_prep/kic_dataprep_ausw/_2data_final") / "data_test.csv",
        base_data_path = Path(f"{root}../../../data_prep/kic_dataprep_ausw/_1data_tmp/data.csv"),
        vectorizer_path= Path(output_folder / ModelConfig.vectorizer_file),
        data_folder=str(data_folder),
        val_data_path=val_data_path,
        tensor_log_dir=str(model_folder / "tensor_logs"),
        log_dir=str(model_folder / "logs"),
    )
    cfg.test_data_path = data_folder / cfg.test_data_file

    return cfg
    