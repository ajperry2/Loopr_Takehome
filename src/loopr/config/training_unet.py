from pathlib import Path
import getpass


class TrainingUnetConfig():
    data_dir: Path = Path(
        f"/home/{getpass.getuser()}/.cache/kagglehub/datasets/nexuswho/aitex-fabric-image-database/versions/1/")


    kept_classes: list[int] = [10, 19, 30]
    kept: list[str] = [
        "Broken pick",
        "Fuzzyball",
        "Nep"
    ]
    unkept_classes: list[int] = [
        2 , 
        6, 
        16,
        22,
        23,
        25,
        27,
        29,
        36
    ]
    unkept = [
        "Broken end",
        "Broken yarn",
        "Weft curling",
        "Cut selvage",
        "Crease",
        "Warp ball",
        "Knots",
        "Contamination",
        "Weft crack"
    ]
    class_label_to_name: dict[int, str] = {
        2: 	"Broken end",
        6: 	"Broken yarn",
        10 :	"Broken pick",
        16 :	"Weft curling",
        19: 	"Fuzzyball",
        22: 	"Cut selvage",
        23: 	"Crease",
        25: 	"Warp ball",
        27: 	"Knots",
        29 :	"Contamination",
        30: 	"Nep",
        36: 	"Weft crack"
    }
    epochs: int = 10
    num_workers: int = 1
    height: int = 256
    width: int = 256
    batch_size: int  = 8
    pretrained_path: str = Path("best_model.pth")
    pretrained_mlp_path: str = Path("best_mlp.pth")
    pretrained_contrastive_path: str = Path("best_contrastive_layer.pth")
    lr: float = 3e-4
    weight_decay: float = 1e-4
    train_split: float = 0.8
    threshold: float = 0.5
    device: str = "cuda"
    early_stopping_patience: int = 5
    distance_metric = "l2_norm"
    