import pandas as pd
import pathlib

from zero_deforestation import data


DATA_PATH = pathlib.Path(data.__file__).parent

FORESTNET_FIELD_MAPPER = {
    "Plantation": 0,
    "Grassland shrubland": 1,
    "Smallholder agriculture": 2,
}


def main():
    zero_deforestation_train_df = pd.read_csv(
        DATA_PATH / "/home/agarcia/repos/ZeroDef/zero_deforestation/data/train.csv"
    )

    forestnet_train_df = pd.read_csv(
        DATA_PATH / "ForestNetDataset/deep/downloads/ForestNetDataset/train.csv"
    )
    forestnet_val_df = pd.read_csv(
        DATA_PATH / "ForestNetDataset/deep/downloads/ForestNetDataset/val.csv"
    )
    forestnet_test_df = pd.read_csv(
        DATA_PATH / "ForestNetDataset/deep/downloads/ForestNetDataset/test.csv"
    )
    forestnet_df = pd.concat([forestnet_train_df, forestnet_val_df, forestnet_test_df])

    forestnet_df["example_path"] = forestnet_df.example_path.str.replace(
        "examples", "ForestNetDataset/deep/downloads/ForestNetDataset/examples"
    )
    forestnet_df["example_path"] = (
        forestnet_df["example_path"] + "/images/visible/composite.png"
    )
    forestnet_df = forestnet_df.loc[
        forestnet_df.merged_label.isin(
            ["Plantation", "Grassland shrubland", "Smallholder agriculture"]
        )
    ]
    forestnet_df["label"] = forestnet_df.merged_label.map(FORESTNET_FIELD_MAPPER)

    extended_df = pd.concat(
        [forestnet_df, zero_deforestation_train_df]
    ).drop_duplicates(subset=["label", "latitude", "longitude", "year"])

    extended_df[["label", "latitude", "longitude", "year", "example_path"]].to_csv(
        "/home/agarcia/repos/ZeroDef/zero_deforestation/temp/extended_data.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
