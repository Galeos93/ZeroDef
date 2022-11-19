import pathlib
import sys

import pandas as pd


def main():
    csv_path = sys.argv[1]
    train_proportion = float(sys.argv[2])

    df = pd.read_csv(csv_path)
    dataset_length = len(df)

    train_indexes = set(
        df.sample(int(dataset_length * train_proportion), random_state=1234).index
    )
    val_indexes = set(df.index) - train_indexes

    output_filename_no_ext = str(pathlib.Path(csv_path).with_suffix(""))

    train_df = df.loc[train_indexes]
    val_df = df.loc[val_indexes]

    train_df.to_csv(f"{output_filename_no_ext}_train_split.csv", index=False)
    val_df.to_csv(f"{output_filename_no_ext}_val_split.csv", index=False)


if __name__ == "__main__":
    main()
