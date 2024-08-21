import os
import shutil
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from time import sleep
from progress.bar import Bar


def split_csv_by_sensor(input_file: str) -> None:
    """
    Splits a CSV file by sensor ID and sensor type.
    The output files are saved in a 'processed' directory relative to the input file's location.

    Parameters:
    input_file (str): The path to the input CSV file to be processed.

    Returns:
    None
    """
    processed_dir = "../../data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(input_file, skiprows=40, delimiter=";")

    df = df.iloc[:, :-1]

    base_filename = os.path.basename(input_file).replace(".csv", "")
    base_filename = "_".join(base_filename.split("_")[:6])

    parts = base_filename.split("_")
    parts[-1] = parts[-1].zfill(2)

    base_filename = "_".join(parts)

    for sensor_id in df[" Sensor ID"].unique():
        for sensor_type in df[" Sensor Type"].unique():
            df_filtered = df[
                (df[" Sensor ID"] == sensor_id) & (df[" Sensor Type"] == sensor_type)
            ]

            if df_filtered.empty:
                continue

            df_filtered = downsampling_and_padding(df_filtered)
            df_filtered["% TimeStamp"] = df_filtered["% TimeStamp"].astype(int)
            df_filtered[" Sensor Type"] = df_filtered[" Sensor Type"].astype(int)
            df_filtered[" Sensor ID"] = df_filtered[" Sensor ID"].astype(int)

            new_filename = f"{base_filename}_{str(sensor_type).zfill(2)}_{str(sensor_id).zfill(2)}.csv"
            output_path = os.path.join(processed_dir, new_filename)
            df_filtered.to_csv(output_path, index=False, sep=";")


def split_data() -> None:
    """
    Splits the CSV files in the 'processed' directory into training (70%), validation (15%), and test (15%) sets,
    and moves them into corresponding subdirectories. Also shuffles the files within each directory.

    Returns:
    None
    """
    base_dir = "../../data/processed"

    # da fare refactor
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    valid_dir = os.path.join(base_dir, "valid")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"Directory {base_dir} empty.")

    data = []

    for f in files:
        if "_ADL" in f:
            label = "ADL"
        elif "_Fall" in f:
            label = "Fall"
        else:
            continue
        data.append({"filename": f, "label": label})

    df = pd.DataFrame(data)

    X = df[["filename"]]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=0.7,
        stratify=y,
        shuffle=True,
        random_state=42,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        shuffle=True,
        random_state=42,
    )

    train_df = X_train.copy()
    train_df["label"] = y_train
    valid_df = X_valid.copy()
    valid_df["label"] = y_valid
    test_df = X_test.copy()
    test_df["label"] = y_test

    move_files(train_df, base_dir, train_dir)
    move_files(valid_df, base_dir, valid_dir)
    move_files(test_df, base_dir, test_dir)


def downsampling_and_padding(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 2000:
        df = df.iloc[::10, :]

    if len(df) >= 300:
        df = df.iloc[:300, :]
    elif len(df) < 300:
        num_needed = 300 - len(df)
        augmented_df = augment_empty_data(df.iloc[-1], num_augmentations=num_needed)
        df = pd.concat([df, augmented_df], ignore_index=True)

    return df


def augment_empty_data(row, num_augmentations=1):
    augmented_rows = []
    for _ in range(num_augmentations):
        jittered = row.copy()
        jittered[" X-Axis"] += np.random.normal(0, 0.05)
        jittered[" Y-Axis"] += np.random.normal(0, 0.05)
        jittered[" Z-Axis"] += np.random.normal(0, 0.05)
        augmented_rows.append(jittered)
    return pd.DataFrame(augmented_rows)


def move_files(df: pd.DataFrame, base_dir: str, destination_dir: str) -> None:
    """
    Moves files listed in the DataFrame to the specified destination directory.

    Parameters:
    df (pd.DataFrame): DataFrame containing filenames to be moved.
    destination_dir (str): The path to the directory where the files should be moved.

    Returns:
    None
    """
    for _, row in df.iterrows():
        src = os.path.join(base_dir, row["filename"])
        dest = os.path.join(destination_dir, row["filename"])
        shutil.move(src, dest)


if __name__ == "__main__":
    raw_directory = "../../data/raw"
    processed_directory = "../../data/processed"

    if os.path.exists(processed_directory):
        shutil.rmtree(processed_directory)

    total_files = len([f for f in os.listdir(raw_directory)])

    with Bar(
        "Processing", fill="#", suffix="%(percent).1f%% - %(eta)ds", max=total_files
    ) as bar:
        for filename in os.listdir(raw_directory):
            if filename.endswith(".csv"):
                input_file = os.path.join(raw_directory, filename)
                split_csv_by_sensor(input_file)
                bar.next()
        split_data()
