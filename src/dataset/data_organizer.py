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

    df = pd.read_csv(input_file, skiprows=40, delimiter=";")

    base_filename = os.path.basename(input_file).replace(".csv", "")
    base_filename = "_".join(base_filename.split("_")[:6])

    parts = base_filename.split("_")
    if len(parts[-1]) == 1:
        parts[-1] = parts[-1].zfill(2)
    base_filename = "_".join(parts)

    output_dir = os.path.join(os.path.dirname(input_file), "../processed")

    os.makedirs(output_dir, exist_ok=True)

    for sensor_id in df[" Sensor ID"].unique():
        for sensor_type in df[" Sensor Type"].unique():
            df_filtered = df[
                (df[" Sensor ID"] == sensor_id) & (df[" Sensor Type"] == sensor_type)
            ]

            new_filename = f"{base_filename}_{str(sensor_type).zfill(2)}_{str(sensor_id).zfill(2)}.csv"
            output_path = os.path.join(output_dir, new_filename)

            df_filtered.to_csv(output_path, index=False, sep=";")


def split_data() -> None:
    """
    Splits the CSV files in the 'processed' directory into training (70%), validation (15%), and test (15%) sets,
    and moves them into corresponding subdirectories. Also shuffles the files within each directory.

    Returns:
    None
    """
    base_dir = "../../data/processed"

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
    # df = df.sample(frac=1).reset_index(drop=True)  # uncomment for shuffle, data processed will be different

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
        print("Warning: The 'processed' directory already exist.")
        print("Running this script again will overwrite the existing files.")
        user_input = input("Do you want to proceed? (y to continue): ")
        if user_input.lower() != "y":
            print("Operation cancelled.")
            sys.exit()
        shutil.rmtree(processed_directory)

    total_files = len(
        [
            filename
            for filename in os.listdir(raw_directory)
            if filename.endswith(".csv")
        ]
    )

    with Bar(
        "Processing", fill="#", suffix="%(percent).1f%% - %(eta)ds", max=total_files
    ) as bar:
        for filename in os.listdir(raw_directory):
            if filename.endswith(".csv"):
                input_file = os.path.join(raw_directory, filename)
                split_csv_by_sensor(input_file)
                bar.next()
        split_data()
