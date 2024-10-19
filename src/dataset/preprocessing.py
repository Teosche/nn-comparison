from asyncio import sleep
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ahrs.filters import Madgwick
from progress.bar import Bar


def make_processed_df(input_file: str) -> pd.DataFrame:
    """
    Processes a CSV file by reading its contents, filtering out specific sensor data,
    and returning a DataFrame.

    Args:
        input_file (str): The path to the input CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered data.
    """
    try:
        df = pd.read_csv(input_file, skiprows=40, delimiter=";")
        df = df.iloc[:, :-1]
        df_filtered = df[(df[" Sensor ID"] != 0) & (df[" Sensor Type"] != 2)]

        return df_filtered
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")
        return pd.DataFrame()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of the DataFrame to standard names.

    Args:
        df (pd.DataFrame): The DataFrame with original column names.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    df.columns = [
        "timestamp",
        "sample",
        "x-axis",
        "y-axis",
        "z-axis",
        "sensor_type",
        "sensor_id",
    ]
    return df


def generate_new_filename(file: str) -> str:
    """
    Generates a new filename based on the original filename, adjusting for specific activities.

    Args:
        file (str): The original filename.

    Returns:
        str: The new filename formatted as per the specified rules.
    """
    base_filename = os.path.basename(file).replace(".csv", "")
    parts = base_filename.split("_")

    if "LyingDown_OnABed" in base_filename:
        activity = "ADL"
        sub_activity = "LyingDownOnABed"
        subject = f"sub{parts[2].zfill(2)}"
        exp_n = f"exp{parts[6].zfill(2)}"
    elif "Sitting_GettingUpOnAChair" in base_filename:
        activity = "ADL"
        sub_activity = "SittingGettingUpOnAChair"
        subject = f"sub{parts[2].zfill(2)}"
        exp_n = f"exp{parts[6].zfill(2)}"
    else:
        activity = parts[3]
        sub_activity = parts[4]
        subject = f"sub{parts[2].zfill(2)}"
        exp_n = f"exp{parts[5].zfill(2)}"

    return f"{activity}_{sub_activity}_{subject}_{exp_n}.csv"


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the processed DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The path where the CSV file will be saved.
        input_file (str): The path of the input file for reference in error messages.
    """
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")


def generate_csv(input_dir: str, output_dir: str) -> None:
    """
    Generates processed CSV files from all CSV files in the input directory.

    Args:
        input_directory (str): The path to the directory containing input CSV files.
        output_dir (str): The path to the directory where processed CSV files will be saved.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    raw_len = len(files)

    with Bar(
        "Processing raw data",
        fill="#",
        suffix="%(percent).1f%% - %(eta)ds",
        max=raw_len,
    ) as bar:
        for file in files:
            process_raw_file(file, input_dir, output_dir)
            bar.next()

    print(f"Generated files: {len(os.listdir(output_dir))}/{raw_len}")


def process_raw_file(file, input_dir, output_dir) -> None:
    input_file = os.path.join(input_dir, file)
    processed_df = make_processed_df(input_file)
    df_filtered = rename_columns(processed_df)
    df_filtered = preprocess_sensor_data(df_filtered)
    new_filename = generate_new_filename(file)
    new_file_path = os.path.join(output_dir, new_filename)
    save_csv(df_filtered, new_file_path)


def generate_dataframe(input_data: list[str]) -> pd.DataFrame:
    data = []

    for f in input_data:
        if "ADL_" in f:
            label = "ADL"
        elif "Fall_" in f:
            label = "Fall"
        else:
            continue
        data.append({"filename": f, "label": label})

    df = pd.DataFrame(data)

    return df


def get_csv_files(source_dir: str) -> list[str]:
    """
    Retrieves all CSV files from the source directory.

    Parameters:
    source_dir (str): The directory where the CSV files are located.

    Returns:
    list: List of CSV file names.
    """
    files = [f for f in os.listdir(source_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"Error: Directory {source_dir} is empty.")

    return files


def train_valid_test_split(df: pd.DataFrame) -> dict:
    """
    Splits the DataFrame into train, validation, and test sets.

    Parameters:
    df (pd.DataFrame): DataFrame containing filenames and labels.

    Returns:
    dict: Dictionary containing DataFrames for train, valid, and test sets.
    """
    X = df[["filename"]]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.7, stratify=y, shuffle=True, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, shuffle=True, random_state=42
    )

    train_df = X_train.copy()
    train_df["label"] = y_train
    valid_df = X_valid.copy()
    valid_df["label"] = y_valid
    test_df = X_test.copy()
    test_df["label"] = y_test

    return {"train": train_df, "valid": valid_df, "test": test_df}


def copy_files_to_split_dirs(
    file_splits: dict, source_dir: str, split_dirs: dict
) -> None:
    """
    Copy files to their corresponding train, validation, and test directories.

    Parameters:
    file_splits (dict): Dictionary containing DataFrames with filenames for each split (train, valid, test).
    source_dir (str): Directory where the files are located.
    split_dirs (dict): Dictionary with directories for each split (train, valid, test).

    Returns:
    None
    """
    for split, df in file_splits.items():
        split_dir = split_dirs[split]
        for _, row in df.iterrows():
            src_file = os.path.join(source_dir, row["filename"])
            dst_file = os.path.join(split_dir, row["filename"])
            shutil.copy(src_file, dst_file)


def generate_dirs(base_dir: str, dataset_type: str) -> dict:
    """
    Generates training, validation, and test directories for the given dataset type.

    Parameters:
    base_dir (str): Base directory where the directories will be created.
    dataset_type (str): Type of the dataset (e.g., 'real', 'quaternion').

    Returns:
    dict: Dictionary containing paths for train, valid, and test directories.
    """
    dataset_dir = os.path.join(base_dir, dataset_type)
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    valid_dir = os.path.join(dataset_dir, "valid")

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for dir_path in [train_dir, test_dir, valid_dir]:
        os.makedirs(dir_path, exist_ok=True)

    return {"train": train_dir, "valid": valid_dir, "test": test_dir}


def split_data(source_dir: str, destination_dir: str, dataset_type: str) -> None:
    """
    Splits the CSV files in the 'processed' directory into training (70%), validation (15%), and test (15%) sets,
    and moves them into corresponding subdirectories. Also shuffles the files within each directory.

    Returns:
    None
    """

    split_dirs = generate_dirs(destination_dir, dataset_type)
    files = get_csv_files(source_dir)
    df = generate_dataframe(files)
    file_splits = train_valid_test_split(df)
    copy_files_to_split_dirs(file_splits, source_dir, split_dirs)


def generate_quaternions_data(input_dir, output_dir):
    expected_sensor_ids = [1, 2, 3, 4]

    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    processed_len = len(files)

    with Bar(
        "Forging quaternions",
        fill="#",
        suffix="%(percent).1f%% - %(eta)ds",
        max=processed_len,
    ) as bar:
        for file in files:
            process_single_file(file, input_dir, output_dir, expected_sensor_ids)
            bar.next()


def process_single_file(file, input_dir, output_dir, expected_sensor_ids):
    """ """
    input_file = os.path.join(input_dir, file)
    output_file = os.path.join(output_dir, file)

    data = load_and_preprocess_data(input_file)

    quaternion_data = generate_quaternions_for_sensors(data, expected_sensor_ids)

    save_csv(quaternion_data, output_file)


def load_and_preprocess_data(input_file):
    """ """
    data = pd.read_csv(input_file)
    acc_data = data[data["sensor_type"] == 0]
    gyro_data = data[data["sensor_type"] == 1]
    present_sensor_ids = data["sensor_id"].unique()
    return {
        "acc_data": acc_data,
        "gyro_data": gyro_data,
        "present_sensor_ids": present_sensor_ids,
    }


def generate_quaternions_for_sensors(data, expected_sensor_ids):
    """ """
    quaternion_data = []

    acc_data = data["acc_data"]
    gyro_data = data["gyro_data"]
    present_sensor_ids = data["present_sensor_ids"]

    for sensor_id in expected_sensor_ids:
        if sensor_id in present_sensor_ids:
            acc_sensor_data = acc_data[acc_data["sensor_id"] == sensor_id]
            gyro_sensor_data = gyro_data[gyro_data["sensor_id"] == sensor_id]

            acc_sensor_data, gyro_sensor_data = sync_samples(
                acc_sensor_data, gyro_sensor_data
            )

            if not acc_sensor_data.empty and not gyro_sensor_data.empty:
                quaternion_df = make_quaternion_df(
                    acc_sensor_data, gyro_sensor_data, sensor_id
                )
                quaternion_data.append(quaternion_df)
        else:
            missing_data = make_missing_data(sensor_id)
            quaternion_data.append(missing_data)

    concatenated_quaternion_data = pd.concat(quaternion_data, ignore_index=True)

    concatenated_quaternion_data = concatenated_quaternion_data[
        ["sample", "timestamp", "q0", "q1", "q2", "q3", "sensor_id"]
    ]
    return concatenated_quaternion_data


def sync_samples(acc_sensor_data, gyro_sensor_data):
    """ """
    common_samples = np.intersect1d(
        acc_sensor_data["sample"].values, gyro_sensor_data["sample"].values
    )
    acc_sensor_data = acc_sensor_data[acc_sensor_data["sample"].isin(common_samples)]
    gyro_sensor_data = gyro_sensor_data[gyro_sensor_data["sample"].isin(common_samples)]
    return acc_sensor_data, gyro_sensor_data


def make_quaternion_df(
    acc_sensor_data: pd.DataFrame, gyro_sensor_data: pd.DataFrame, sensor_id: int
) -> pd.DataFrame:
    """ """
    madgwick = Madgwick(
        gyr=gyro_sensor_data[["x-axis", "y-axis", "z-axis"]].values,
        acc=acc_sensor_data[["x-axis", "y-axis", "z-axis"]].values,
    )
    quaternions = madgwick.Q

    quaternion_df = pd.DataFrame(quaternions, columns=["q0", "q1", "q2", "q3"])
    quaternion_df["timestamp"] = (
        acc_sensor_data["timestamp"].astype(int).values[: len(quaternion_df)]
    )
    quaternion_df["sample"] = (
        acc_sensor_data["sample"].astype(int).values[: len(quaternion_df)]
    )
    quaternion_df["sensor_id"] = sensor_id

    return quaternion_df


def make_missing_data(sensor_id: int) -> pd.DataFrame:
    """ """
    missing_data = pd.DataFrame(
        np.nan, index=range(300), columns=["q0", "q1", "q2", "q3"]
    )
    missing_data["timestamp"] = np.nan
    missing_data["sample"] = np.nan
    missing_data["sensor_id"] = sensor_id

    return missing_data


def adjust_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust the 'sample' column so that each group starts from 0."""
    df["sample"] = df.groupby(["sensor_id", "sensor_type"])["sample"].transform(
        lambda x: x - x.min()
    )
    return df


def downsampling_and_padding(df: pd.DataFrame, sensor_id, sensor_type) -> pd.DataFrame:
    df = df.sort_values(by="sample").reset_index(drop=True)

    num_samples = len(df)

    if num_samples > 300:
        df = df.iloc[:300].copy()
    elif num_samples < 300:
        num_needed = 300 - num_samples
        augmented_df = augment_empty_data(
            df.iloc[-1], num_augmentations=num_needed, start_sample=num_samples
        )
        df = pd.concat([df, augmented_df], ignore_index=True)

    df["sample"] = range(300)
    return df


def augment_empty_data(row, num_augmentations=1, start_sample=0):
    """
    Function that creates fake samples for padding.
    Adds a 'sample' column starting from 'start_sample'.
    """
    augmented_rows = []
    for i in range(num_augmentations):
        jittered = row.copy()
        jittered["x-axis"] += np.random.normal(0, 0.05)
        jittered["y-axis"] += np.random.normal(0, 0.05)
        jittered["z-axis"] += np.random.normal(0, 0.05)
        jittered["sample"] = start_sample + i
        augmented_rows.append(jittered)

    return pd.DataFrame(augmented_rows)


def preprocess_sensor_data(df: pd.DataFrame):
    """
    Main function that processes the data for each sensor_id and sensor_type.
    """
    df = adjust_sample(df)

    processed_dfs = []
    for (sensor_id, sensor_type), group in df.groupby(["sensor_id", "sensor_type"]):
        processed_df = downsampling_and_padding(group, sensor_id, sensor_type)
        processed_dfs.append(processed_df)

    result_df = pd.concat(processed_dfs, ignore_index=True)

    result_df["timestamp"] = result_df["timestamp"].astype(int)
    result_df["sensor_id"] = result_df["sensor_id"].astype(int)
    result_df["sensor_type"] = result_df["sensor_type"].astype(int)

    return result_df


def remake_dir(dir: str) -> None:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


if __name__ == "__main__":
    data_dir = "../../data"
    raw_dir = "../../data/raw"
    processed_dir = "../../data/processed"
    quaternions_dir = "../../data/quaternion"

    remake_dir(processed_dir)
    remake_dir(quaternions_dir)

    generate_csv(raw_dir, processed_dir)

    generate_quaternions_data(processed_dir, quaternions_dir)

    split_data(processed_dir, data_dir, "real_dataset")
    split_data(quaternions_dir, data_dir, "quaternions_dataset")
