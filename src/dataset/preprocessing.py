import os
import pandas as pd
from progress.bar import Bar


def process_file(input_file: str) -> pd.DataFrame:
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
        activity = "LyingDownOnABed"
        sub_activity = parts[5]
        subject = f"sub{parts[2].zfill(2)}"
        exp_n = f"exp{parts[6].zfill(2)}"
    elif "Sitting_GettingUpOnAChair" in base_filename:
        activity = "SittingGettingUpOnAChair"
        sub_activity = parts[5]
        subject = f"sub{parts[2].zfill(2)}"
        exp_n = f"exp{parts[6].zfill(2)}"
    else:
        activity = parts[3]
        sub_activity = parts[4]
        subject = f"sub{parts[2].zfill(2)}"
        exp_n = f"exp{parts[5].zfill(2)}"

    return f"{activity}_{sub_activity}_{subject}_{exp_n}.csv"


def save_csv(df: pd.DataFrame, output_path: str, input_file: str) -> None:
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


def generate_csv(input_directory: str, output_dir: str) -> None:
    """
    Generates processed CSV files from all CSV files in the input directory.

    Args:
        input_directory (str): The path to the directory containing input CSV files.
        output_dir (str): The path to the directory where processed CSV files will be saved.
    """
    files = [f for f in os.listdir(input_directory) if f.endswith(".csv")]
    raw_len = len(files)

    with Bar(
        "Processing", fill="#", suffix="%(percent).1f%% - %(eta)ds", max=raw_len
    ) as bar:
        for file in files:
            input_file = os.path.join(input_directory, file)
            df_filtered = process_file(input_file)

            df_filtered = rename_columns(df_filtered)
            new_filename = generate_new_filename(file)
            new_file_path = os.path.join(output_dir, new_filename)
            save_csv(df_filtered, new_file_path, input_file)

            bar.next()

    print(f"Generated files: {len(os.listdir(output_dir))}/{raw_len}")


if __name__ == "__main__":
    raw_dir = "../../data/raw"
    processed_dir = "../../data/processed"

    os.makedirs(processed_dir, exist_ok=True)

    generate_csv(raw_dir, processed_dir)
