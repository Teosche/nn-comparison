import os
import pandas as pd


def split_csv_by_sensor(input_file):
    """
    Splits a CSV file by sensor ID and sensor type.

    This function reads a CSV file, filters the data by unique sensor IDs and sensor types,
    and then writes separate CSV files for each combination of sensor ID and sensor type.
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
            print(f"File saved: {new_filename}")


if __name__ == "__main__":
    input_directory = "../../data/raw"

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_file = os.path.join(input_directory, filename)
            split_csv_by_sensor(input_file)
