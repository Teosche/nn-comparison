import os
import pandas as pd
from progress.bar import Bar


def split_by_sensor_tag(input_file: str) -> None:
    rotation_matrices_data_dir = "../../data/rotation_matrices"
    os.makedirs(rotation_matrices_data_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file, skiprows=40, delimiter=";")
    except Exception as e:
        print(f"read error: {input_file}: {e}")
        return

    base_filename = os.path.basename(input_file).replace(".csv", "")
    parts = base_filename.split("_")

    activity = parts[3]
    sub_activity = parts[4]
    subject = f"sub{parts[2].zfill(2)}"
    exp_n = f"exp{parts[5].zfill(2)}"

    sensor_ids = df[" Sensor ID"].unique()

    for sensor_id in sensor_ids:
        if sensor_id == 0:
            continue

        df_filtered = df[df[" Sensor ID"] == sensor_id]
        sensor_tag = f"st{str(sensor_id).zfill(2)}"
        new_filename = f"{activity}_{sub_activity}_{subject}_{exp_n}_{sensor_tag}.csv"

        new_file_path = os.path.join(rotation_matrices_data_dir, new_filename)

        try:
            df_filtered.to_csv(new_file_path, index=False)
        except Exception as e:
            print(f"save error: {new_file_path}: {e}")


if __name__ == "__main__":
    raw_directory = "../../data/raw"

    raw_len = len([f for f in os.listdir(raw_directory)])

    with Bar(
        "Processing", fill="#", suffix="%(percent).1f%% - %(eta)ds", max=raw_len
    ) as bar:
        for file in os.listdir(raw_directory):
            input_file = os.path.join(raw_directory, file)
            split_by_sensor_tag(input_file)
            bar.next()
