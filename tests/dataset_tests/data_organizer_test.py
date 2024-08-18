import pandas as pd
import pytest
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from dataset import data_organizer


def test_split_csv_by_sensor(mocker) -> None:
    mock_csv = pd.DataFrame(
        {
            "% TimeStamp": [145, 145, 146],
            " Sample No": [1, 2, 3],
            " X-Axis": [-0.4218206405639648, -0.5496244430541992, -0.6334832310676575],
            " Y-Axis": [1.136497378349304, 0.902985155582428, 0.8695393800735474],
            " Z-Axis": [0.278784453868866, 0.1047172695398331, 0.0339203476905822],
            " Sensor Type": [0, 0, 0],
            " Sensor ID": [0, 1, 2],
        }
    )

    mock_read_csv = mocker.patch("pandas.read_csv", return_value=mock_csv)
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
    mock_makedirs = mocker.patch("os.makedirs")
    mocker.patch("os.path.basename", return_value="mock_data.csv")
    mocker.patch("os.path.dirname", return_value="data/raw")
    mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))

    input_file = "data/raw/mock_data.csv"
    data_organizer.split_csv_by_sensor(input_file)

    mock_read_csv.assert_called_once_with(input_file, skiprows=40, delimiter=";")

    expected_number_of_files = 3

    assert mock_to_csv.call_count == expected_number_of_files

    expected_filenames = [
        "data/raw/../processed/mock_data_00_00.csv",
        "data/raw/../processed/mock_data_00_01.csv",
        "data/raw/../processed/mock_data_00_02.csv",
    ]

    for expected_file in expected_filenames:
        mock_to_csv.assert_any_call(expected_file, index=False, sep=";")

    mock_makedirs.assert_called_once_with("data/raw/../processed", exist_ok=True)
