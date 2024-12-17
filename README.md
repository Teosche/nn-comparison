# nn-comparison

**nn-comparison** is a thesis research project comparing the performance of LSTM (Long Short-Term Memory) and QLSTM (Quaternion Long Short-Term Memory) networks on human activity recognition (HAR) tasks. The project focuses on:

- **LSTM Networks**: Standard real-valued neural networks used for sequential data.
- **QLSTM Networks**: Neural networks leveraging quaternions for improved spatial representation and efficiency.

- **Sensor Fusion**: Quaternion data is generated using the **Madgwick Filter** to fuse accelerometer and gyroscope measurements.

## Objectives

The primary goal is to analyze and compare the accuracy and training efficiency of LSTM and QLSTM networks for human activity recognition using a sensor-based HAR dataset. Special emphasis is placed on:

- Evaluating the computational efficiency of QLSTM networks.
- Understanding the advantages of quaternion-based representations in modeling spatial rotations.
- Demonstrating potential improvements in accuracy and training time reduction.

## Project Structure

- **data/**: Directory containing raw and processed dataset.
- **src/**: Implementation of LSTM and QLSTM networks, along with utilities.

## Dataset

The dataset used in this study is derived from the **UMAFall** dataset, containing motion data captured via inertial sensors (accelerometers and gyroscopes). It includes recordings of **daily activities** and **fall events**.



### Data Setup:
1. Download the dataset from [Dropbox](https://www.dropbox.com/scl/fo/apjuwq4n4i9e8k2b5fnh8/AL1iRBcakCLoR6JbkT14ZtI?rlkey=8ihd7tkkryk5w01lukjk73uv1&st=g5va5kdg&dl=0).
2. Extract the files and replace the `data` directory in the project.
3. Re-process the raw data using the following command:
   ```bash
   python3 src/dataset/preprocessing.py

## Dataset

The dataset used for train the neural networks should be downloaded from [Dropbox](https://www.dropbox.com/scl/fo/apjuwq4n4i9e8k2b5fnh8/AL1iRBcakCLoR6JbkT14ZtI?rlkey=8ihd7tkkryk5w01lukjk73uv1&st=g5va5kdg&dl=0).
Unzip the file and replace 'data' directory.
For re-process the raw data launch:
```bash
python3 nn-comparison/src/dataset/preprocessing.py
```

## Training the Models

### LSTM

```bash
python3 nn-comparison/src/lstm/train.py
```

### QLSTM

```bash
python3 nn-comparison/src/qlstm/train.py
```

## Results Overview

While the QLSTM achieved a slightly lower accuracy than the LSTM, its performance remains promising due to significantly reduced training time and its ability to generalize well with fewer data points. This demonstrates the potential of QLSTM for efficient deployment in resource-constrained environments, such as microcontrollers.

## Citation

Eduardo Casilari, Jose A. Santoyo-Ramón, Jose M. Cano-García,
UMAFall: A Multisensor Dataset for the Research on Automatic Fall Detection,
Procedia Computer Science,
Volume 110,
2017,
Pages 32-39,
ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2017.06.110.
https://www.sciencedirect.com/science/article/pii/S1877050917312899

## License

This project is licensed under the MIT License.
