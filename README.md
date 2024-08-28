# nn-comparison

**nn-comparison** is a research project aimed at comparing the performance of neural networks using different data representations and optimization algorithms. Specifically, this project focuses on:

- **Real Neural Networks**: Standard neural networks operating with real numbers.
- **Quaternion Neural Networks**: Networks that leverage quaternions to represent and process multidimensional data.
- **Levenberg-Marquardt Optimized Quaternion Networks**: Quaternion networks that utilize the Levenberg-Marquardt algorithm to potentially accelerate training and improve convergence.

The dataset used for these studies is a dataset of falls and daily activities, derived from the "UMAFall: A Multisensor Dataset for the Research on Automatic Fall Detection" study. This dataset includes movement traces acquired through systematic emulation of predefined Activities of Daily Life (ADLs) and falls. Details of the study are cited below.


## Objectives

The primary goal of this project is to demonstrate that quaternion neural networks, when combined with the Levenberg-Marquardt optimization algorithm, can outperform both standard real-valued networks and quaternion networks using standard backpropagation in terms of training speed and accuracy.

## Project Structure

- **data/**: Directory for dataset. Should be contain 'processed' and 'raw' directories.
- **src/**: Contains the source code for different network implementations and utilities.
- **models/**: Stores trained models for the different networks.
- **logs/**: Holds log files generated during training and evaluation.
- **scripts/**: Shell scripts for automating training and evaluation processes.
- **tests/**: Unit tests to ensure code reliability.

## Dataset

The dataset used for train the neural networks should be downloaded from [Dropbox](https://www.dropbox.com/scl/fo/apjuwq4n4i9e8k2b5fnh8/AL1iRBcakCLoR6JbkT14ZtI?rlkey=8ihd7tkkryk5w01lukjk73uv1&st=g5va5kdg&dl=0).
Unzip the file and replace 'data' directory.
For re-process the raw data launch:
```bash
python3 nn-comparison/src/dataset/preprocessing.py
```

## Citation

Eduardo Casilari, Jose A. Santoyo-Ramón, Jose M. Cano-García,
UMAFall: A Multisensor Dataset for the Research on Automatic Fall Detection,
Procedia Computer Science,
Volume 110,
2017,
Pages 32-39,
ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2017.06.110.
(https://www.sciencedirect.com/science/article/pii/S1877050917312899)

## License

This project is licensed under the MIT License.
