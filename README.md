# QuaterNet

**QuaterNet** is a research project aimed at comparing the performance of neural networks using different data representations and optimization algorithms. Specifically, this project focuses on:

- **Real Neural Networks**: Standard neural networks operating with real numbers.
- **Quaternion Neural Networks**: Networks that leverage quaternions to represent and process multidimensional data.
- **Levenberg-Marquardt Optimized Quaternion Networks**: Quaternion networks that utilize the Levenberg-Marquardt algorithm to potentially accelerate training and improve convergence.

## Objectives

The primary goal of this project is to demonstrate that quaternion neural networks, when combined with the Levenberg-Marquardt optimization algorithm, can outperform both standard real-valued networks and quaternion networks using standard backpropagation in terms of training speed and accuracy.

## Project Structure

- **src/**: Contains the source code for different network implementations and utilities.
- **models/**: Stores trained models for the different networks.
- **logs/**: Holds log files generated during training and evaluation.
- **scripts/**: Shell scripts for automating training and evaluation processes.
- **tests/**: Unit tests to ensure code reliability.

## License

This project is licensed under the MIT License.
