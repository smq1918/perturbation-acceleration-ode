# Acceleration via Perturbations on Low-Resolution ODEs

This repository contains the implementation code for the numerical experiments presented in our paper on acceleration methods for low-resolution ordinary differential equations.

## Repository Structure

| File/Folder          | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `main_quadratic.m`   | Core algorithm for quadratic function tests                      |
| `main_logistic.m`    | Core algorithm for logistic regression tests                     |
| `libsvmread.mexa64`  | Pre-compiled data reading utility for Linux systems              |
| `libsvmread.mexw64`  | Pre-compiled data reading utility for Windows systems            |
| `dataset/`           | Directory containing the experimental datasets                   |

## Getting Started

### Prerequisites
- MATLAB

### Usage
1. Clone the repository
2. Navigate to the directory and run in MATLAB:
   - For testing on **quadratic functions**:
     ```matlab
     main_quadratic
     ```
   
   - For testing on **logistic regression functions**:
     ```matlab
     main_nonquadratic
     ```

