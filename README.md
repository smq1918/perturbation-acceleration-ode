# Acceleration via Perturbations on Low-Resolution ODEs

This repository contains the code for the numerical experiments presented in our paper "Acceleration via Perturbations on Low-resolution
Ordinary Differential Equations".

## Repository Structure

| File/Folder          | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `main_quadratic.m`   | Core algorithm for tests on quadratic programming                 |
| `main_logistic.m`    | Core algorithm for tests on logistic regression                   |
| `libsvmread.mexa64`  | Pre-compiled code for reading data on Linux                       |
| `libsvmread.mexw64`  | Pre-compiled code for reading data on Windows                     |
| `dataset/`           | Directory containing the datasets used in the tests               |

## Getting Started

### Prerequisites
- MATLAB

### Usage
1. Clone the repository
2. Navigate to the directory and run in MATLAB:
   - Numerical experiments on convex quadratic programming:
     ```matlab
     main_quadratic
     ```
   
   - Numerical experiments on $\ell_2$-regularized logistic regression:
     ```matlab
     main_nonquadratic
     ```

