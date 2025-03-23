# Muon Project

This project aims to conduct a systematic study of various algorithms, tuning them in the same way to ensure a fair comparison. The algorithms under consideration are:

- Muon
- SoftLion
- AdamW

## Objective

The objective is to determine the best performing algorithm by running the same golden search script on all three algorithms.

## Methodology

1. **Algorithm Tuning**: All algorithms will be tuned using the same parameters and settings to ensure an apple-to-apple comparison.
2. **Golden Search Script**: A standardized script will be used to evaluate the performance of each algorithm.

## Getting Started

To run the golden search script on all three algorithms, follow these steps:

0. Modify the "auto_tune.py" file to include

    a. Copy paste your optimizer class to "opt.py"
    ```python
    class <Your Optimizer Class>(Optimizer):
        def __init__(
        self,
        params,
        ...
    ```
    b. For your Optimizer, if you want to tune a parameter, for example, learning rate, beta, weight decay, eps:
    ```python
    class <Your Optimizer Class>(Optimizer):
        def __init__(
            self,
            params,
            lr: list = [-5., -2.], # log scale tuning interval
            beta: list = [0.6, 0.99], 
            eps: list = [-12.3, -1.], # log scale
            weight_decay: list = [-3., -.7], # log scale
            correct_bias: bool = True,
    ):
    ```
1. Clone the repository:
    ```sh
    git clone https://github.com/L-z-Chen/golden_section
    ```
2. Navigate to the project directory:
    ```sh
    cd golden_section
    ```
3. Run the golden search script:
    ```sh
    ./run.sh
    ```

## Results

The results of the study will be documented and analyzed to determine the best performing algorithm.

