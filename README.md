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

    a. include your optimizer class
    ```python
    class SoftSign(Optimizer):
        def __init__(
        self,
        params,
        ...
    ```
    b. Configure the optimizer's hyperparameters
    ```python
    def wrapped_train_fn(x):
        return trainer.train_on(
            params_dict,
            <Your Optimizer Class>,
            x
        )
    params_dict = {
        "lr": [-5., -2.], #hyper parameters interval you want to tune
        "weight_decay": [-3.2, -.7],#hyper parameters interval you want to tune
        "eps": [-12.3, -1.],#hyper parameters interval you want to tune
        "beta": 0.9,#fixed hyper parameters 
    }
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

