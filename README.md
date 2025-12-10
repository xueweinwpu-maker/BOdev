Trans-Domain Morphing Aircraft Optimization Framework
This project contains the source code for a dual-layer Bayesian optimization framework designed for the multi-objective aerodynamic optimization of trans-domain morphing aircraft.

Project Structure
main.py: The main entry point to run the optimization framework.

README.md: This file.

src/: Contains the core source code for the optimization logic.

problem_def.py: The master configuration file for defining an optimization problem.

bo_optimizer.py: The main Bayesian Optimizer class.

blackbox_wrapper.py: The wrapper for the aerodynamic evaluation function, handling inner-loop optimization.

post_processing.py: Functions for plotting and visualizing results.

history_logger.py: Module for saving the optimization history to a CSV file.

surrogates/: A self-contained module for building and managing surrogate models.

model.py: Defines the GPyTorchSurrogateModel class for training and validation.

pipelines.py: Defines the SurrogateBackedPipeline that uses the trained models for evaluation.

data/: Contains all the CSV datasets used for training and validating the surrogate models.

train_m04.csv

verify_m04.csv

train_m10.csv

verify_m10.csv

docs/: Contains all project documentation and paper drafts.

How to Run
Ensure all required Python packages (e.g., torch, botorch, gpytorch, pandas, scikit-learn, matplotlib) are installed.

Place your training and verification CSV files in the data/ directory.

Configure the optimization problem in src/problem_def.py.

Run the main script from the root directory:

python main.py
