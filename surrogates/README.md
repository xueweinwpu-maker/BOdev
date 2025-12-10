Surrogate Modeling Module
1. Overview
This module is responsible for creating, training, and validating the high-fidelity surrogate models that serve as the predictive engine for the main Bayesian optimization framework. It replaces computationally expensive CFD simulations with fast, accurate statistical models, making the entire optimization process feasible.

The core components are:

model.py: Defines the GPyTorchSurrogateModel class, which is a wrapper for a single-output Gaussian Process model. It also contains a standalone script for model validation.

pipelines.py: Defines the SurrogateBackedPipeline, which integrates the trained models into the optimization workflow, providing the necessary run method for the black-box wrapper.

2. Model Architecture
The surrogate model of choice is a Gaussian Process (GP), implemented using the robust and flexible SingleTaskGP from the BoTorch library.

The specific architecture includes several key features designed for high performance in complex engineering design spaces:

Kernel: A Matern 5/2 kernel is used. This kernel is a good default choice for modeling smooth, real-world physical phenomena without being overly simplistic.

Automatic Relevance Determination (ARD): The kernel is equipped with ARD. This is a critical feature for our high-dimensional (16-variable) input space. ARD assigns a unique length-scale hyperparameter to each of the 16 input variables. During training, the model automatically learns these length-scales, effectively determining which input variables are most influential on the output. This provides a powerful, data-driven form of feature selection.

Data Scaling: The module automatically performs best-practice data scaling to ensure numerical stability and improve model accuracy.

Inputs (X): A MinMaxScaler is used to scale all 16 input variables to the range [0, 1].

Outputs (Y): A StandardScaler is used to transform the output data (e.g., CL, CD) to have a mean of 0 and a standard deviation of 1, which is the standard convention for GP models.

3. Key Advantages
This modeling approach offers several significant advantages for our aerodynamic optimization problem:

Probabilistic Predictions: Unlike many other machine learning models, a GP provides not just a single point prediction but a full posterior distribution (a mean and a variance). This measure of the model's own uncertainty is essential for the intelligent decision-making of the Bayesian optimizer's acquisition function.

Data Efficiency: GPs are exceptionally good at learning complex, non-linear functions from a relatively small number of data points. This makes them ideal for applications like ours, where each data point comes from an expensive CFD simulation.

High-Dimensional Capability: The use of an ARD kernel allows the model to effectively handle the 16-dimensional input space without being overwhelmed by uninfluential variables.

Robustness: The automated data cleaning (handling whitespace, NaN values) and scaling implemented in the script make the modeling process reliable and repeatable.

4. Usage and Validation
The quality of the entire optimization framework depends on the accuracy of these surrogate models. Therefore, a built-in validation script is provided within model.py.

How to Run the Test
To validate the models for all available flight conditions, navigate to the project's root directory in your terminal and run the following command:

python surrogates/model.py

What the Test Does
This script will automatically:

Locate the required training and verification datasets in the data/ directory.

Perform data cleaning and transformation.

Train a separate GP model for each aerodynamic coefficient (CL and CD) for each Mach number (Ma=0.4 and Ma=10.0).

For each trained model, it will evaluate its performance on the independent verification dataset.

Interpreting the Output
For each model, the script produces two forms of output:

Console Metrics: It prints four key statistical metrics to the console. High R-squared values (>0.98) and low error metrics (RMSE, MAE, MaxAE) indicate a high-quality model.

--- Validating Model for CL (Ma=0.4) ---
R-squared (R2):                  0.999535
Root Mean Squared Error (RMSE):    0.000102
Mean Absolute Error (MAE):         0.000075
Maximum Absolute Error (MaxAE):    0.000295

Validation Plots: It generates and displays a comprehensive 2x2 plot for a detailed visual assessment of the model's performance, checking for accuracy, bias, and error distribution.