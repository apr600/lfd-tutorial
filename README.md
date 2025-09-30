# LfD DS-based Simple Tutorial

## Installation 

To load the environment, run the following code that creates a conda environment `lfd-tutorial-env` with the necessary requirements: 
```
conda env create -f environment.yml
```
To activate, run: 
```
conda activate lfd-tutorial-env
```

## Running the Code
To run the tutorial, open and run the LfD-DS Tutorial Jupyter notebook. This contains a basic example that 
1. loads the demonstration trajectory set from the included `demo_trajectories.npy`. 
2. Runs a DS-based LfD Example
3. Prints the results, including how the state would evolve over time if using the learned motion policy. 

## Learn More

This simple Learning from Demonstration example is based on the methods described in the papers listed below in the References. For a more in-depth explanation of the theory and comprehension, look at the tutorial slides and examples provided at https://epfl-lasa.github.io/TutorialICRA2019.io/. 

## References
* [1] Figueroa, N. and Billard, A. (2018) A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning. In Proceedings of the 2nd Conference on Robot Learning (CoRL).
* [2] Khansari Zadeh, S. M. and Billard, A. (2011) Learning Stable Non-Linear Dynamical Systems with Gaussian Mixture Models. IEEE Transaction on Robotics, vol. 27, num 5, p. 943-957.