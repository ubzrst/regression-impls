# Regression Implementations
---
A repository of simple implementations of common regression models used in machine learning:
- Linear Regression
- Polynomial Regression
- Decision Tree Regression

## Linear Regression
---
Linear regression models a real-valued target as a function that is linear in its parameters. The model is given by
$$
y = \theta^\top \mathbf{x} + b + \varepsilon
$$
where $\mathbf{x} \in \mathbb{R}^d$ is the vector of predictors, $\theta \in \mathbb{R}^d$ is the parameter vector, $b \in \mathbb{R}$ is the bias term, and $\varepsilon$ represents noise capturing unexplained variation.

Given an input $\mathbf{x}$, the predicted value is
$$
\hat{y} = \theta^\top \mathbf{x} + b.
$$

The parameters are typically estimated by minimizing the mean squared error (MSE) loss:
$$
J(\theta, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2,
$$
where $n$ is the number of training examples. This choice of loss corresponds to assuming zero-mean Gaussian noise.

One common optimization method is gradient descent, which iteratively updates the parameters in the direction of the negative gradient of the loss:
$$
\theta \leftarrow \theta - \alpha \nabla_\theta J,
$$
$$
b \leftarrow b - \alpha \frac{\partial J}{\partial b},
$$
where $\alpha > 0$ is the learning rate.

