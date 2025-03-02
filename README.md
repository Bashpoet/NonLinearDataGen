# NonlinearDataGenerator

A flexible Python framework for generating, fitting, and visualizing synthetic nonlinear data with customizable noise profiles.

![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%20|%20scipy%20|%20matplotlib-orange.svg)

## 🚀 Overview

The **NonlinearDataGenerator** is an advanced framework for simulating, manipulating, and analyzing nonlinear data patterns. It provides researchers, data scientists, and educators with a comprehensive toolkit for generating synthetic datasets, adding realistic noise profiles, fitting mathematical models, and evaluating results with robust error metrics.

## ✨ Key Features

- **Customizable Function Library**: Generate data from built-in functions (polynomial, exponential, sigmoid, sine) or define your own custom functions.
- **Diverse Noise Models**: Add various types of noise (Gaussian, uniform, proportional, outliers) to simulate real-world data imperfections.
- **Flexible Fitting Options**: Fit data using polynomial regression or custom functions with support for robust fitting methods to handle outliers.
- **Comprehensive Error Metrics**: Evaluate model performance using multiple metrics (MSE, RMSE, MAE, R²) and parameter uncertainty estimates.
- **Publication-Quality Visualization**: Generate professional plots with optional confidence intervals for clear result interpretation.
- **Extensive Documentation**: Detailed docstrings and examples to guide users through all functionality.

## 📊 Example Usage

```python
# Basic usage with default quadratic function
data_gen = NonlinearDataGenerator()
data_gen.add_noise(noise_level=0.2)
data_gen.fit_data()
data_gen.interpolate_results()
data_gen.plot_results()

# Using a custom sine function with outliers
sine_params = {'amplitude': 2, 'frequency': 0.5, 'phase': 1, 'offset': 3}
sine_gen = NonlinearDataGenerator(
    true_function=NonlinearDataGenerator.sine_function,
    function_params=sine_params
)
sine_gen.add_noise(noise_level=0.3, noise_type='outliers')
sine_gen.fit_data(fit_function=NonlinearDataGenerator.sine_function, robust=True)
sine_gen.interpolate_results()
sine_gen.plot_results(include_confidence=True)
```

## 📋 Applications

- **Research**: Test and benchmark curve-fitting algorithms and noise-reduction techniques
- **Education**: Demonstrate statistical concepts, regression analysis, and uncertainty quantification
- **Data Science**: Generate synthetic datasets for machine learning model validation
- **Signal Processing**: Simulate signal recovery from noisy measurements
- **Time Series Analysis**: Model and forecast data with nonlinear trends

## 📝 Class Structure

```
NonlinearDataGenerator
├── __init__(true_function, function_params, polynomial_order, x_range, num_points)
├── Built-in Functions
│   ├── _default_quadratic(x, a, b, c)
│   ├── exponential_function(x, a, b, c)
│   ├── sigmoid_function(x, a, b, c, d)
│   └── sine_function(x, amplitude, frequency, phase, offset)
├── Data Operations
│   ├── add_noise(noise_level, noise_type, seed)
│   ├── fit_data(fit_function, p0, method, bounds, robust)
│   └── interpolate_results(x_new, num_points)
└── Analysis and Visualization
    ├── calculate_error()
    └── plot_results(include_confidence, confidence_level, figure_size)
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/NonlinearDataGenerator.git

# Install dependencies
pip install numpy scipy matplotlib
```

## 🧪 Use Cases

1. **Algorithm Development**: Test and optimize curve-fitting algorithms with controlled noise levels
2. **Educational Demonstrations**: Show how different noise patterns affect model fitting
3. **Benchmark Testing**: Compare performance of different fitting methods or libraries
4. **Data Augmentation**: Generate additional training data for machine learning models
5. **Sensitivity Analysis**: Assess how parameter changes affect output with varying noise conditions

## 📚 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
