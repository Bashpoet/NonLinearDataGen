# NonLinearDataGen
A Python class designed for generating synthetic nonlinear data, adding Gaussian noise, performing polynomial regression, and interpolating the fitted results

markdown
Copy
Edit
# NonlinearDataGenerator

The **NonlinearDataGenerator** is a Python class designed to generate, manipulate, and fit nonlinear datasets. It enables the creation of synthetic data using a quadratic function, introduces customizable Gaussian noise, performs polynomial regression, and interpolates the fitted curve for smooth visualization. This tool is perfect for data science education, algorithm benchmarking, and research in statistical modeling.

## 🚀 Features

- **Data Generation:** Simulates nonlinear datasets using a customizable quadratic function.
- **Noise Injection:** Adds Gaussian noise with adjustable variance to mimic real-world data irregularities.
- **Polynomial Regression:** Fits noisy data using polynomial models of arbitrary order.
- **Interpolation:** Provides smooth curves by interpolating fitted data over dense intervals.

---

## 📦 Installation

Ensure you have Python and the following libraries installed:

```bash
pip install numpy scipy
⚡ Usage
python
Copy
Edit
from nonlinear_data_gen import NonlinearDataGenerator

# Initialize with a polynomial order of 2 (quadratic)
data_gen = NonlinearDataGenerator(polynomial_order=2)

# Add Gaussian noise to the true data
data_gen.add_noise(noise_level=0.2)

# Fit the noisy data using polynomial regression
data_gen.fit_data()

# Interpolate the fitted results for smoother visualization
data_gen.interpolate_results()
🎯 Example
The following example demonstrates how to generate noisy data, fit a polynomial regression, and interpolate the fitted curve.

python
Copy
Edit
import matplotlib.pyplot as plt

# Plotting the true data, noisy data, and the fitted curve
plt.scatter(data_gen.x, data_gen.y_noisy, label='Noisy Data', color='red', s=10)
plt.plot(data_gen.x, data_gen.y_true, label='True Function', color='blue')
plt.plot(data_gen.x_interp, data_gen.y_interp, label='Fitted Curve', color='green')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Nonlinear Data Fitting with Polynomial Regression')
plt.show()
⚙️ Parameters
polynomial_order (int): Degree of the polynomial for fitting (default = 2).
noise_level (float): Standard deviation of Gaussian noise added to the data.
🔍 Use Cases
Educational Tool: Demonstrate overfitting/underfitting, regression, and noise effects in data science courses.
Machine Learning: Generate synthetic datasets for algorithm benchmarking and robustness testing.
Scientific Computing: Simulate experimental data for hypothesis testing and model validation.
🧪 Contributing
Feel free to fork this repository and contribute with pull requests, feature suggestions, or bug reports.

📜 License
This project is licensed under the MIT License.

vbnet
Copy
Edit

Let me know if you'd like to expand or modify any sections!










Search


ChatGPT can make mistakes. Check important info
