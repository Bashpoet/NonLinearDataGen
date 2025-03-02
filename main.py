import numpy as np
from scipy import optimize

class NonlinearDataGenerator:
    def __init__(self, true_function=None, function_params=None, polynomial_order=2):
        self.polynomial_order = polynomial_order
        self.x = np.linspace(0, 10, 100)

        # Default to quadratic if no function is provided
        if true_function is None:
            self.true_function = self._default_quadratic
            self.function_params = {'a': 3, 'b': 5, 'c': 7} if function_params is None else function_params
        else:
            self.true_function = true_function
            self.function_params = {} if function_params is None else function_params  # Ensure params are a dict

        self.y_true = self.true_function(self.x, **self.function_params) #Use the parameters


    @staticmethod
    def _default_quadratic(x, a=3, b=5, c=7):  # Private method for the default
        return a * x**2 + b * x + c

    # Example of another preset function
    @staticmethod
    def exponential_function(x, a=1, b=1, c=0):
        return a * np.exp(b * x) + c
    
    def add_noise(self, noise_level=0.1, noise_type="gaussian"):
        """Adds noise to the true data.

        Parameters:
            noise_level (float):  Magnitude of the noise.  Interpretation depends on noise_type.
            noise_type (str): Type of noise.  'gaussian' (default), 'uniform'.  Could add more.
        """

        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, size=len(self.x))
        elif noise_type == "uniform":
            noise = np.random.uniform(-noise_level, noise_level, size=len(self.x))
        # Add more noise types here (e.g., Poisson, etc.) as needed.
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")

        self.y_noisy = self.y_true + noise

    # ... (rest of the class remains largely the same, adjust fit_data if needed) ...

    def fit_data(self, fit_function=None):
        """Fits the noisy data.  Allows custom fit functions.

        Args:
            fit_function (callable, optional):  The function to fit.  If None, defaults
                to a polynomial of order self.polynomial_order.
        """

        if fit_function is None:
            def model(x, *coefs):
                return sum(coefs[i] * x**i for i in range(len(coefs)))
            p0 = np.ones(self.polynomial_order + 1)  # Initial guess for polynomial
        else:
            model = fit_function
            #  A robust implementation would try to infer the number of parameters
            #  from the fit_function (using inspect.signature), but that's more advanced.
            #  For simplicity, we'll assume the user provides a reasonable p0.
            p0 = self.function_params if self.function_params else [1.0] * (self.polynomial_order + 1) # or some other reasonable default


        self.coefs, self.cov_matrix = optimize.curve_fit(model, self.x, self.y_noisy, p0=p0, full_output=False)


    def interpolate_results(self, x_new=None):
        """Interpolates the fitted results.  Now takes optional x values."""

        if x_new is None:
            self.x_interp = np.linspace(0, 10, 500)
        else:
            self.x_interp = x_new

        # Use the fitted coefficients for interpolation
        if callable(self.coefs):  # If coefs is actually a function (unlikely, but good practice)
            self.y_interp = self.coefs(self.x_interp)  #  Assume user knows how to call it
        else:
             self.y_interp = np.polyval(self.coefs[::-1], self.x_interp) # Assume polynomial


# Example usage with custom function:
def my_sine_function(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset

sine_params = {'amplitude': 2, 'frequency': 0.5, 'phase': 1, 'offset': 3}
sine_generator = NonlinearDataGenerator(true_function=my_sine_function, function_params=sine_params)
sine_generator.add_noise(noise_level=0.5)
sine_generator.fit_data(fit_function=my_sine_function)  # Pass the fitting function!
sine_generator.interpolate_results()

# Example Usage with preset
exp_gen = NonlinearDataGenerator(true_function=NonlinearDataGenerator.exponential_function)
exp_gen.add_noise()
exp_gen.fit_data(fit_function=NonlinearDataGenerator.exponential_function) # Pass fitting function
exp_gen.interpolate_results()

# Example usage with default quadratic
data_gen = NonlinearDataGenerator()  # Defaults to quadratic
data_gen.add_noise()
data_gen.fit_data()
data_gen.interpolate_results()
