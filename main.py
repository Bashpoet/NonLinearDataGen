import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, Tuple, Union, Any


class NonlinearDataGenerator:
    """
    A class for generating and fitting nonlinear data with various functions and noise patterns.
    
    This class allows you to:
    1. Generate synthetic data using custom or predefined functions
    2. Add different types of noise to your data
    3. Fit models to noisy data
    4. Interpolate and visualize results
    5. Calculate error metrics
    """
    
    def __init__(self, true_function: Optional[Callable] = None, 
                function_params: Optional[Dict[str, float]] = None, 
                polynomial_order: int = 2,
                x_range: Tuple[float, float] = (0, 10),
                num_points: int = 100):
        """
        Initialize the NonlinearDataGenerator with a specified function and parameters.
        
        Parameters
        ----------
        true_function : callable, optional
            The function to generate true data. If None, defaults to quadratic.
        function_params : dict, optional
            Parameters for the true_function. If None, defaults are used.
        polynomial_order : int, optional
            Order of polynomial for default fitting. Default is 2.
        x_range : tuple(float, float), optional
            Range of x values to generate. Default is (0, 10).
        num_points : int, optional
            Number of data points to generate. Default is 100.
        """
        self.polynomial_order = polynomial_order
        self.x = np.linspace(x_range[0], x_range[1], num_points)
        
        # Default to quadratic if no function is provided
        if true_function is None:
            self.true_function = self._default_quadratic
            self.function_params = {'a': 3, 'b': 5, 'c': 7} if function_params is None else function_params
        else:
            self.true_function = true_function
            self.function_params = {} if function_params is None else function_params
        
        # Generate true y values
        self.y_true = self.true_function(self.x, **self.function_params)
        
        # Initialize other attributes
        self.y_noisy = None
        self.coefs = None
        self.cov_matrix = None
        self.x_interp = None
        self.y_interp = None
        self.fit_function = None
        self.fit_params = None
    
    @staticmethod
    def _default_quadratic(x: np.ndarray, a: float = 3, b: float = 5, c: float = 7) -> np.ndarray:
        """
        Default quadratic function: f(x) = a*x^2 + b*x + c.
        
        Parameters
        ----------
        x : np.ndarray
            Input values.
        a, b, c : float, optional
            Coefficients. Defaults are 3, 5, and 7.
            
        Returns
        -------
        np.ndarray
            Function values at x.
        """
        return a * x**2 + b * x + c
    
    @staticmethod
    def exponential_function(x: np.ndarray, a: float = 1, b: float = 1, c: float = 0) -> np.ndarray:
        """
        Exponential function: f(x) = a * exp(b * x) + c.
        
        Parameters
        ----------
        x : np.ndarray
            Input values.
        a, b, c : float, optional
            Coefficients. Defaults are 1, 1, and 0.
            
        Returns
        -------
        np.ndarray
            Function values at x.
        """
        return a * np.exp(b * x) + c
    
    @staticmethod
    def sigmoid_function(x: np.ndarray, a: float = 1, b: float = 1, c: float = 0, d: float = 0) -> np.ndarray:
        """
        Sigmoid function: f(x) = a / (1 + exp(-b * (x - c))) + d.
        
        Parameters
        ----------
        x : np.ndarray
            Input values.
        a, b, c, d : float, optional
            Coefficients. Defaults are 1, 1, 0, and 0.
            
        Returns
        -------
        np.ndarray
            Function values at x.
        """
        return a / (1 + np.exp(-b * (x - c))) + d
    
    @staticmethod
    def sine_function(x: np.ndarray, amplitude: float = 1, frequency: float = 1, 
                      phase: float = 0, offset: float = 0) -> np.ndarray:
        """
        Sine function: f(x) = amplitude * sin(frequency * x + phase) + offset.
        
        Parameters
        ----------
        x : np.ndarray
            Input values.
        amplitude, frequency, phase, offset : float, optional
            Coefficients. Defaults are 1, 1, 0, and 0.
            
        Returns
        -------
        np.ndarray
            Function values at x.
        """
        return amplitude * np.sin(frequency * x + phase) + offset
    
    def add_noise(self, noise_level: float = 0.1, noise_type: str = "gaussian", 
                  seed: Optional[int] = None) -> np.ndarray:
        """
        Add noise to the true data.
        
        Parameters
        ----------
        noise_level : float, optional
            Magnitude of the noise. Default is 0.1.
        noise_type : str, optional
            Type of noise. Options: 'gaussian', 'uniform', 'proportional', 'outliers'.
            Default is 'gaussian'.
        seed : int, optional
            Random seed for reproducibility. Default is None.
            
        Returns
        -------
        np.ndarray
            The noisy data.
            
        Raises
        ------
        ValueError
            If an invalid noise_type is specified.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if noise_type == "gaussian":
            # Standard Gaussian noise
            noise = np.random.normal(0, noise_level, size=len(self.x))
        elif noise_type == "uniform":
            # Uniform noise in range [-noise_level, noise_level]
            noise = np.random.uniform(-noise_level, noise_level, size=len(self.x))
        elif noise_type == "proportional":
            # Noise proportional to signal magnitude
            noise = np.random.normal(0, noise_level * np.abs(self.y_true), size=len(self.x))
        elif noise_type == "outliers":
            # Gaussian noise with occasional outliers
            noise = np.random.normal(0, noise_level, size=len(self.x))
            # Add outliers to ~5% of points
            outlier_idx = np.random.choice(len(self.x), size=int(0.05 * len(self.x)), replace=False)
            noise[outlier_idx] *= 5
        else:
            raise ValueError(f"Invalid noise type: {noise_type}. Choose from 'gaussian', 'uniform', 'proportional', or 'outliers'.")
        
        self.y_noisy = self.y_true + noise
        return self.y_noisy
    
    def fit_data(self, fit_function: Optional[Callable] = None, 
                 p0: Optional[Union[Dict[str, float], np.ndarray, list]] = None,
                 method: str = 'lm', bounds=(-np.inf, np.inf), 
                 robust: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the noisy data using a specified function or default polynomial.
        
        Parameters
        ----------
        fit_function : callable, optional
            The function to fit. If None, defaults to a polynomial of order self.polynomial_order.
        p0 : dict or array-like, optional
            Initial parameter guess. If None, uses reasonable defaults.
        method : str, optional
            Optimization method for curve_fit. Default is 'lm'.
        bounds : tuple, optional
            Bounds for parameters. Default is (-inf, inf).
        robust : bool, optional
            Whether to use robust fitting to handle outliers. Default is False.
            
        Returns
        -------
        tuple
            (coefs, cov_matrix): The fitted coefficients and covariance matrix.
            
        Raises
        ------
        ValueError
            If y_noisy is None (add_noise not called) or fit fails.
        """
        if self.y_noisy is None:
            raise ValueError("No noisy data available. Call add_noise() first.")
        
        self.fit_function = fit_function
        
        if fit_function is None:
            # Default to polynomial fitting
            def model(x, *coefs):
                return sum(coefs[i] * x**i for i in range(len(coefs)))
            
            p0_array = np.ones(self.polynomial_order + 1) if p0 is None else np.array(p0)
        else:
            # Use the provided function
            model = fit_function
            
            # Handle different types of initial parameter guesses
            if p0 is None:
                # Use function_params if available, otherwise default to ones
                if isinstance(self.function_params, dict) and self.function_params:
                    p0_array = np.array(list(self.function_params.values()))
                else:
                    # Default to array of ones (risky but simple)
                    p0_array = np.ones(self.polynomial_order + 1)
            elif isinstance(p0, dict):
                # Convert dict to array in the expected order
                p0_array = np.array(list(p0.values()))
            else:
                # Assume p0 is already in array-like format
                p0_array = np.array(p0)
        
        try:
            # Set fitting options based on robustness
            fit_kws = {'method': method, 'bounds': bounds}
            if robust:
                fit_kws.update({'loss': 'soft_l1', 'f_scale': 0.1})
            
            # Perform the fit
            self.coefs, self.cov_matrix = optimize.curve_fit(
                model, self.x, self.y_noisy, p0=p0_array, **fit_kws
            )
            
            # Store the fitting parameters for later use
            self.fit_params = {
                'function': fit_function,
                'model': model,
                'p0': p0_array
            }
            
            return self.coefs, self.cov_matrix
            
        except Exception as e:
            raise ValueError(f"Fitting failed: {str(e)}") from e
    
    def interpolate_results(self, x_new: Optional[np.ndarray] = None, 
                           num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the fitted results at new x values.
        
        Parameters
        ----------
        x_new : np.ndarray, optional
            The x values for interpolation. If None, generates a linear space.
        num_points : int, optional
            Number of points to generate if x_new is None. Default is 500.
            
        Returns
        -------
        tuple
            (x_interp, y_interp): The interpolated x and y values.
            
        Raises
        ------
        ValueError
            If fit_data() has not been called.
        """
        if self.coefs is None:
            raise ValueError("No fitted coefficients available. Call fit_data() first.")
        
        # Set interpolation x values
        if x_new is None:
            x_min, x_max = min(self.x), max(self.x)
            self.x_interp = np.linspace(x_min, x_max, num_points)
        else:
            self.x_interp = x_new
        
        # Calculate interpolated y values
        if self.fit_function is None:
            # For polynomial fits, use polyval
            self.y_interp = np.polyval(self.coefs[::-1], self.x_interp)
        else:
            # For custom functions, reconstruct the model
            model = self.fit_params['model']
            self.y_interp = model(self.x_interp, *self.coefs)
        
        return self.x_interp, self.y_interp
    
    def calculate_error(self) -> Dict[str, float]:
        """
        Calculate error metrics between true, noisy, and fitted data.
        
        Returns
        -------
        dict
            Dictionary of error metrics.
            
        Raises
        ------
        ValueError
            If required data is missing.
        """
        if self.y_noisy is None or self.coefs is None:
            raise ValueError("Missing data. Ensure add_noise() and fit_data() have been called.")
        
        # For error against noisy data, we need the fitted curve at original x points
        if self.fit_function is None:
            # Polynomial fit
            y_fit = np.polyval(self.coefs[::-1], self.x)
        else:
            # Custom function fit
            model = self.fit_params['model']
            y_fit = model(self.x, *self.coefs)
        
        # Calculate errors
        mse_true = np.mean((self.y_true - y_fit) ** 2)
        rmse_true = np.sqrt(mse_true)
        mae_true = np.mean(np.abs(self.y_true - y_fit))
        
        mse_noisy = np.mean((self.y_noisy - y_fit) ** 2)
        rmse_noisy = np.sqrt(mse_noisy)
        mae_noisy = np.mean(np.abs(self.y_noisy - y_fit))
        
        # R-squared calculation
        ss_total = np.sum((self.y_noisy - np.mean(self.y_noisy)) ** 2)
        ss_residual = np.sum((self.y_noisy - y_fit) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        return {
            'MSE_true': mse_true,
            'RMSE_true': rmse_true,
            'MAE_true': mae_true,
            'MSE_noisy': mse_noisy,
            'RMSE_noisy': rmse_noisy,
            'MAE_noisy': mae_noisy,
            'R_squared': r_squared,
            'Parameter_uncertainty': np.sqrt(np.diag(self.cov_matrix)) if self.cov_matrix is not None else None
        }
    
    def plot_results(self, include_confidence: bool = False, 
                    confidence_level: float = 0.95, 
                    figure_size: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the original data, noisy data, and fitted curve.
        
        Parameters
        ----------
        include_confidence : bool, optional
            Whether to include confidence intervals. Default is False.
        confidence_level : float, optional
            Confidence level for intervals (0-1). Default is 0.95 (95%).
        figure_size : tuple, optional
            Size of the figure (width, height). Default is (10, 6).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
            
        Raises
        ------
        ValueError
            If required data is missing.
        """
        if self.y_noisy is None or self.coefs is None or self.y_interp is None:
            raise ValueError(
                "Missing data for plotting. Ensure add_noise(), fit_data(), "
                "and interpolate_results() have been called."
            )
        
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Plot true function
        ax.plot(self.x, self.y_true, 'b-', linewidth=2, label='True Function')
        
        # Plot noisy data
        ax.scatter(self.x, self.y_noisy, color='red', s=30, alpha=0.5, label='Noisy Data')
        
        # Plot fitted curve
        ax.plot(self.x_interp, self.y_interp, 'g--', linewidth=2, label='Fitted Function')
        
        # Add confidence intervals if requested
        if include_confidence and self.cov_matrix is not None:
            try:
                # This is simplified and works best with the default polynomial case
                # For custom functions, would need more sophisticated approach
                from scipy import stats
                
                # Get the appropriate t-value for the confidence level
                alpha = 1 - confidence_level
                dof = max(0, len(self.x) - len(self.coefs))  # Degrees of freedom
                t_value = stats.t.ppf(1 - alpha/2, dof)
                
                # For each point in x_interp, calculate the confidence interval
                y_err = []
                
                # This simplified version works for polynomials
                if self.fit_function is None:
                    X = np.vander(self.x_interp, len(self.coefs))
                    y_err = np.sqrt(np.sum((X @ self.cov_matrix) * X, axis=1))
                else:
                    # For non-polynomial fits, this is just an approximation
                    # A proper implementation would use numerical derivatives
                    y_err = np.ones_like(self.x_interp) * np.mean(np.sqrt(np.diag(self.cov_matrix)))
                
                ax.fill_between(
                    self.x_interp,
                    self.y_interp - t_value * y_err,
                    self.y_interp + t_value * y_err,
                    color='green', alpha=0.2,
                    label=f'{int(confidence_level*100)}% Confidence Interval'
                )
            except Exception as e:
                print(f"Could not generate confidence intervals: {str(e)}")
        
        # Add error metrics
        error_metrics = self.calculate_error()
        rmse = error_metrics['RMSE_noisy']
        r2 = error_metrics['R_squared']
        ax.set_title(f'Nonlinear Data Fitting (RMSE: {rmse:.4f}, R²: {r2:.4f})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Example 1: Default quadratic function
    print("\nExample 1: Default quadratic function")
    data_gen = NonlinearDataGenerator()
    data_gen.add_noise(noise_level=0.2, seed=42)
    data_gen.fit_data()
    data_gen.interpolate_results()
    metrics = data_gen.calculate_error()
    print(f"Error metrics: {metrics}")
    fig = data_gen.plot_results()
    
    # Example 2: Custom sine function
    print("\nExample 2: Custom sine function")
    sine_params = {'amplitude': 2, 'frequency': 0.5, 'phase': 1, 'offset': 3}
    sine_gen = NonlinearDataGenerator(
        true_function=NonlinearDataGenerator.sine_function,
        function_params=sine_params
    )
    sine_gen.add_noise(noise_level=0.3, noise_type='outliers', seed=42)
    sine_gen.fit_data(fit_function=NonlinearDataGenerator.sine_function, p0=sine_params, robust=True)
    sine_gen.interpolate_results()
    metrics = sine_gen.calculate_error()
    print(f"Error metrics: {metrics}")
    fig = sine_gen.plot_results(include_confidence=True)
    
    # Example 3: Exponential function
    print("\nExample 3: Exponential function")
    exp_params = {'a': 0.5, 'b': 0.3, 'c': 1}
    exp_gen = NonlinearDataGenerator(
        true_function=NonlinearDataGenerator.exponential_function,
        function_params=exp_params
    )
    exp_gen.add_noise(noise_level=0.2, noise_type='proportional')
    exp_gen.fit_data(fit_function=NonlinearDataGenerator.exponential_function, p0=exp_params)
    exp_gen.interpolate_results()
    metrics = exp_gen.calculate_error()
    print(f"Error metrics: {metrics}")
    fig = exp_gen.plot_results()
    
    plt.show()
