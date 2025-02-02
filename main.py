import numpy as np
from scipy import optimize

class NonlinearDataGenerator:
    """
    A class for generating and fitting nonlinear data with different errors.
    It implements methods for adding varying levels of Gaussian noise, 
    performing curve fitting, and interpolating the results.
    """
    def __init__(self, polynomial_order=2):
        self.polynomial_order = polynomial_order
        self.x = np.linspace(0, 10, 100)
        self.y_true = self.true_function(self.x)

    @staticmethod
    def true_function(x, a=3, b=5, c=7):
        """
        The underlying nonlinear function used to generate the true data.

        Parameters
        ----------
        x : np.ndarray, shape (N, )
            The independent variable values to evaluate at.
        a : float, optional
            Coefficient for the quadratic term. Default is 3.
        b : float, optional
            Coefficient for the linear term. Default is 5.
        c : float, optional 
            The intercept. Default is 7.

        Returns
        -------
        y_true : np.ndarray, shape (N, )
            The true y values corresponding to each x value.
        """
        return a * x ** 2 + b * x + c

    def add_noise(self, noise_level=0.1):
        """
        Adds varying levels of Gaussian noise to the true data.

        Parameters
        ----------
        noise_level : float, optional
            The standard deviation of the Gaussian noise to add. Default is 0.1.

        Returns
        -------
        None
        """
        self.y_noisy = self.y_true + np.random.normal(0, noise_level, size=len(self.x))

    def fit_data(self):
        """
        Fits the noisy data using polynomial regression.

        Parameters
        ----------
        None

        Returns
        -------
        self.coefs : np.ndarray, shape (p + 1, )
            The coefficients of the fitted polynomial.
        """
        def model(x, *coefs):
            return sum(coefs[i] * x**i for i in range(len(coefs)))

        self.coefs = optimize.curve_fit(model, self.x, self.y_noisy, p0=np.ones(self.polynomial_order + 1))[0]

    def interpolate_results(self):
        """
        Interpolates the fitted results at new x values.

        Parameters
        ----------
        None

        Returns
        -------
        self.y_interp : np.ndarray, shape (N, )
            The y values corresponding to each x value from the interpolation.
        """
        self.x_interp = np.linspace(0, 10, 500)
        self.y_interp = np.polyval(self.coefs[::-1], self.x_interp)

# Initialize the class with a polynomial order of 2
data_gen = NonlinearDataGenerator(polynomial_order=2)

# Generate the true data
data_gen.add_noise()

# Fit the noisy data using polynomial regression
data_gen.fit_data()

# Interpolate the fitted results at new x values
data_gen.interpolate_results()
