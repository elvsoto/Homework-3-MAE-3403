import math


def gamma_function(alpha):
    """
    Calculate the value of the gamma function for a given alpha using Simpson's rule for numerical integration.
    """
    def integrand(t):
        return math.exp(-t) * t ** (alpha - 1)

    return simpsons_rule(integrand, 0, math.inf, 1000)


def simpsons_rule(func, a, b, n):
    """
    Numerical integration using Simpson's rule.
    """
    # Calculate the width of each subinterval
    h = (b - a) / n
    # Initialize the result with the values at the boundaries
    result = func(a) + func(b)

    # Iterate over odd-indexed intervals and add 4 times the function value at their midpoints
    for i in range(1, n, 2):
        result += 4 * func(a + i * h)

    # Iterate over even-indexed intervals (excluding the last one) and add 2 times the function value at their midpoints
    for i in range(2, n - 1, 2):
        result += 2 * func(a + i * h)

    # Multiply the sum by h/3 as per Simpson's rule
    return result * h / 3.0


def t_distribution_probability(upper_limit, dof):
    """
    Calculate the probability of a given upper limit of integration under the t-distribution with specified degrees of freedom
    using Simpson's rule for numerical integration.
    """
    # Alias for degrees of freedom
    m = dof
    # Constant multiplier
    km = gamma_function(0.5 * m + 0.5) / (math.sqrt(math.pi * m) * gamma_function(0.5 * m))

    def integrand(u):
        return (1 + (u ** 2 / m)) ** (- (m + 1))

    # Return the product of the constant multiplier and the integral using Simpson's rule
    return km * simpsons_rule(integrand, -math.inf, upper_limit, 1000)


def main():
    # Prompt user for degrees of freedom and upper limit of integration
    degrees_of_freedom = int(input("Enter degrees of freedom (m): "))
    upper_limit = float(input("Enter upper limit of integration for F(z): "))

    # Calculate probability using t-distribution
    probability = t_distribution_probability(upper_limit, degrees_of_freedom)

    # Print the calculated probability
    print("Probability:", probability)


if __name__ == "__main__":
    main()
