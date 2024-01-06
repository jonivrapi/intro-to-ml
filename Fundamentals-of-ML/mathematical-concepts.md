---
order: 98
label: Math Concepts
---

# A (not) exhaustive collection of the mathematical concepts related to ML

### Mean (Average)

The mean is the sum of all values divided by the number of values.

$$
    \bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

Where $N$ is the number of observations and $x_i$ is each individual observation.

### Median

The median is the middle value in a data set when the values are arranged in ascending or descending order.

### Mode

The mode is the most frequently occurring value in a data set.

### Standard Deviation

Standard deviation measures the amount of variation or dispersion in a set of values.

$$
    \sigma = \sqrt{\frac{1}{N} \sum\_{i=1}^{N} (x_i - \mu)^2}
$$

Where $\mu$ is the mean.

### Normal Distribution

The normal distribution is also known as the Gaussian distribution.

$$
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{ -\frac{(x-\mu)^2}{2\sigma^2} }
$$

Where $\mu$ is the mean and $\sigma^2$ is the variance.

### Binomial Distribution

The binomial distribution represents the number of successes in a sequence of independent experiments.

$$
    P(k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

Where $n$ is the number of trials, $p$ is the probability of success, and $k$ is the number of successes.

### Pearson Correlation

Correlation measures the strength and direction of a linear relationship between two variables.

$$
    r_{xy} = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y}) }{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2 \sum_{i=1}^{N} (y_i - \bar{y})^2}}
$$

Where $\bar{x}$ and $\bar{y}$ are the means of the two variables. A score of $1$ would indicate a perfect correlation and that high values of $x$ correspond with high values of $y$. A score or $-1$ would indicate a perfect inverse correlation and that high values of $x$ correspond with low values of $y$. A score of $0$ would indicate no correlation and that a change in $x$ does not correspond with a change in $y$.

### Covariance

Covariance indicates the direction of the linear relationship between variables.

For a **population**: $$ \text{Cov}(x, y) = \frac{1}{N} \sum\_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y}) $$

For a **sample**: $$ \text{Cov}(x, y) = \frac{1}{N-1} \sum\_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y}) $$

Where $\bar{x}$ and $\bar{y}$ are the means of the variables $x$ and $y$, respectively.

### Z-test

Used when the data follows a normal distribution, and the population variance is known.

$$
    Z = \frac{\bar{x} - \mu}{\frac{\sigma}{\sqrt{n}}}
$$

Where $\bar{x}$ is the sample mean, $\mu$ is the population mean, $\sigma$ is the population standard deviation, and $n$ is the sample size.

### 1-Sample T-test

Used when the population variance is unknown.

$$
    t = \frac{\bar{x} - \mu}{\frac{\sigma}{\sqrt{n}}}
$$

Where $\bar{x}$ is the sample mean, $\mu$ is the population mean, $\sigma$ is the population standard deviation, and $n$ is the sample size.

### Linear Regression

In linear regression, we model the relationship between two variables by fitting a linear equation to observed data. The formula for a simple linear regression is:

$$
    Y_i = \beta_0 + \beta_1 X_i + \epsilon
$$

Where $Y$ is the dependent variable, $X$ is the independent variable, $\beta_0$ is the y-intercept, $\beta_1$ is the slope of the line, and $\epsilon$ is the error term.

### Multiple Regression

Multiple regression is an extension of linear regression into a relationship with more than one independent variable:

$$
    Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon
$$

### Bayes' Theorem

Bayes' Theorem is fundamental to Bayesian statistics:

$$
    P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}
$$

Where $P(A|B)$ is the probability of $A$ given $B$, and $P(B|A)$ is the probability of $B$ given $A$.

### Entropy

Entropy is a measure of randomness or uncertainty.

$$
    H(X) = - \sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
$$

Where $H(X)$ is the entropy of a random variable $X$ and $P(x_i)$ is the probability of each outcome.
