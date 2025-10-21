---
created: 2025-01-15
tags: [statistics, mixed-models, REML, hierarchical-models]
status: 🌿 growing
---

# Mixed Effects Models

## What is this?

A **mixed effects model** contains both fixed effects (parameters we want to estimate) and random effects (parameters whose variation we want to model). These models are essential for hierarchical or grouped data.

## The Basic Structure

$$y_{ij} = \mu + b_j + \epsilon_{ij}$$

Where:
- $y_{ij}$ = observation $i$ in group $j$
- $\mu$ = overall mean (fixed effect)
- $b_j \sim N(0, \sigma^2_b)$ = group-specific effect (random effect)
- $\epsilon_{ij} \sim N(0, \sigma^2)$ = individual-level residual

**Key insight:** We don't care about the specific values of $b_j$, but we do care about $\sigma^2_b$ (how much groups vary).
## Intuition: Modelling Mean vs. Variance

### The Parallel Structure

In regression, we model the mean:
$E[\mathbf{y}] = X\beta$

In mixed models, we **also** model the variance:
$\text{Var}(\mathbf{y}) = V = \sigma^2_b ZZ^T + \sigma^2 I$

**The key insight:** Just as $X\beta$ decomposes the mean into components explained by different covariates, random effects decompose the variance into components at different levels of hierarchy.

### What This Means

**Fixed effects explain the mean:**
- "Students in urban schools score 5 points higher **on average**"
- Systematic differences in expected values
- We estimate $\beta$ to explain patterns in means

**Random effects explain the variance:**
- "Schools vary with **standard deviation = 10 points**"
- How much variation exists at different levels
- We estimate $\sigma^2_b$ and $\sigma^2$ to explain patterns in variability

### Why This Matters

By decomposing variance, we can answer:

1. **How much variation is between groups vs. within groups?**
   - Intraclass correlation: $\text{ICC} = \frac{\sigma^2_b}{\sigma^2_b + \sigma^2}$

2. **Are groups really different?**
   - If $\sigma^2_b \approx 0$, groups don't meaningfully differ
   - If $\sigma^2_b$ is large, group membership matters!

3. **How correlated are observations in the same group?**
   - Observations in the same group share the group effect $b_j$
   - This creates correlation

## The Basic Structure

$y_{ij} = \underbrace{\mu}_{\substack{\text{mean structure} \\ \text{(fixed effects)}}} + \underbrace{b_j + \epsilon_{ij}}_{\substack{\text{variance structure} \\ \text{(random effects)}}}$

Where:
- $y_{ij}$ = observation $i$ in group $j$
- $\mu$ = overall mean (fixed effect) - explains where the mean is
- $b_j \sim N(0, \sigma^2_b)$ = group-specific effect (random effect) - explains between-group variation
- $\epsilon_{ij} \sim N(0, \sigma^2)$ = individual-level residual - explains within-group variation

**Key insight:** We don't care about the specific values of $b_j$, but we do care about $\sigma^2_b$ (how much groups vary). The random effects let us decompose total variance into interpretable components.

## Example: Students in Schools

Students nested within schools provide a classic example:
- Students in the same school are more similar than students in different schools
- Each school has its own "effect" on test scores
- We want to quantify: How much variation is between schools vs. within schools?

### Generating Example Data

```r
# Setup
set.seed(123)
n_schools <- 3
n_students_per_school <- 4

# True parameters
true_mean <- 50           # Overall average score
true_school_sd <- 10      # Between-school variation
true_student_sd <- 15     # Within-school variation

# Generate school effects: b_j ~ N(0, σ²_school)
school_effects <- rnorm(n_schools, mean = 0, sd = true_school_sd)
print("School effects:")
print(school_effects)

# Generate student data
school_id <- rep(1:n_schools, each = n_students_per_school)

student_scores <- true_mean +                          # Overall mean
                  school_effects[school_id] +          # School effect
                  rnorm(12, mean = 0, sd = true_student_sd)  # Student variation

data <- data.frame(
  school = school_id,
  score = student_scores
)

print(data)
```

## Variance Components

The model has **two sources of variation**:

### 1. Between-School Variance ($\sigma^2_b$)
How much do schools differ from each other?

### 2. Within-School Variance ($\sigma^2$)
How much do students within the same school differ?

### Key Relationship

The variance of school means contains both sources:

$$\text{Var}(\bar{y}_j) = \sigma^2_b + \frac{\sigma^2}{n}$$

Where $n$ is the number of students per school.

**Intuition:**
- With 1 student per school: $\text{Var}(\bar{y}_j) = \sigma^2_b + \sigma^2$ (can't separate!)
- With 100 students per school: $\text{Var}(\bar{y}_j) \approx \sigma^2_b$ (student noise averages out)

```r
# Calculate school means
school_means <- tapply(data$score, data$school, mean)
print("School means:")
print(school_means)

# Variance of school means
print("Variance of school means:")
print(var(school_means))

# Expected variance of school means
expected_var <- true_school_sd^2 + (true_student_sd^2 / n_students_per_school)
print("Expected variance:")
print(expected_var)
```

## Estimation: ML vs REML

### Maximum Likelihood (ML)
Estimates variance by treating the overall mean as if it were known:

$$\hat{\sigma}^2_b \text{ (ML)} = \text{variance of school means} - \frac{\text{within variance}}{n}$$

Uses $n$ in the denominator when calculating variance.

### Restricted Maximum Likelihood (REML)
Accounts for having to estimate the overall mean from data:

$$\hat{\sigma}^2_b \text{ (REML)} = \text{same calculation but with } (n-1) \text{ degrees of freedom}$$

**Why REML is better:** Estimating $\mu$ "uses up" one degree of freedom, so we should divide by $(n-1)$ not $n$.

### Manual Calculation

```r
# Overall mean
overall_mean <- mean(data$score)

# Within-school variance
within_school_deviations <- data$score - school_means[data$school]
within_school_variance <- sum(within_school_deviations^2) / (length(data$score) - n_schools)

# ML approach: divide by n
variance_of_means_ML <- sum((school_means - overall_mean)^2) / n_schools
between_school_var_ML <- variance_of_means_ML - (within_school_variance / n_students_per_school)

# REML approach: divide by (n-1)
variance_of_means_REML <- sum((school_means - overall_mean)^2) / (n_schools - 1)
between_school_var_REML <- variance_of_means_REML - (within_school_variance / n_students_per_school)

print("=== COMPARISON ===")
print(paste("True between-school variance:", true_school_sd^2))
print(paste("ML estimate:                 ", round(between_school_var_ML, 2)))
print(paste("REML estimate:               ", round(between_school_var_REML, 2)))
```

## REML from Scratch

We can implement REML by directly optimizing the REML likelihood function.

### Step 1: Build Covariance Matrix

For mixed models, the covariance matrix $V$ has block structure:

```r
build_V_matrix <- function(sigma2_between, sigma2_within, school_id, n_students_per_school) {
  n <- length(school_id)
  V <- matrix(0, n, n)
  
  for (i in 1:n) {
    for (j in 1:n) {
      if (school_id[i] == school_id[j]) {
        if (i == j) {
          V[i, j] <- sigma2_between + sigma2_within  # Diagonal
        } else {
          V[i, j] <- sigma2_between  # Same school, different students
        }
      }
      # Different schools: V[i,j] = 0
    }
  }
  
  return(V)
}
```

**Structure:**
- **Diagonal:** $\sigma^2_b + \sigma^2$ (total variance of one observation)
- **Within-school blocks:** $\sigma^2_b$ (students in same school are correlated)
- **Between-school blocks:** $0$ (students in different schools are independent)

### Step 2: REML Objective Function

#### Deriving the REML Likelihood

Let's build up the REML objective function step by step.

**Starting point - the model in matrix form:**

$\mathbf{y} = X\beta + Zb + \epsilon$

Where:
- $\mathbf{y}$ = vector of all observations
- $X\beta$ = fixed effects (overall mean $\mu$ in our case)
- $Zb$ = random effects ($b_j$ for each school)
- $\epsilon$ = residuals

**Distribution of y:**

Since $b \sim N(0, \sigma^2_b I)$ and $\epsilon \sim N(0, \sigma^2 I)$:

$\mathbf{y} \sim N(X\beta, V)$

Where the covariance matrix is:
$V = \text{Cov}(Zb + \epsilon) = \sigma^2_b ZZ^T + \sigma^2 I$

**Maximum Likelihood:**

For multivariate normal, the log-likelihood is:

$\log L_{ML}(\beta, \sigma^2_b, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log|V| - \frac{1}{2}(\mathbf{y} - X\beta)^T V^{-1}(\mathbf{y} - X\beta)$

**The Problem:** ML treats $\beta$ (the mean) as if we know it, but we estimate it from the data!

**REML Solution:**

REML transforms the data to remove dependence on $\beta$. Find a matrix $K$ such that:
- $K\mathbf{y}$ doesn't depend on $\beta$
- This requires $KX = 0$

Alternatively (and equivalently), REML "integrates out" the nuisance parameter $\beta$:

$L_{REML}(\sigma^2_b, \sigma^2) = \int L_{ML}(\beta, \sigma^2_b, \sigma^2) d\beta$

**After integration** (skipping the calculus), the REML log-likelihood is:

$\log L_{REML} = -\frac{n-p}{2}\log(2\pi) - \frac{1}{2}\log|V| - \frac{1}{2}\log|X^T V^{-1} X| - \frac{1}{2}(\mathbf{y} - X\hat{\beta})^T V^{-1}(\mathbf{y} - X\hat{\beta})$

Where $p$ is the number of fixed effects (just 1 for us - the intercept).

**Negative log-likelihood** (what we minimize):

$-2\log L_{REML} = \log|V| + \log|X^T V^{-1} X| + (\mathbf{y} - X\hat{\beta})^T V^{-1}(\mathbf{y} - X\hat{\beta}) + \text{constants}$

We drop constants that don't depend on the parameters.

**The three terms:**

1. $\log|V|$ - Penalizes large variances
2. $\log|X^T V^{-1} X|$ - **The REML correction** (accounts for estimating $\beta$)
3. $(\mathbf{y} - X\hat{\beta})^T V^{-1}(\mathbf{y} - X\hat{\beta})$ - Weighted residual sum of squares

**Key insight:** The extra term $\log|X^T V^{-1} X|$ (compared to ML) automatically gives us the degrees of freedom correction we need!

#### Implementation

```r
reml_objective <- function(params, y, X, school_id, n_students_per_school) {
  # Use exp() to ensure positive variances
  sigma2_between <- exp(params[1])
  sigma2_within <- exp(params[2])
  
  # Build covariance matrix
  V <- build_V_matrix(sigma2_between, sigma2_within, school_id, n_students_per_school)
  V_inv <- solve(V)
  
  # Estimate beta (mean)
  XtVinvX <- t(X) %*% V_inv %*% X
  XtVinvy <- t(X) %*% V_inv %*% y
  beta_hat <- solve(XtVinvX, XtVinvy)
  
  # Calculate residuals
  residuals <- y - X %*% beta_hat
  
  # Three terms of REML
  term1 <- determinant(V, logarithm = TRUE)$modulus[1]  # log|V|
  term2 <- determinant(XtVinvX, logarithm = TRUE)$modulus[1]  # log|X^T V^{-1} X|
  term3 <- t(residuals) %*% V_inv %*% residuals  # Residual sum of squares
  
  return(as.numeric(term1 + term2 + term3))
}
```

**Three components:**
1. $\log|V|$ - Penalizes large variances
2. $\log|X^T V^{-1} X|$ - **REML correction** (accounts for estimating fixed effects)
3. $(y - X\hat{\beta})^T V^{-1}(y - X\hat{\beta})$ - Model fit

The key difference from ML is **term 2** - this automatically gives us the degrees of freedom correction!

### Step 3: Optimize

```r
# Design matrix (column of 1's for intercept)
X <- matrix(1, nrow = length(school_id), ncol = 1)

# Starting values (on log scale)
start_params <- c(log(50), log(200))

# Optimize
result <- optim(
  par = start_params,
  fn = reml_objective,
  y = data$score,
  X = X,
  school_id = school_id,
  n_students_per_school = n_students_per_school,
  method = "BFGS"
)

# Extract estimates
sigma2_between_reml <- exp(result$par[1])
sigma2_within_reml <- exp(result$par[2])

print("From-scratch REML estimates:")
print(paste("Between-school variance:", round(sigma2_between_reml, 2)))
print(paste("Within-school variance: ", round(sigma2_within_reml, 2)))
```

### Step 4: Compare with lme4

```r
library(lme4)

model_reml <- lmer(score ~ 1 + (1|school), data = data, REML = TRUE)
print(summary(model_reml))

# Extract variances
lmer_vars <- as.data.frame(VarCorr(model_reml))
print("lmer() estimates match our from-scratch implementation!")
```

## When to Use Mixed Models

**Use mixed effects models when:**
- Data has natural grouping/hierarchy (students in schools, patients in hospitals)
- Observations within groups are correlated
- You want to account for group-level variation
- You want to generalize to new groups

**Examples:**
- Longitudinal data (repeated measures on same subjects)
- Spatial data (observations in geographic regions)
- Multi-site clinical trials
- Educational studies (students nested in classrooms)

## Key Takeaways

1. **Mixed models** separate variation into components (between-group vs. within-group)
2. **REML is better than ML** for estimating variance components (less biased)
3. **REML accounts for degrees of freedom** used to estimate fixed effects
4. The correction is automatic when using the REML likelihood
5. With few groups, the ML vs REML difference can be substantial

## Connected to

- [[reml]] - General REML theory
- [[hierarchical_models]] - Broader class of models
- [[variance_components]] - What we're estimating
- [[penalized_regression]] - Related framework using REML for smoothing

## Further Exploration

- How does this extend to multiple random effects?
- What about crossed random effects (not nested)?
- How to handle unbalanced data (different group sizes)?
- Connection to Bayesian hierarchical models?

## References

- Laird, N.M. and Ware, J.H. (1982). Random-effects models for longitudinal data. Biometrics, 38(4), 963-974.
- Pinheiro, J.C. and Bates, D.M. (2000). Mixed-Effects Models in S and S-PLUS. Springer.
- Bates, D., Mächler, M., Bolker, B., and Walker, S. (2015). Fitting Linear Mixed-Effects Models Using lme4. Journal of Statistical Software, 67(1), 1-48.
