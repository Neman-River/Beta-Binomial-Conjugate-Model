
data {
  int<lower=0> N; // number of trials
  int<lower=0> y; // number of successes
  real<lower=0> alpha; // prior hyperparameter
  real<lower=0> beta; // prior hyperparameter
}
parameters {
  real<lower=0, upper=1> theta; // probability of success
}
model {
  theta ~ beta(alpha, beta); // prior
  y ~ binomial(N, theta); // likelihood

}
generated quantities {
  int y_pred;
  y_pred = binomial_rng(N, theta); // posterior predictive distribution
}
