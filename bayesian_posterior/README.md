# Bayesian Posterior

In this problem we want to find the distribution of posteriors given the point predictions and outcomes of a
regression Machine Learning model.

We are looking into this problem so that we can use the distribution of posteriors in order to port
[performance estimation](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html) on regression
problems. Currently [CBPE algorithm](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html)
only works at classification problems.

We looked into many methods of getting a distribution of posteriors from point predictions and outcomes. However in
this case we are focusing on the Bayesian approach. We think our approach is sound on the high level but has some
implementation issues that need resolving before it can give descent results.

## Setting Up

We assume you have installed `conda` in the way most appropriate for your system.
The following commands will create an appropriate environment.

```
$ conda create -c conda-forge -n dev1 "pymc>=4" python=3.9
$ conda activate dev1
$ conda install -c conda-forge jupyterlab scikit-learn python-graphviz

```

## Running the Notebooks

- First we run `NB0 Create Dataset.ipynb` to create the dataset we will use.<br>
  The dataset has 4 numerical features, and a target that is a linear combination of those features. We have also added
  heteroscedastic gaussian noise with standard deviation being the square of feature4.
  We also train a `GradientBoostingRegressor` on those target values.
- Then we run `NB1 PYMC Estimate Posterior.ipynb` where we are trying to predict the distribution of residuals of what our
  model predicts versus the actual target. We are using `pymc` to do that. To evaluate the learnt distribution of residuals
  we are using [mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Evaluating results

- The results are unstable, I am probably not using a proper way to fix the seed and make the results reproducible.
  However re-running the notebook, may not yield the same (or similar enough) results as a previous run.
- The estimated error from the learnt distribution of residuals is not close to the actual error.

## Potential remedies?

- Problem can be formulated in a better way?
- `pymc` can be used in a better way?
