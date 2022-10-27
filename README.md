# anc-prob: Ancestral Probability (AP)

This repository contains code and an example for the Ancestral Probability (AP) procedure[^1]. The AP procedure performs local causal discovery (3-5 variables) in a Bayesian manner while accounting for the possibility of latent confounding. This is done by first calculating an approximation of the marginal likelihood for all MAG models[^2] with a modified version of the BIC score[^3][^4][^5] and then marginalizing over features of interest, such as if one variable is an ancestor (cause) of another.

## Dependencies

### Required

   * numpy
   * itertools
   * pandas
   * matplotlib
   * pickle
   * json
   * os

### Optional

   * graphviz

## Models

We approximate the marginal likelihood of MAG models[^2] with a modified version of the BIC score[^3][^4][^5]. This score requires a parametric model from a curved exponential family. This repository contains the following models:
1. Multivariate Gaussian (continuous):
   * ap.MG(*data*)
      * *data* = pandas.DataFrame
2. Multinomial (discrete):
   * ap.MN(*data*)
      * *data* = pandas.DataFrame
3. Lee and Hastie[^6][^7] (mixed):
   * ap.LH(*data*)
      * *data* = pandas.DataFrame

## Analysis

   * ap.AP(*model*)
      * *model* = ap.Model (ap.MG, ap.MN, or ap.LH)
   * set_knowledge(*filename*)
      * *filename* = path/filename (JSON knowledge file)
   * set_selected(*selected*)
      * *selected* = list (subset of variables to be analyzed)
   * get_counts()
   * get_best(*top=None*, *gdot=None*, *plt_dir=None*)
      * (optional) *top* = int (number of graphs to return)
      * (optional) *gdot* = graphviz.Graph (graphviz Graph object, if None no figures are produced)
      * (optional) *plt_dir* = path/directory (figure output directory, if None no figures are produced)
   * compute(*plt_dir=None*)
      * (optional) *plt_dir* = path/directory (figure output directory, if None no figures are produced)
   * resample(*reps*, *plt_dir=None*)
      * *reps* = int (number of resampling repetitions)
      * (optional) *plt_dir* = path/directory (figure output directory, if None no figures are produced)

## Knowledge

Knowledge is used to require or forbid various types of relationships between user defined groups of variables. Knowledge can be added to the analysis by JSON file. The JSON file should be constructed with two mandatory arrays:
   * **"sets"** is an array containing users defined group names for users defined groups of variables:
      * groups should not be named *"sets"* or *"rels"*;
      * groups can contain overlapping variables;
      * groups can be singletons.
   * **"rels"** is an array containing (required/forbidden) relationships between pairs of the user defined groups, e.g. for groups *A* and *B* (*A = B* is allowed):
      * ***A adj B*** requires that all variables in *A* are adjacent to all variables in *B*;
      * ***A !adj B*** forbids any variable in *A* from being adjacent to any variable in *B*;
      * ***A anc B*** requires that all variables in *A* are ancestors of all variables in *B*;
      * ***A !anc B*** forbids any variable in *A* from being an ancestor of any variable in *B*;
      * ***A uncf B*** forbids any varaible in *A* from being connected to any variable in *B* by a bi-directed edge[^8].

The two mandatory arrays should be followed by an array for each user defined group name where each array containing the variables belonging to the corresponding user defined group.

### Example

```
[
	{
		"sets": ["disc", "cont"],
		"rels": ["cont !anc disc", "cont uncf disc"],
		"disc": ["CHAS"],
		"cont": ["CRIM", "ZN", "INDUS", "NOX", 
			"RM", "AGE", "DIS", "RAD", "TAX", 
			"PTRATIO", "B", "LSTAT", "MEDV"]
	}
]
```

In the example above, we define two group names, *"disc"* and *"cont"*, which are intended to contain the discrete and continuous variables, respectively. After defining these group names, we specify two relationships that we wish to enforce in the analysis:
   * ***cont !anc disc*** forbids models where any variable in *"cont"* is an ancestor of a variable in *"disc"*;
   * ***cont uncf disc*** forbids models with dependence between one or more variables in *"cont"* and one or more variables in *"disc"* that can only be explained by latent confounding.

## Usage

We start by choosing a dataset to analyze. Here we investigate a dataset concerning housing values in suburbs of Boston which we load from a repository of datasets suitable for causal discovery analysis[^9][^10].

```python
import os
import ap

import pandas as pd
import graphviz as gviz

# Make/clean directory for example
dir_ex = "figs_ex"
os.system("rm -r -f " + dir_ex)
os.system("mkdir " + dir_ex)

# Load the data and make sure the data-types are correct
df = pd.read_csv("https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/boston-housing/data/boston-housing.continuous.txt", sep="\t")
df = df.astype({"RAD": "float64", "TAX": "float64"})
print(df.head())
print("instances:", len(df.index))
```

```
     CRIM    ZN  INDUS  CHAS    NOX     RM  ...  RAD    TAX  PTRATIO       B  LSTAT  MEDV
0  0.0063  18.0   2.31     0  0.538  6.575  ...  1.0  296.0     15.3  396.90   4.98  24.0
1  0.0273   0.0   7.07     0  0.469  6.421  ...  2.0  242.0     17.8  396.90   9.14  21.6
2  0.0273   0.0   7.07     0  0.469  7.185  ...  2.0  242.0     17.8  392.83   4.03  34.7
3  0.0324   0.0   2.18     0  0.458  6.998  ...  3.0  222.0     18.7  394.63   2.94  33.4
4  0.0691   0.0   2.18     0  0.458  7.147  ...  3.0  222.0     18.7  396.90   5.33  36.2

[5 rows x 14 columns]
instances: 506
```

Next, we choose a parametric model to fit to the data. Our dataset contains 13 continuous variables and one binary variable, so we choose the Lee and Hastie model because it can be applied to mixed datasets.

```python
# Create the model and AP object
model = ap.LH(df)
anc_prob = ap.AP(model)
```

Next, we choose a subset of variables to analyze. In this example we choose the following variables:
   * **AGE** is the proportion of owner-occupied units built prior to 1940;
   * **CHAS** is a indicator variable for the Charles River (1 if tract bounds river and 0 otherwise);
   * **CRIM** is the per capita crime rate by town;
   * **MEDV** is the median value of owner-occupied homes in $1,000's;
   * **TAX** is the full-value property-tax rate per $10,000.

These variables were selected somewhat at random for this demonstration. We encourage users to try selecting a different subset! If you do, please limit yourself to subsets of 5 variables or less.

```python
# Select a subset of the variable to analyze
selected = ["AGE", "CHAS", "CRIM", "MEDV", "TAX"]
anc_prob.set_selected(selected)
num_mecs, num_mags = anc_prob.get_counts()
print("selected:", anc_prob.get_selected())
print("num mecs:", num_mecs)
print("num mags:", num_mags)
```

```
selected: ['AGE', 'CHAS', 'CRIM', 'MEDV', 'TAX']
num mecs: 24259
num mags: 328924
```

Above, we see that we are currently considering 328,924 MAG models. We can narrow this down to a more relevant set of models by incorporating knowledge.

```python
# Load knowledge
anc_prob.set_knowledge("knwl_ex.json")
num_mecs, num_mags = anc_prob.get_counts()
print("selected:", anc_prob.get_selected())
print("num mecs:", num_mecs)
print("num mags:", num_mags)
```

```
selected: ['AGE', 'CHAS', 'CRIM', 'MEDV', 'TAX']
num mecs: 9896
num mags: 39872
```

Now we are ready to analyze our data. We start by visualizing the most probable Markov equivalence classes (MECs) of MAG models, which are illustrated using PAGs[^2].

```python
# Get the top 6 PAGs
gdot = gviz.Graph(format='png', engine='neato')
plt_dir = "./" + dir_ex + "/best_pags"
os.mkdir(plt_dir)
anc_prob.get_best(top=6, gdot=gdot, plt_dir=plt_dir)
```

|First PAG|Second PAG|Third PAG|
|:-:|:-:|:-:|
|![First PAG](https://github.com/bja43/anc-prob/blob/main/figs_ex/best_pags/pag_1.png "pag_1.png")|![Second PAG](https://github.com/bja43/anc-prob/blob/main/figs_ex/best_pags/pag_2.png "pag_2.png")|![Third PAG](https://github.com/bja43/anc-prob/blob/main/figs_ex/best_pags/pag_3.png "pag_3.png")|

|Fourth PAG|Fifth PAG|Sixth PAG|
|:-:|:-:|:-:|
|![Fourth PAG](https://github.com/bja43/anc-prob/blob/main/figs_ex/best_pags/pag_4.png "pag_4.png")|![Fifth PAG](https://github.com/bja43/anc-prob/blob/main/figs_ex/best_pags/pag_5.png "pag_5.png")|![Sixth PAG](https://github.com/bja43/anc-prob/blob/main/figs_ex/best_pags/pag_6.png "pag_6.png")|

Given the most probable MECs/PAGs, it might be interesting to investigate the possible causes of **CRIM** (per capita crime rate by town). In particular, what are the probabilities that the following variables cause **CRIM**:
   * **AGE** (proportion of owner-occupied units built prior to 1940);
   * **MEDV** (median value of owner-occupied homes in $1,000's);
   * **TAX** (full-value property-tax rate per $10,000).

```python
# Run the procedure
plt_dir = "./" + dir_ex + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|CRIM x TAX|
|:-:|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_single/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_single/CRIM_MEDV.png "CRIM_MEDV.png")|![CRIM x TAX](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_single/CRIM_TAX.png "CRIM_TAX.png")|

We can gain confidence in the robustivity of our results using resampling techniques. In this case, we bootstrap the dataset 100 times.

```python
# Resample the procedure
plt_dir = "./" + dir_ex + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|CRIM x TAX|
|:-:|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_resamp/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_resamp/CRIM_MEDV.png "CRIM_MEDV.png")|![CRIM x TAX](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_resamp/CRIM_TAX.png "CRIM_TAX.png")|

Our impromptu analysis has bore fruit! According to the AP procedure, it is highly likely that **MEDV** causes **CRIM** and somewhat likely that **TAX** causes **CRIM**. 

---

[^1]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4240385
[^2]: https://projecteuclid.org/journals/annals-of-statistics/volume-30/issue-4/Ancestral-graph-Markov-models/10.1214/aos/1031689015.full
[^3]: https://www.jstor.org/stable/2958889
[^4]: https://www.jstor.org/stable/2241441
[^5]: https://arxiv.org/pdf/2207.08963.pdf
[^6]: http://proceedings.mlr.press/v31/lee13a.pdf
[^7]: http://proceedings.mlr.press/v104/andrews19a/andrews19a.pdf
[^8]: This forbids models with dependence between one or more variables in *A* and one or more variables in *B* that can only be explained by latent confounding.
[^9]: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
[^10]: https://github.com/cmu-phil/example-causal-datasets/