# anc-prob: Ancestral Probability (AP)
This repository contains code and examples for using the Ancestral Probability procedure[^1]. The AP procedure performed local causal discovery (3-5 variables) in a Bayesian manner while allowing for latent confounding.

## Dependencies



## Models

We approximate the marginal likelihood of the models using the BIC[^2][^3].

1. Multivariate Gaussian:
   * for continuous datasets
   * ap.MG(*data*)
      * *data* = DataFrame
2. Multinomial:
   * for discrete datasets
   * ap.MN(*data*)
      * *data* = DataFrame
3. Lee and Hastie[^4][^5]
   * for mixed datasets
   * ap.LH(*data*)
      * *data* = DataFrame

## Knowledge

Knowledge is used to require or forbid various types of relationships between user defined groups of variables. Knowledge can be added to the analysis by JSON file. The JSON file should be constructed with to manidtory arrays:
   * **"sets"** is an array containing users defined group names for users defined groups of variables:
      * groups can contain overlapping variables;
      * groups can be singletons.
   * **"rels"** is an array containing (required/forbidden) relationships between pairs of the user defined groups, e.g. for groups *A* and *B* (*A = B* is allowed):
      * ***A adj B*** requires that all variables in *A* are adjacent to all variables in *B*;
      * ***A !adj B*** forbids any variable in *A* from being adjacent to any variable in *B*;
      * ***A anc B*** requires that all variables in *A* are ancestors of all variables in *B*;
      * ***A !anc B*** forbids any variable in *A* from being an ancestor of any variable in *B*;
      * ***A uncf B*** forbids any varaible in *A* from being connected to any variable in *B* by a bi-directed edge[^6].

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

In the example above, we have defined group names **"disc"** and **"cont"** which are intended to contain the discrete and continuous variables, respectively. After defining these group names, we specify two relationships that we wish to enforce in the analysis:
   * ***cont !anc disc*** forbids models where any variable in *cont* is an ancestor of a variable in *disc*;
   * ***cont uncf disc*** forbids models with dependence between one or more variables in *cont* and one or more variables in *disc* that can only be explained by latent confounding.

## Analysis

ap.AP(model)
   * set_knowledge(filename)
   * set_selected(selected)
   * get_counts()
   * get_best(top, gdot, plt_dir)
   * compute(plt_dir, rsmp, frac, n, rplc)
   * resample(reps, plt_dir, frac, n, rplc)

## Usage

We start by choosing a dataset to analyze. Here we investigate a dataset concerning housing values in suburbs of Boston which we pull from a repository of datasets suitable for causal discovery analysis[^7][^8].

```python
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

Next, we choose a parameteric model class to fit to the data. Our dataset has 13 continuous variables and one binary variable, so we choose the Lee and Hastie model class because it can be applied to mixed datasets.

```python
# Create the model and AP object
model = ap.LH(df)
anc_prob = ap.AP(model)
```

Next, we choose a subset of the variables to analyze. In this example we choose the follow variables:
   * **AGE** is the proportion of owner-occupied units built prior to 1940;
   * **CHAS** is a indicator variable for the Charles River (1 if tract bounds river and 0 otherwise);
   * **CRIM** is the per capita crime rate by town;
   * **MEDV** is the median value of owner-occupied homes in $1,000's;
   * **TAX** is the full-value property-tax rate per $10,000.

These variables where selected somewhat at random for this demostration. We encourage uses to try selecting a different subset! Please limit yourself to subsets of 5 variables or less.

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

Above, we see that there are currently 328,924 causal models that we are currently considering. We can narrow this set down to a more relevant set by incorporating knowledge.

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

Now we are ready to analyze our data.

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

Of interest are:
   * CRIM
   * CRIM

```python
# Run the procedure
plt_dir = "./" + current + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|
|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_single/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_single/CRIM_MEDV.png "CRIM_MEDV.png")|

We can gain confidence in the robustivity of our results using resampling techniques.

```python
# Resample the procedure
plt_dir = "./" + current + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|
|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_resamp/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/figs_ex/plots_resamp/CRIM_MEDV.png "CRIM_MEDV.png")|

---

[^1]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4240385
[^2]: https://www.jstor.org/stable/2958889
[^3]: https://www.jstor.org/stable/2241441
[^4]: http://proceedings.mlr.press/v31/lee13a.pdf
[^5]: http://proceedings.mlr.press/v104/andrews19a/andrews19a.pdf
[^6]: This forbids models with dependence between one or more variables in *A* and one or more variables in *B* that can only be explained by latent confounding.
[^7]: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
[^8]: https://github.com/cmu-phil/example-causal-datasets/