# anc-prob
Repository for the Ancestral Probability procedure

# usage

```python
# Load the data and make sure the data-types are correct
df = pd.read_csv("https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/boston-housing/data/boston-housing.continuous.txt", sep="\t")
df = df.astype({"RAD": "float64", "TAX": "float64"})
print(df.head())
print("instances:", len(df.index)
```

```
     CRIM    ZN  INDUS  CHAS    NOX     RM  ...  RAD    TAX  PTRATIO       B  LSTAT  MEDV
0  0.0063  18.0   2.31     0  0.538  6.575  ...  1.0  296.0     15.3  396.90   4.98  24.0
1  0.0273   0.0   7.07     0  0.469  6.421  ...  2.0  242.0     17.8  396.90   9.14  21.6
2  0.0273   0.0   7.07     0  0.469  7.185  ...  2.0  242.0     17.8  392.83   4.03  34.7
3  0.0324   0.0   2.18     0  0.458  6.998  ...  3.0  222.0     18.7  394.63   2.94  33.4
4  0.0691   0.0   2.18     0  0.458  7.147  ...  3.0  222.0     18.7  396.90   5.33  36.2

[5 rows x 14 columns]
instances: 507
```

```python
# Create the model and AP object
model = ap.LH(df)
anc_prob = ap.AP(model)

# Load knowledge
anc_prob.set_knowledge("knwl_ex.json")
```

```python

# Make directory for current run
current = str(time.time())
os.mkdir(current)

# Select a subset of the variable to analyze
selected = ["AGE", "CRIM", "MEDV", "TAX"]
anc_prob.set_selected(selected)
print("selected:", anc_prob.get_selected())

# Print MEC and MAG counts
num_mecs, num_mags = anc_prob.get_counts()
print("num mecs:", num_mecs)
print("num mags:", num_mags)
```

```
selected: ['AGE', 'CRIM', 'MEDV', 'TAX']
num mecs: 248
num mags: 2492
```

```python
# Run the procedure
plt_dir = "./" + current + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|
|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/1666660143.3428633/plots_single/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/1666660143.3428633/plots_single/CRIM_MEDV.png "CRIM_MEDV.png")|


```python

# Resample the procedure
plt_dir = "./" + current + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|
|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/1666660143.3428633/plots_resamp/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/1666660143.3428633/plots_resamp/CRIM_MEDV.png "CRIM_MEDV.png")|

```python
# Get the top 5 best PAGs
gdot = gviz.Graph(format='png', engine='neato')
plt_dir = "./" + current + "/best_pags"
os.mkdir(plt_dir)
anc_prob.get_best(top=5, gdot=gdot, plt_dir=plt_dir)
```

|First PAG|Second PAG|
|:-:|:-:|
|![First PAG](https://github.com/bja43/anc-prob/blob/main/1666660143.3428633/best_pags/pag_1.png "pag_1.png")|![Second PAG](https://github.com/bja43/anc-prob/blob/main/1666660143.3428633/best_pags/pag_2.png "pag_2.png")|


```python

# Make directory for current run
current = str(time.time())
os.mkdir(current)

# Select a subset of the variable to analyze
selected = ["AGE", "CHAS", "CRIM", "MEDV"]
anc_prob.set_selected(selected)
print("selected:", anc_prob.get_selected())

# Print MEC and MAG counts
num_mecs, num_mags = anc_prob.get_counts()
print("num mecs:", num_mecs)
print("num mags:", num_mags)
```

```
selected: ['AGE', 'CHAS', 'CRIM', 'MEDV']
num mecs: 163
num mags: 448
```

```python
# Run the procedure
plt_dir = "./" + current + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|
|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/1666660158.7818575/plots_single/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/1666660158.7818575/plots_single/CRIM_MEDV.png "CRIM_MEDV.png")|


```python

# Resample the procedure
plt_dir = "./" + current + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)
```

|AGE x CRIM|CRIM x MEDV|
|:-:|:-:|
|![AGE x CRIM](https://github.com/bja43/anc-prob/blob/main/1666660158.7818575/plots_resamp/AGE_CRIM.png "AGE_CRIM.png")|![CRIM x MEDV](https://github.com/bja43/anc-prob/blob/main/1666660158.7818575/plots_resamp/CRIM_MEDV.png "CRIM_MEDV.png")|

```python
# Get the top 5 best PAGs
gdot = gviz.Graph(format='png', engine='neato')
plt_dir = "./" + current + "/best_pags"
os.mkdir(plt_dir)
anc_prob.get_best(top=5, gdot=gdot, plt_dir=plt_dir)
```

|First PAG|Second PAG|
|:-:|:-:|
|![First PAG](https://github.com/bja43/anc-prob/blob/main/1666660158.7818575/best_pags/pag_1.png "pag_1.png")|![Second PAG](https://github.com/bja43/anc-prob/blob/main/1666660158.7818575/best_pags/pag_2.png "pag_2.png")|

