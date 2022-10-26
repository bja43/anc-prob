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

# Create the model and AP object
model = ap.LH(df)
anc_prob = ap.AP(model)

# Select a subset of the variable to analyze
selected = ["AGE", "CHAS", "CRIM", "MEDV", "TAX"]
anc_prob.set_selected(selected)
num_mecs, num_mags = anc_prob.get_counts()
print("selected:", anc_prob.get_selected())
print("num mecs:", num_mecs)
print("num mags:", num_mags)

# Load knowledge
anc_prob.set_knowledge("knwl_ex.json")
num_mecs, num_mags = anc_prob.get_counts()
print("selected:", anc_prob.get_selected())
print("num mecs:", num_mecs)
print("num mags:", num_mags)

# Get the top 6 PAGs
gdot = gviz.Graph(format='png', engine='neato')
plt_dir = "./" + dir_ex + "/best_pags"
os.mkdir(plt_dir)
anc_prob.get_best(top=6, gdot=gdot, plt_dir=plt_dir)

# Run the procedure
plt_dir = "./" + dir_ex + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)

# Resample the procedure
plt_dir = "./" + dir_ex + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)
