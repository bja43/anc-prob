
import os
import time
import ap

import graphviz as gviz
import pandas as pd
import numpy as np



# Load the data and make sure the data-types are correct
df = pd.read_csv("https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/main/real/boston-housing/data/boston-housing.continuous.txt", sep="\t")
df = df.astype({"RAD": "float64", "TAX": "float64"})
print(df.head())
print("instances:", len(df.index))

# Create the model and AP object
model = ap.MG(df)
# model = ap.LH(df)
anc_prob = ap.AP(model)

# Load knowledge
anc_prob.set_knowledge("knwl_ex.json")



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

# Run the procedure
plt_dir = "./" + current + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)

# Resample the procedure
plt_dir = "./" + current + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)

# Get the top 5 best PAGs
gdot = gviz.Graph(format='png', engine='neato')
plt_dir = "./" + current + "/best_pags"
os.mkdir(plt_dir)
anc_prob.get_best(top=5, gdot=gdot, plt_dir=plt_dir)



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

# Run the procedure
plt_dir = "./" + current + "/plots_single"
os.mkdir(plt_dir)
anc_prob.compute(plt_dir=plt_dir)

# Resample the procedure
plt_dir = "./" + current + "/plots_resamp"
os.mkdir(plt_dir)
anc_prob.resample(reps=100, plt_dir=plt_dir)

# Get the top 5 best PAGs
gdot = gviz.Graph(format='png', engine='neato')
plt_dir = "./" + current + "/best_pags"
os.mkdir(plt_dir)
anc_prob.get_best(top=5, gdot=gdot, plt_dir=plt_dir)