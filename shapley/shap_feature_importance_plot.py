import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
import cPickle as cPickle
import sklearn
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection #for splitting data
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as st
from xgboost import plot_tree
from graphviz import Graph
import shap
#note: not all imported packages are used

modelinput = sys.argv[1] #.cPickle model file
traininginput = sys.argv[2] #.tsv.gz feature matrix file

with open("directory/Models/%s" %modelinput, 'rb') as fp:
            model = cPickle.load(fp)

#read feature matrix (rows: ~10mil SNPs, columns: baseline-LD features and label ('Y' = 1 for top 10%, 0 for bottom 40%)
df = pd.read_csv("directory/trainingdata_baselineLD_MAF005_merged.%s.tsv.gz" %traininginput, sep="\t")

df = df.drop("Unnamed: 0",axis=1)
df = df.drop("SNP",axis=1)
Y = df[["Y"]]
df = df.drop("Y",axis=1)

#exclude not used features (MAF bins)
X = df.drop(["MAFbin1","MAFbin2","MAFbin3","MAFbin4","MAFbin5","MAFbin6","MAFbin7","MAFbin8","MAFbin9","MAFbin10"], axis=1)
Y = Y.as_matrix()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, y=Y)
#shap.initjs()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig("summary_plot_exact_%s.png" %traininginput, dpi=300, bbox_inches = "tight")

shap.summary_plot(shap_values, X, show=False)
plt.savefig("summary_plot_signed_exact_%s.png" %traininginput, dpi=300, bbox_inches = "tight")

#shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], matplotlib=True, show=False)

