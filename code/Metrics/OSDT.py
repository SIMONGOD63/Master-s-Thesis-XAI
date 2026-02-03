'''
it computes all the metrics for OSDT, i.e. Optimal Sparse Decition Tree
This code was done and cleaned with the help of chatGPT.
'''

#%% Import section

import pandas as pd 
import numpy as np
from gosdt import GOSDTClassifier, ThresholdGuessBinarizer
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

path = r"C:\Users\simeo\A. Data Science\Master thesis\01. Data"

import os
path2 = r"C:\\Users\\simeo\\A. Data Science\\Master thesis\\02. Code\\Techniques"
os.chdir(path2)

import functions_and_class as fc
import metrics 
# ---- the data
X_train_enco = pd.read_csv(path + "/00. FINALS/X_train_aligned.csv",index_col=0)
#y_train_enco = pd.read_csv(path + "/00. FINALS/y_train_aligned.csv",index_col=0)
X_test_enco = pd.read_csv(path + "/00. FINALS/X_test_aligned.csv",index_col=0)
#y_test_enco= pd.read_csv(path + "/00. FINALS/y_test_aligned.csv",index_col=0)
raw_ds = pd.read_parquet(path +"/00. FINALS/raw_aligned.parquet")
#---- Parameters

# Threshold guessing
GBDT_N_EST = 40#40 => keep it at 40
GBDT_MAX_DEPTH = 1

# Optimization parameters
reg_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
REGULARIZATION = reg_values[4] # penalize the number of leaves
SIMILAR_SUPPORT = False
DEPTH_BUDGET = 6#6 # Maximum allowed depth of the tree
TIME_LIMIT = 60*4 # Maximum optimization time in seconds
VERBOSE = True

#%% Get the training set 

# Index of the training
training_idx = list(set(X_train_enco["absolute_idx"]))
testing_idx = list(X_test_enco["absolute_idx"]) # no oversampling, no need for set

# Double check natural index is aligned with absolute idx
assert (raw_ds.index.values == raw_ds["absolute_idx"].values).all()

train_set = raw_ds.loc[training_idx].copy()
testing_set = raw_ds.loc[testing_idx].copy()

#---- Define the same training and testing set as my NN network
X_train = train_set.drop(columns = ["op_intent","absolute_idx","lat","long"],axis = 0).copy()
y_train = train_set["op_intent"].copy()

X_test = testing_set.drop(columns = ["op_intent","absolute_idx","lat","long"], axis = 0).copy()
y_test = testing_set["op_intent"].copy()

# assert correct shape
assert X_train.shape[1] == X_test.shape[1] == train_set.shape[1]- 4

del(X_train_enco,X_test_enco)

#%% Encode it for OSDT, into K categories and not k-1

# =============================================================================
# #---- Map the encoding variables.
# =============================================================================

ORDINAL_MAPS = {
    "ses_educ": {
        "None":0, "Prim": 1, "Sec": 2, "Coll": 3,
        "Bacc": 4, "Master": 5, "PhD": 6
    },
    "ses_income": {
        "None": 0, "i1to30": 1, "i31to60": 2,
        "i61to90": 3, "i91to110": 4,
        "i111to150": 5, "i151to200": 6,
        "i201toInf": 7
    },
    "age":{
        "34m":0, "3554":1, "55p":2
    },
    "day":{"34m": 0, "3554": 1, "55p": 2}
}

for key in ORDINAL_MAPS.keys():
    X_train[key].map(ORDINAL_MAPS[key])
    X_test[key].map(ORDINAL_MAPS[key])

#------- Activities
acts = ["act_VisitsMuseumsGaleries","act_Fishing","act_Hunting",
       "act_MotorizedOutdoorActivities","act_Volunteering"]

for var in acts:
    X_train[var] = X_train[var]*4
    X_test[var] = X_test[var]*4

# =============================================================================
# #---- encode into k categories
# =============================================================================

to_encode = X_train.select_dtypes("object").columns.tolist()

X_train = pd.concat([pd.get_dummies(X_train[to_encode],dtype = int),X_train], axis =1 )
X_test = pd.concat([pd.get_dummies(X_test[to_encode],dtype = int),X_test], axis =1 )

#---Deleted them
X_train.drop(to_encode, axis =1, inplace = True)
X_test.drop(to_encode, axis =1, inplace = True)

#--- Check if correct shape 
assert X_train.shape[1] == X_test.shape[1]

print(f"X_train has a shape after encoding of : {X_train.shape}")
#%% Guess the thresholds

enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST,
                              max_depth=GBDT_MAX_DEPTH,
                              random_state=4243)
enc.set_output(transform="pandas")

X_train_guessed = enc.fit_transform(X_train, y_train)
X_test_guessed = enc.transform(X_test)
                         
print(f"After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}")
print(f"train set column names == test set column names: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}")

#%% Guess Lower Bounds

#--- Create a class weights
class_counts = y_train.value_counts()
n_samples = len(y_train)
n_classes = class_counts.shape[0]

# Inverse frequency weights
class_weights = n_samples /(n_classes * class_counts)
weights_dic = class_weights.to_dict()
classes = list(class_weights.index.values) 
# Convert to GradientBoosting formats
GB_weights = y_train.map(weights_dic) 

# Convert to OSDT format
cost_matrix ={
    c_true: {
        c_pred: (0 if c_pred == c_true else weights_dic[c_true])
        for c_pred in classes
    }
    for c_true in classes
}

# I can specify class weight to avoid this classifier to ignore of one my minority class
enc = GradientBoostingClassifier(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH,
                                 #sample_weight = GB_weights, 
                                 random_state=4142)

enc.fit(X_train_guessed, y_train,sample_weight=GB_weights)
warm_labels = enc.predict(X_train_guessed)

np.unique(warm_labels, return_counts = True)
#%% Inspect warm_labels

np.unique(warm_labels, return_counts=True)
assert  len(np.unique(warm_labels)) == n_classes
# Not one predictions of "PLQ", one of my occurring class => this is problematic.

#%% Train the GOSDT Classifier

clf = GOSDTClassifier(regularization=REGULARIZATION, 
                      similar_support=SIMILAR_SUPPORT, 
                      time_limit=TIME_LIMIT, 
                      depth_budget=DEPTH_BUDGET,
                      verbose=VERBOSE) 


clf.fit(X_train_guessed, y_train) #y_ref=warm_labels no ref cuz ref was ignoring one class

#%% Evaluate the model

print("Evaluating the model, extracting tree and scores", flush=True)


print(f"Model training time: {clf.result_.time}")
print(f"Training accuracy: {round(clf.score(X_train_guessed, y_train)*100,2)}") # 56.69 % => 57.19
print(f"Test accuracy: {round(clf.score(X_test_guessed, y_test)*100,2)}") # 56.65 % => 57.25

#%% Inspect the created model and creates Root object 

results = clf.get_result()
results.keys()
results["graph_size"]
results["time"]
results["model_loss"]

print(f"Has the model converged : {results['status']}")
print(f"Training time of {results['time']:.2f} seconds")
print(results["models_string"])

for tree in clf.trees_:
    print(tree)
    

tree_vars = ['cons_coffee_place_ind <= 0.5', 'ses_income_i201toInf <= 0.5',
       'ses_dwelling_App <= 0.5', 'act_transport_Car <= 0.5', 'age_34m <= 0.5',
       'age_55p <= 0.5', 'people_predict_CAQ <= 0.5',
       'act_VisitsMuseumsGaleries <= 0.5', 'act_VisitsMuseumsGaleries <= 1.5',
       'act_MotorizedOutdoorActivities <= 0.5', 'vehicule_VUS <= 0.5']


tree_map_vars_semantic = {
    0: "consumes coffee at place = No",
    1: "income ∈ [200, inf[",
    2: "dwelling ≠ Apartment",
    3: "transport ≠ Car",
    4: "age 34m = 0",
    5: "age 55p = 0",
    6: "predicts CAQ = No",
    7: "museum visits = 0",
    8: "museum visits ≤ 1",
    9: "motorized outdoor activities = No",
    10: "vehicle ≠ SUV"
}

tree_map_vars = {i: name for i,name in enumerate(tree_vars)}

tree_map_classes = {i :classe for i,classe in enumerate(clf.classes_)}

#----Creating the root object
import json

models = json.loads(results["models_string"])
type(models),len(models)

root = models[0]
type(root), root.keys() 

#%% Function to print 

def print_rules(node, path=None):
    if path is None:
        path = []

    # Leaf
    if "prediction" in node:
        pred_idx = node["prediction"]
        pred_label = tree_map_classes[pred_idx]

        rule = " AND ".join(path) if path else "TRUE"
        print(f"IF {rule} THEN class = {pred_label}")
        return

    # Internal node
    feat_idx = node["feature"]
    predicate = tree_map_vars_semantic[feat_idx]

    # False branch
    print_rules(
        node["false"],
        path + [f"NOT ({predicate})"]
    )

    # True branch
    print_rules(
        node["true"],
        path + [f"({predicate})"]
    )
    

print_rules(root)
#%% Visualize the tree 

# Create a recursive function to get the stats.
def tree_stats(node):
    # Leaf
    if "prediction" in node:
        return {
            "internal_nodes": 0,
            "leaves": 1,
            "max_depth": 0,          # depth counted as number of splits (internal nodes) from root
            "used_predicates": set()
        }

    # Internal node
    feat = node["feature"]
    left = tree_stats(node["false"])
    right = tree_stats(node["true"])

    return {
        "internal_nodes": 1 + left["internal_nodes"] + right["internal_nodes"],
        "leaves": left["leaves"] + right["leaves"],
        "max_depth": 1 + max(left["max_depth"], right["max_depth"]), # exactly like Java exercices...
        "used_predicates": left["used_predicates"] | right["used_predicates"] | {feat}
    }

S = tree_stats(root)
n_internal = S["internal_nodes"]
n_leaves   = S["leaves"]
depth_obs  = S["max_depth"]          
n_used     = len(S["used_predicates"])

print("Internal nodes:", n_internal) # 35
print("Leaves:", n_leaves) # 36
print("Observed depth:", depth_obs) # 6
print("Unique predicates used:", n_used) # 10

# 3) Proportion of predicates used (needs denominator)
P = X_train_guessed.shape[1]
prop_used = n_used / P
print("Proportion of predicates used:", prop_used)


#%% Implement the average length 
'''
This metric will be used for F3 Selectivity.
'''
X_test_guessed.shape

def path_length_for_instance(x: pd.DataFrame, root):
    """
    x: pandas.DataFrame (one row of X_train_guessed)
    root: dict (parsed GOSDT tree)
    returns: int (number of internal nodes visited)
    """
    node = root
    depth = 0

    while "feature" in node:
        feat_idx = node["feature"]
        feature_name = tree_map_vars[feat_idx]
        feat_value = x[feature_name].values[0]

        depth += 1

        if feat_value == 1:
            node = node["true"]
        else:
            node = node["false"]

    return depth

single_test_instance = X_test_guessed.iloc[[0.0]]
path_length_for_instance(single_test_instance, root)

def average_path_length(X, root):
    """
    X: pandas.DataFrame (X_train_guessed or X_test_guessed)
    root: dict (parsed GOSDT tree)
    returns: float
    """
    total = 0
    n = len(X)

    for i in range(n):
        x = X.iloc[[i]]
        total += path_length_for_instance(x, root)

    return total / n

# used for F3 selectivity 
avg_path_testing_set = average_path_length(X_test_guessed, root)
print(round(avg_path_testing_set)) # 5
#%% Implements MAP ONE INSTANCE => ONE RULE + predicted output

def explain_instance(x: pd.DataFrame, root, var_dic : dict):
    """
    x: pd.DataFrame with exactly one row (same columns as X_train_guessed)
    root: dict (parsed GOSDT tree)
    var_dic: Pass the dictionnary to rename variable given it's for presentation purpose (semantic one)
    or re use in code ()

    returns:
        predicted_class (str)
        rule (list of str)
    """
    node = root
    rule = []

    while "feature" in node:
        feat_idx = node["feature"]
        predicate = tree_map_vars[feat_idx]
        semantically_correct = var_dic[feat_idx]

        feat_value = x[predicate].values[0]

        if feat_value == 1:
            rule.append(f"({semantically_correct})")
            node = node["true"]
        else:
            rule.append(f"NOT ({semantically_correct})")
            node = node["false"]

    # Leaf
    pred_idx = node["prediction"]
    predicted_class = tree_map_classes[pred_idx]

    return predicted_class, rule

#----- Testing the function
one_instance = X_test_guessed.iloc[[0]]
correct_class = y_test.iloc[[0]]
pred_class, rule = explain_instance(one_instance, root, tree_map_vars_semantic)
print(len(rule))
for r in rule:
    print(" ",r)
    
#------ Functions that works for severals rows

def gosdt_predict(X: pd.DataFrame, root, var_dic):
    preds = []
    for i in range(len(X)):
        x = X.iloc[[i]]
        pred, _ = explain_instance(x, root,var_dic)
        preds.append(pred)
    return preds

def gosdt_predict_with_explanations(X: pd.DataFrame, root,var_dic):
    outputs = []
    for i in range(len(X)):
        x = X.iloc[[i]]
        pred, rule = explain_instance(x, root,var_dic)
        outputs.append({
            "prediction": pred,
            "rule": rule
        })
    return outputs

#----- Testing the functions
instances_test = X_test_guessed.sample(10, random_state = 4243)

gosdt_predict(instances_test, root, tree_map_vars_semantic)

out = gosdt_predict_with_explanations(instances_test.iloc[[0]], root, tree_map_vars)
len(out) # One for each point ! 
print("Everything works as expected !")
    
#%% F4 target sensitivity

# =============================================================================
# =============================================================================
# #  This section compute the F4 target sensitivity score
# =============================================================================
# =============================================================================

# done with the help of chatGPT
from typing import Tuple, List
from collections import Counter, defaultdict


# function to predict and get unique, compact, easily readable signature
def predict_and_signature(x: pd.DataFrame, root) -> Tuple[str, tuple]:
    """
    x: pd.DataFrame with exactly ONE ROW (same columns as X_*_guessed)
    root: parsed JSON tree dict
    returns:
      predicted_class (str)
      signature (tuple of (feat_idx:int, went_true:bool))
    """
    node = root
    sig: List[Tuple[int, bool]] = []

    while "feature" in node:
        feat_idx = node["feature"]
        predicate = tree_map_vars[feat_idx]
        feat_value = x[predicate].values[0]

        went_true = (feat_value == 1)
        sig.append((feat_idx, went_true))

        node = node["true"] if went_true else node["false"]

    pred_idx = node["prediction"]
    pred_class = tree_map_classes[pred_idx]
    return pred_class, tuple(sig)

x0 = X_test_guessed.iloc[[0]]
pred0, sig0 = predict_and_signature(x0, root)
pred0, sig0[:3], len(sig0)
len(sig0) == path_length_for_instance(x0, root)


def target_sensitivity_B2(X: pd.DataFrame, root):
    """
    Returns per-class stats:
      total_predicted[c]
      signature_counts[c] = Counter(signature -> count)
      p_max[c]
      TS_B2[c] = 1 - p_max[c]
    """
    sig_counts_by_class = defaultdict(Counter)
    totals = Counter()

    for i in range(len(X)):
        x = X.iloc[[i]]
        pred_class, sig = predict_and_signature(x, root)
        totals[pred_class] += 1
        sig_counts_by_class[pred_class][sig] += 1

    results = {}
    for c, total in totals.items():
        most_common = sig_counts_by_class[c].most_common(1)[0][1]
        p_max = most_common / total
        ts_b2 = 1 - p_max
        results[c] = {
            "n_pred": total,
            "n_unique_paths": len(sig_counts_by_class[c]),
            "p_max": p_max,
            "TS_B2": ts_b2
        }
    return results

ts = target_sensitivity_B2(X_test_guessed , root)
ts
ts.keys()

sensitivities = [ts[key]["TS_B2"] for key in ts.keys()]
average_sensitivity =  np.mean(sensitivities) # 0.42
std_sensitivity = np.std(sensitivities) # 0.2856
np.std(sensitivities) # 0.2856
np.median(sensitivities) # 0.5 => REPORTED RESULT (0.5310880829015544)
#$$$$$$

import scipy
scipy.stats.iqr(sensitivities) # 0.4182
print(f"Average F4.2 Target sensitivity of OSDT : { round(average_sensitivity,2)} with standard deviation : {round(std_sensitivity,4)}")
#%% F7 Metric Incremental Deletion

# Same selected index in absolute_idx's values



def get_prediction_and_path_indices(x: pd.DataFrame, root, map_var,map_dic ):
    """
    x: pd.DataFrame with exactly one row (true predicate columns)
    root: dict (parsed GOSDT tree)

    returns:
      predicted_class (str)
      path_features (list of int, in traversal order)
    """
    node = root
    path_features = []

    while "feature" in node:
        feat_idx = node["feature"]
        predicate = map_var[feat_idx]
        feat_value = x[predicate].values[0]

        path_features.append(feat_idx)

        node = node["true"] if feat_value == 1 else node["false"]

    pred_idx = node["prediction"]
    predicted_class = map_dic[pred_idx]
    return predicted_class, path_features


x0 = X_test_guessed.iloc[[0]]
pred0, path0 = get_prediction_and_path_indices(x0, root, tree_map_vars,tree_map_classes)

def flip_predicates(x: pd.DataFrame, feature_indices):
    """
    x: pd.DataFrame with exactly one row
    feature_indices: iterable of int (indices in tree_map_vars)

    returns:
      x_flipped: pd.DataFrame (one row)
    """
    x_flipped = x.copy()

    for feat_idx in feature_indices:
        predicate = tree_map_vars[feat_idx]
        x_flipped[predicate] = 1 - x_flipped[predicate].values[0]

    return x_flipped

x0 = X_test_guessed.iloc[[0]]
#_, path0 = get_prediction_and_path_indices(x0, root, tree_map_vars,tree_map_classes)

x1 = flip_predicates(x0, path0[:1])   # flip first path feature
x2 = flip_predicates(x0, path0[:2])   # flip first two

x0.equals(x1), x1.equals(x2)

import random

def incremental_deletion_instance(
    x: pd.DataFrame,
    root,
    n_random : int = 5,
):
    """
    returns:
      dict with:
        'on_path': list[int]   # 1 if prediction changed at step k, else 0
        'off_path': list[float]  # average over random trials
    """
    
    original_pred, path = get_prediction_and_path_indices(x, root, tree_map_vars,tree_map_classes)
    L = len(path)

    all_features = set(tree_map_vars.keys())
    off_path_features = list(all_features - set(path))

    on_path_changes = []
    off_path_changes = []

    for k in range(1, L + 1):
        # --- On-path deletion ---
        x_on = flip_predicates(x, path[:k])
        pred_on, _ = get_prediction_and_path_indices(x_on, root,tree_map_vars,tree_map_classes)
        on_path_changes.append(int(pred_on != original_pred))

        # --- Off-path deletion (random baseline) ---
        changes = []
        for i in range(n_random):
            if (k > len(off_path_features)):
                random.seed(4243 + i )
                chosen = random.sample(off_path_features, len(off_path_features))
                
            else :
                random.seed(4243 +1 + i )
                chosen = random.sample(off_path_features, k)
                
            x_off = flip_predicates(x, chosen)
            pred_off, _ = get_prediction_and_path_indices(x_off, root,tree_map_vars,tree_map_classes)
            changes.append(int(pred_off != original_pred))

        off_path_changes.append(sum(changes) / n_random)

    return {
        "on_path": on_path_changes,
        "off_path": off_path_changes
    }

x0 = X_test_guessed.iloc[[0]]
curves = incremental_deletion_instance(x0, root)

curves["on_path"]
curves["off_path"]

#%% COmpute the metric

'''

The selected instance were just obtained in the F7 metrics.py file using 
selected_instance_idx = (y_comp.loc[mask_CAQ,"absolute_idx"].sample( np.minimum(30,sum(mask_CAQ)), random_state = 4243))
print(selected_instance_idx.values)
[37777. 54854. 14245. 53366. 12406. 46304. 11296. 32867. 16534. 52366.
 23423. 17606. 61923. 17118. 53398. 16614. 62971. 17823.]

'''
selected_instance_idx = [37777., 54854., 14245., 53366., 12406., 46304.,11296., 32867., 16534., 52366.,
 23423., 17606., 61923., 17118., 53398., 16614., 62971., 17823.]
selected_instance_osdt = X_test_guessed.loc[selected_instance_idx].copy()

results_per_instance = [] 
for i in range(len(selected_instance_osdt)):
    try:
        results_per_instance.append(incremental_deletion_instance(selected_instance_osdt.iloc[[i]], root))
    except:
        print(f"problem at index {i}")
        

# This function is there because not all path decision have the same length.
def normalize_curve(curve):
    """
    curve: list of values (length L)
    returns:
      x: np.ndarray of shape (L,)
      y: np.ndarray of shape (L,)
    """
    L = len(curve)
    x = np.arange(1, L + 1) / L
    y = np.array(curve)
    return x, y

x, y = normalize_curve(results_per_instance[0]["on_path"])
x[-1] == 1.0

# =============================================================================
# #--------------------
# =============================================================================

COMMON_X = np.linspace(0.0, 1.0, 50)

def interpolate_curve(x, y, common_x):
    """
    x, y: instance-specific curve
    common_x: shared grid
    returns:
      y_interp: np.ndarray of shape (len(common_x),)
    """
    return np.interp(common_x, x, y, left=0.0, right=y[-1])

# =============================================================================
# #--------------------
# =============================================================================

def aggregate_curves(results_per_instance, common_x):
    """
    results_per_instance: list of dicts with 'on_path' and 'off_path'
    returns:
      mean_on, std_on, mean_off, std_off
    """
    on_curves = []
    off_curves = []

    for res in results_per_instance:
        x_on, y_on = normalize_curve(res["on_path"])
        x_off, y_off = normalize_curve(res["off_path"])

        on_curves.append(interpolate_curve(x_on, y_on, common_x))
        off_curves.append(interpolate_curve(x_off, y_off, common_x))

    on_curves = np.vstack(on_curves)
    off_curves = np.vstack(off_curves)

    return {
        "mean_on": on_curves.mean(axis=0),
        "std_on": on_curves.std(axis=0, ddof=1),
        "mean_off": off_curves.mean(axis=0),
        "std_off": off_curves.std(axis=0, ddof=1),
    }

agg = aggregate_curves(results_per_instance, COMMON_X)
agg["mean_on"].shape == COMMON_X.shape
agg["std_on"].min() >= 0

# =============================================================================
# #--------------------
# =============================================================================

def compute_auc(y, x):
    return np.trapz(y, x)

auc_on  = compute_auc(agg["mean_on"], COMMON_X)
auc_off = compute_auc(agg["mean_off"], COMMON_X)
delta_auc = auc_on - auc_off

incremental_deletion_osdt = round(auc_on,1) #0.6


# =============================================================================
# #--------------------COMPUTE confidence INTERVALS
# =============================================================================
def bootstrap_auc(results_per_instance, common_x, n_boot=1000, seed=0):
    auc_on_list = []
    auc_off_list = []

    n = len(results_per_instance)

    for i in range(n_boot):
        random.seed(4243 + i)
        sample = [results_per_instance[random.randrange(n)] for _ in range(n)]
        agg = aggregate_curves(sample, common_x)
        auc_on_list.append(compute_auc(agg["mean_on"], common_x))
        auc_off_list.append(compute_auc(agg["mean_off"], common_x))

    return np.array(auc_on_list), np.array(auc_off_list)


auc_on_bs, auc_off_bs = bootstrap_auc(results_per_instance, COMMON_X)
delta_bs = auc_on_bs - auc_off_bs

#--- Get interval
ci_on = np.percentile(auc_on_bs,[2.5,97.5])
ci_off =  np.percentile(auc_off_bs,[2.5,97.5])
ci_delta = np.percentile(delta_bs,[2.5,97.5])

#%% F7.2 ROAR 


#%% F7.3 
def true_tree_rule(x):
    """
    Ground-truth decision process (white box).
    Returns class {0,1}.
    """
    if x["x1"] <= 0.5:
        return 0
    else:
        if x["x2"] <= 0.3:
            return 1
        else:
            return 0
        


np.random.seed(42)
n = 500

X = pd.DataFrame({
    "x1": np.random.rand(n),
    "x2": np.random.rand(n),
})

y = X.apply(true_tree_rule, axis=1)
y = y.astype(int)


#%% Train my model on it 
from sklearn.model_selection import train_test_split

X_train_simple, X_test_simple, y_train_simple,y_test_simple = train_test_split(X,y,
                                                                               test_size= 0.2,
                                                                               random_state = 4243)

enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST,
                              max_depth=GBDT_MAX_DEPTH,
                              random_state=4243)
enc.set_output(transform="pandas")


X_train_guessed_simple = enc.fit_transform(X_train_simple, y_train_simple)
X_test_guessed_simple = enc.transform(X_test_simple)

# Sanity check
assert X_train_guessed.isin([0,1]).all().all()

#---------- Train the model and get roots
clf.fit(X_train_guessed_simple , y_train_simple)


print("Evaluating the model, extracting tree and scores", flush=True)


print(f"Model training time: {clf.result_.time}")
print(f"Training accuracy: {round(clf.score(X_train_guessed_simple, y_train_simple)*100,2)}") # 100%
print(f"Test accuracy: {round(clf.score(X_test_guessed_simple, y_test_simple)*100,2)}") # 100%

#%% Inspect the created model ! 

results = clf.get_result()

tree_vars_simple = X_train_guessed_simple.columns.tolist()

tree_map_vars_simple = {i: name for i,name in enumerate(tree_vars_simple)}
map_classes_simple = {i :classe for i,classe in enumerate(clf.classes_)}

#----Creating the root object
import json

models = json.loads(results["models_string"])
type(models),len(models)

root_simple = models[0]
type(root_simple), root_simple.keys() 

get_prediction_and_path_indices(X_test_guessed_simple.iloc[[0]], root_simple, tree_map_vars_simple,map_classes_simple)
get_prediction_and_path_indices(X_test_guessed_simple.iloc[:5,:], root_simple, tree_map_vars_simple,map_classes_simple)


#%%
agreements = []

for i in range(len(X_test_simple)):
    score = fc.osdt_f7_3_instance(
        X_test_simple.iloc[i],
        X_test_guessed_simple.iloc[[i]],
        root_simple
    )
    agreements.append(score)
    
print(agreements)
mean_agreement = np.mean(agreements)
std_agreement = np.std(agreements)
print(f"Mean agreement of {round(mean_agreement,1)} with standard deviation : {round(std_agreement,4)}")
# saved on the main file
metrics.f7_3_score(mean_agreement)


#%% F9

# ------------------------------------------------
# ------------------------- F9 Stability
# ------------------------------------------------
neighborhoods_abs = {53031.0: [53031.0, 56741.0, 12691.0, 55294.0, 51082.0, 54518.0, 54923.0, 38816.0,45423.0, 41880.0, 48350.0, 53396.0, 52162.0, 18032.0, 22468.0], 
                     34716.0: [34716.0, 27119.0, 19628.0, 14405.0, 33161.0, 3441.0, 34235.0, 29811.0,36035.0, 27738.0, 3948.0, 22058.0, 32086.0, 35123.0, 14489.0], 
                     2542.0: [2542.0, 9735.0, 13757.0, 12230.0, 6931.0, 13412.0, 58225.0, 11876.0, 3887.0, 33827.0, 430.0, 14405.0, 29892.0, 12548.0, 14737.0], 
                     14405.0:[14405.0, 21432.0, 24661.0, 32949.0, 34235.0, 27873.0, 46322.0, 46759.0, 33827.0, 28324.0, 22468.0, 28321.0, 34716.0, 14737.0, 30976.0],
                     60197.0:[60197.0, 3415.0, 45961.0, 34485.0, 63588.0, 13806.0, 61593.0, 58225.0, 57243.0, 12839.0, 3887.0, 61460.0, 9524.0, 57827.0, 12691.0], 
                    52778.0: [52778.0, 52599.0, 45573.0, 12664.0, 55060.0, 31981.0, 15385.0,  47962.0, 53396.0, 31025.0, 430.0, 53031.0, 45269.0, 56741.0, 54289.0],
                    38122.0: [38122.0, 187.0, 37837.0, 41288.0, 17474.0,25684.0, 35785.0, 12206.0, 326.0, 26745.0, 1431.0, 46662.0, 31783.0, 34509.0, 26525.0],
                    46288.0: [46288.0, 51082.0, 46322.0, 3415.0, 8673.0, 22500.0, 18032.0, 11876.0, 35123.0, 2535.0, 58225.0, 17374.0, 49251.0, 41288.0, 2542.0], 
                    39532.0: [39532.0, 14725.0, 43807.0, 22025.0, 37124.0, 51196.0, 49251.0, 53906.0, 3948.0, 14405.0, 33416.0, 31981.0, 33827.0, 55060.0, 18219.0], 
                    20022.0: [20022.0, 35283.0, 26968.0, 11021.0, 56741.0, 13757.0, 33416.0, 52162.0, 34778.0, 24661.0, 26007.0, 34461.0, 17374.0, 18032.0, 11876.0], 
                    13412.0: [13412.0, 27119.0, 10172.0, 12335.0, 34485.0, 1756.0, 2542.0, 29528.0, 11021.0, 12206.0, 17374.0, 2535.0, 13806.0, 4316.0, 25684.0], 
                    28551.0: [28551.0, 52596.0, 17374.0, 19628.0, 27119.0, 26808.0, 35785.0, 26740.0, 62874.0, 34778.0, 14489.0, 34010.0, 28404.0, 28620.0, 5786.0],
                    53396.0: [53396.0, 6838.0, 54881.0, 3333.0, 34461.0, 35785.0, 50711.0, 11021.0, 34999.0, 49251.0, 52239.0, 56525.0, 48350.0, 38564.0, 56741.0], 
                    45573.0: [45573.0, 34461.0, 52778.0, 31981.0, 23906.0, 14489.0, 56087.0, 7929.0, 10006.0, 54289.0, 12548.0, 3887.0, 32977.0, 51196.0, 12664.0], 
                    2987.0: [2987.0, 5117.0, 23906.0, 1784.0, 12230.0, 13812.0, 12206.0, 6162.0, 3887.0, 22025.0, 44279.0, 13412.0, 10006.0, 45269.0, 62562.0], 
                    57827.0: [57827.0, 57243.0, 61460.0, 58225.0, 3415.0, 27873.0, 1756.0, 54881.0, 22468.0, 62874.0, 11876.0, 1668.0, 54923.0, 60197.0, 57143.0],
                    52596.0: [52596.0, 28551.0, 46322.0, 17374.0, 14489.0, 45617.0, 54289.0, 26740.0, 49251.0, 49790.0, 53386.0, 55294.0, 54881.0, 35785.0, 34778.0], 
                    17768.0: [17768.0, 18032.0, 34461.0, 29811.0, 29471.0, 18157.0, 35123.0, 31715.0, 35283.0, 7929.0, 12680.0, 8538.0, 35785.0, 26740.0, 21492.0], 
                    26968.0: [26968.0, 35283.0, 20022.0, 56741.0, 34778.0, 54881.0, 12335.0, 22468.0, 34461.0, 17374.0, 13806.0, 18032.0, 51082.0, 26740.0, 12548.0], 
                    14573.0: [14573.0, 12247.0, 33827.0, 46759.0, 20022.0, 29528.0, 34235.0, 23525.0, 34461.0, 33416.0, 24661.0, 14405.0, 19200.0, 52239.0, 30976.0]}

# function to predict and get unique, compact, easily readable signature
#def predict_and_signature(x: pd.DataFrame, root) -> Tuple[str, tuple]:

similarities = []
similarity_std = []
for key in neighborhoods_abs.keys():
    
    signature_decision = []
    for neighbor in neighborhoods_abs[key]:
        
        pred, sign = fc.predict_and_signature(X_test_guessed.loc[[neighbor]], root)
        
        if pred == "CAQ":
            signature_decision.append(sign)
        
   
    # check if the following is meaningful
    if len(signature_decision) < 2:
        continue
    
    # #--- I need to compute the pairwise distance with reference point
    pairwise_distances = []
    for i in range(1,len(signature_decision)):
            pairwise_distances.append(fc.compute_sign_dist(signature_decision[0], signature_decision[i]))
    
    
    #batch_avg_dist = np.mean(pairwise_distances)
    #batch_standard_deviation = np.std(pairwise_distances)
    
    batch_similarity = np.mean([1 / (1 + d) for d in pairwise_distances])
    batch_sim_std = np.std([1 / (1 + d) for d in pairwise_distances])
    
    similarities.append(batch_similarity)
    similarity_std.append(batch_sim_std)
    
    
avg_similarity = np.mean(similarities) #
mean_std_sim = np.mean(similarity_std) # 

print(f'OSDT similarity is :{round(avg_similarity,1)} with a mean standard deviation of {round(mean_std_sim,4)}')

#%%% IDENTITY

'''
1 BY CONSTRUCTION SIUUUUUU
'''    
    
#%% F10 
# ------------------------------------------------
# ------------------------- F10 Uncertainty
# ------------------------------------------------

C1 = None
C2 = 1
C3 = "???" # I don't know really
C4 = 0 
C5= 1

#%% F11
from time import time
# ------------------------------------------------
# ------------------------- F11 SPEED
# ------------------------------------------------
X_train_enco = pd.read_csv(path + "/00. FINALS/X_train_aligned.csv",index_col=0)
X_test_enco = pd.read_csv(path + "/00. FINALS/X_test_aligned.csv",index_col=0)


start_time = time()

# =============================================================================
# =============================================================================
# # ----- START THE PROCESS
# =============================================================================
# =============================================================================
# Threshold guessing
GBDT_N_EST = 40#40 => keep it at 40
GBDT_MAX_DEPTH = 1

# Optimization parameters
reg_values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
REGULARIZATION = reg_values[4] # penalize the number of leaves
SIMILAR_SUPPORT = False
DEPTH_BUDGET = 6#6 # Maximum allowed depth of the tree
TIME_LIMIT = 60*4 # Maximum optimization time in seconds
VERBOSE = True


# Index of the training
training_idx = list(set(X_train_enco["absolute_idx"]))
testing_idx = list(X_test_enco["absolute_idx"]) # no oversampling, no need for set

# Double check natural index is aligned with absolute idx
assert (raw_ds.index.values == raw_ds["absolute_idx"].values).all()

train_set = raw_ds.loc[training_idx].copy()
testing_set = raw_ds.loc[testing_idx].copy()

#---- Define the same training and testing set as my NN network
X_train = train_set.drop(columns = ["op_intent","absolute_idx","lat","long"],axis = 0).copy()
y_train = train_set["op_intent"].copy()

X_test = testing_set.drop(columns = ["op_intent","absolute_idx","lat","long"], axis = 0).copy()
y_test = testing_set["op_intent"].copy()

# assert correct shape
assert X_train.shape[1] == X_test.shape[1] == train_set.shape[1]- 4

del(X_train_enco,X_test_enco)

# =============================================================================
# #---- Map the encoding variables.
# =============================================================================

ORDINAL_MAPS = {
    "ses_educ": {
        "None":0, "Prim": 1, "Sec": 2, "Coll": 3,
        "Bacc": 4, "Master": 5, "PhD": 6
    },
    "ses_income": {
        "None": 0, "i1to30": 1, "i31to60": 2,
        "i61to90": 3, "i91to110": 4,
        "i111to150": 5, "i151to200": 6,
        "i201toInf": 7
    },
    "age":{
        "34m":0, "3554":1, "55p":2
    },
    "day":{"34m": 0, "3554": 1, "55p": 2}
}

for key in ORDINAL_MAPS.keys():
    X_train[key].map(ORDINAL_MAPS[key])
    X_test[key].map(ORDINAL_MAPS[key])

#------- Activities
acts = ["act_VisitsMuseumsGaleries","act_Fishing","act_Hunting",
       "act_MotorizedOutdoorActivities","act_Volunteering"]

for var in acts:
    X_train[var] = X_train[var]*4
    X_test[var] = X_test[var]*4

# =============================================================================
# #---- encode into k categories
# =============================================================================

to_encode = X_train.select_dtypes("object").columns.tolist()

X_train = pd.concat([pd.get_dummies(X_train[to_encode],dtype = int),X_train], axis =1 )
X_test = pd.concat([pd.get_dummies(X_test[to_encode],dtype = int),X_test], axis =1 )

#---Deleted them
X_train.drop(to_encode, axis =1, inplace = True)
X_test.drop(to_encode, axis =1, inplace = True)

enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST,
                              max_depth=GBDT_MAX_DEPTH,
                              random_state=4243)
enc.set_output(transform="pandas")

X_train_guessed = enc.fit_transform(X_train, y_train)
X_test_guessed = enc.transform(X_test)

#--- Create a class weights
class_counts = y_train.value_counts()
n_samples = len(y_train)
n_classes = class_counts.shape[0]

# Inverse frequency weights
class_weights = n_samples /(n_classes * class_counts)
weights_dic = class_weights.to_dict()
classes = list(class_weights.index.values) 
# Convert to GradientBoosting formats
GB_weights = y_train.map(weights_dic) 

# Convert to OSDT format
cost_matrix ={
    c_true: {
        c_pred: (0 if c_pred == c_true else weights_dic[c_true])
        for c_pred in classes
    }
    for c_true in classes
}

# I can specify class weight to avoid this classifier to ignore of one my minority class
enc = GradientBoostingClassifier(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH,
                                 #sample_weight = GB_weights, 
                                 random_state=4142)

enc.fit(X_train_guessed, y_train,sample_weight=GB_weights)
warm_labels = enc.predict(X_train_guessed)


clf = GOSDTClassifier(regularization=REGULARIZATION, 
                      similar_support=SIMILAR_SUPPORT, 
                      time_limit=TIME_LIMIT, 
                      depth_budget=DEPTH_BUDGET,
                      verbose=VERBOSE) 


clf.fit(X_train_guessed, y_train) #y_ref=warm_labels no ref cuz ref was ignoring one class
results = clf.get_result()

# =============================================================================
# =============================================================================
# # ---------- END PROCESS
# =============================================================================
# =============================================================================
end_time = time()

runtime_osdt = end_time - start_time
print(f"Runtime DiCE : {runtime_osdt:.2f} seconds")

metrics.calculate_speed_score(runtime_osdt) # 0
'''

Solely training the model yields a score of zero (it takes 25s)
But this metric is unfair. We are training a WHOLE model
'''