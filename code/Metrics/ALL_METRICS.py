'''
This file is dedicated to to computing ALL the metrics for SHAP, LIME, DiCE
'''
#%% Import

#---- Core packages
import pandas as pd
import numpy as np
from tensorflow import keras
import shap
from lime.lime_tabular import LimeTabularExplainer

#---- Other packages
from time import time
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from random import sample,seed
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances

#--- Import my own functions
import os
path2 = r"C:\\Users\\simeo\\A. Data Science\\Master thesis\\02. Code\\Techniques"
os.chdir(path2)

import functions_and_class as fc
import metrics

path = r"C:\Users\simeo\A. Data Science\Master thesis\01. Data"

# Modele
mod = keras.models.load_model(path + "/00. FINALS/last_model.keras")

# Raw datasets
raw = pd.read_parquet(path + "/00. FINALS/raw_aligned.parquet")

# Testing sets
X_test = pd.read_csv(path + "/00. FINALS/X_test_aligned.csv", index_col= 0)
y_test = pd.read_csv(path + "/00. FINALS/y_test_aligned.csv", index_col= 0)

conv_class = {0 : "CAQ" ,1: "PQ", 2: "PLQ", 3: "QS", 4: "PCQ"}
conv_class_letter = {"CAQ":0 ,"PQ":1,"PLQ":2,"QS" :3,"PCQ":4}
del(path2)
#%% Delete the rows in raw response variable's value deleted in training set
mask = (raw["op_intent"] != "Other") & (raw["op_intent"] != "Did not vote")
assert (len(raw) - sum(mask)) == 1655
raw = raw[mask]


# Select the raw part of the test set.
raw = raw.loc[X_test["absolute_idx"]].copy()
# Check if no nan
assert raw.isna().any(axis = 0).sum() == 0, "Nan values introduced"

# check raw and X_test aligned then
assert (raw["absolute_idx"].values == X_test["absolute_idx"].values).all()
assert (X_test.index.values == y_test.index.values).all()

# Reset both since they are aligned
raw.reset_index(inplace = True, drop = True)
#X_test.reset_index(inplace = True, drop = True)
#y_test.reset_index(inplace = True, drop = True)
abs_to_classical = list(X_test["absolute_idx"].copy())

#---- Create y_comp
y_comp = fc.get_YCOMP(mod,X_test,y_test)
correct_points = fc.get_well_class_points(y_comp)
misclass_points = fc.get_missclassified_points(y_comp)

# Get the index of the points of interests
abs_idx = correct_points[0][1]
print(abs_idx)
ind_raw = raw.loc[[abs_idx]]
ind_enco = X_test.loc[[abs_idx]]

#--- does not work anymore for some reasons ** Didn't solve the issue because it's not mandatory at all


# =============================================================================
# boolean,_ = fc.same_individual(ind_raw, ind_enco)
# 
# if boolean:
#     print("It works. I can find individuals automatically ")
# =============================================================================

del(ind_raw,ind_enco)
#%% Define DICE

from dice_ml import Data, Model, Dice

# function and class already imported as fc

nn_features = X_test.drop(columns = ["absolute_idx"]).columns.tolist()

# Define the wrapper.
Dice_wrapper = fc.DiceNNWrapper(mod, nn_features)
    
# Define the dataset
dice_ds,dice_dic_idx, continuous_col,categories_col = fc.generate_dice_ds(raw)
# dice_ds and X_test are aligned since raw and X_test were aligned

dice_model = Model(
    model=Dice_wrapper,
    backend="sklearn"
)
# Define dice data

dice_data = Data(
    dataframe = dice_ds,
    continuous_features = continuous_col,
    categorical_features = categories_col,
    outcome_name ="op_intent"
    )

# Create the DiCE explainer
dice = Dice(
    dice_data,
    dice_model,
    method = "random"
    )

#%% Define the SHAP explainer


#--- Check if dice_ds
# Split dice_ds from now on 

X_dice_ds = dice_ds.drop(columns = "op_intent").copy()
y_dice_ds = pd.DataFrame(dice_ds["op_intent"].copy())

# Here I should definitely change to 100 samples
background_raw = X_dice_ds.sample(50, random_state=4243) # was previously dice_ds.drop(columns = "op_intent").sample(...)

#---- Create a background for lime
X_lime = X_test.drop(columns = ["absolute_idx"]).copy()
X_lime.reset_index(drop = True, inplace = True)
background_LIME = X_lime.loc[background_raw.index.values]


RAW_FEATURES_NAMES = background_raw.columns.tolist()
print(len(RAW_FEATURES_NAMES))

predict_fn = fc.make_predict_proba_from_raw(mod, nn_features,RAW_FEATURES_NAMES)

# define explainer 
kernel_shap_explainer = shap.KernelExplainer(
    predict_fn,
    background_raw,
)

#%% Define the LIME explainer


X_dtypes = X_test.dtypes
#X_test["lat"] = raw["lat"]
#X_test["long"] = raw["long"]

# Every continuous var is there, I checked using ORDINALS_COLS2 and CONTINUOUS_COLS2
X_lime_continuous = ["age","income","educ",
                     "act_VisitsMuseumsGaleries","act_Fishing","act_Hunting","act_MotorizedOutdoorActivities",
                     "act_Volunteering","voting_probability",
                     "lat_scaled","long_scaled"]

X_lime_categorical = [ col for col in X_lime.columns.values if col not in X_lime_continuous]

# categorical_features => list of indices corresponding to categrical columns
# categorical_names => map from in to list of names
map_cat_names = { i:col for i,col in enumerate(X_lime.columns.values) if col in X_lime_categorical}

map_cat_names2 = { i:np.array(X_lime[col]) 
                  for i,col in enumerate(X_lime.columns.values) 
                  if col in X_lime_categorical
                  }
categorical_names = {}

for col in X_lime_categorical:
    
    idx = X_lime.columns.get_loc(col)

    # Sanity check: binary feature

    # Build human-readable names
    #base_name = col.split("_", maxsplit = 1)  # optional cleanup
    nunique = X_lime[col].nunique()
    stock = {} # [0]*nunique
    
    stock[0] = f"no_{col}"
    stock[1] = col

    # LIME expects a LIST where index == value
    categorical_names[idx] = stock

raw["month"].value_counts()
# month 
categorical_names[149] = {9 : "sep",10:"oct"}

#---- Map the 0,1 to categorical names

for key, val in map_cat_names2.items():
    map_cat_names2[key] =  np.vectorize(categorical_names[key].get)(val)  



# Initialize LIME
lime_explainer = LimeTabularExplainer( training_data = X_lime.to_numpy(), # need numpy + no string. Encoded DS
                                      
                                      mode='classification', 
                                      
                                      # This I still need to change
                                      class_names =["CAQ","PQ","PLQ","QS","PCQ"], # same order as the model output
                                      
                                      categorical_features= list(map_cat_names.keys()), # need list of int
                                      categorical_names= map_cat_names2,#•categorical_names,#map_cat_names2,#categorical_names ,#map_cat_names,
                                      
                                      feature_names=X_lime.columns.tolist(), 
                                      
                                      discretize_continuous = False, # to be avoided for Ordinal var.
                                      random_state = 4243)


#%%  Choose a instance to study
# The reshape is needed for because the model does not take vectors
chosen_class = 0 
idx = correct_points[chosen_class][1]

# this is false for some reason # but when run from scratch, it's true.
idx in list(X_lime.index.values) 


instance_lime = X_lime.loc[[idx]]
instance_lime2 = X_lime.iloc[[idx]]

print(y_test.loc[[idx]]) # class 0 
#%% test LIME
lime_explanation = lime_explainer.explain_instance(instance_lime.values[0], mod.predict,num_features=7, top_labels= 1)
lime_explanation.top_labels[0] # 0
lime_explanation.local_pred # 0.2437 # far away but it was an hard point
mod.predict(instance_lime) # 0.3686

lime_explanation.score # 0.2306
# #r2 of surrogate model - a low score suggests that the surrogate model may not be capturing the black-box model's behavior accurately
#%% TEst dice

instance_test_dice = dice_ds.drop(columns = "op_intent").sample(1, random_state = 4243)
predict_fn(instance_test_dice) #4

counterfactual = dice.generate_counterfactuals(
    query_instances=instance_test_dice,
    total_CFs=3,
    desired_class=0,
    random_seed = 4243)

cf_test = counterfactual.cf_examples_list[0].final_cfs_df.drop("op_intent",axis = 1)
counterfactual.visualize_as_dataframe(show_only_changes=True)

#%% F1 Metric (FIXED)

# case-independent 

# =============================================================================
# #---- SCOPE
# =============================================================================

scope = {"LIME": 2, "KernelSHAP": 2,"DiCE" : 2,"OSDT":3}
# LIME provides local explanations
# KernelSHAP provides local explanations

# =============================================================================
# #---- Portability
# =============================================================================

portability = {"LIME": 2, "KernelSHAP": 2,"DiCE":2,"OSDT":1}
# LIME is model-agnostic; works with any black-box model.
# KernelShap similar to LIME

# =============================================================================
# #----- Access
# =============================================================================

access_lime = {"Data Access": 2, "Model Access": 2}  
# Needs data for initialization only (creating an explainer object) & 
#requires access to the model’s prediction function
access_kernel_shap = {"Data Access": 2, "Model Access": 2}  
# Similar to LIME
access_dice = {"Data Access" : 2,"Model Access":2}
access = {"LIME": 4, "KernelSHAP": 4, "DiCE" : 4,"OSDT": 0+ 3}

# =============================================================================
# -------- PRACTICALITY
# =============================================================================

practicality_lime = {"Applicability": 2, "Scalability": 1}  
# Data-agnostic, moderately scalable (computationally expensive for large datasets)
practicality_kernel_shap = {"Applicability": 2, "Scalability": 0}  
# Data-agnostic & impractical for large datasets

practicality_dice = {"Applicability" : 2, "Scalability": 1}
practicality_osdt = {"Applicability" : 1, "Scalability": 1}


practicality = {"LIME": 3, "KernelSHAP": 2, "DiCE" : 3,"OSDT":2}
# Should I say Counterfactuals or DiCE ? I think   


# =============================================================================
# --------- SUMMARIZE
# =============================================================================
total_repr = {key: scope[key] + portability[key] + access[key] + practicality[key] for key in scope}

#--- Datatable
f1 = pd.DataFrame({
    "Scope (F1.1)": [scope["LIME"], scope["KernelSHAP"],scope["DiCE"],scope["OSDT"]],
    "Portability (F1.2)": [portability["LIME"], portability["KernelSHAP"],portability["DiCE"],portability["OSDT"]  ],
    "Access (F1.3)": [access["LIME"], access["KernelSHAP"],access["DiCE"],access["OSDT"]],
    "Practicality (F1.4)": [practicality["LIME"], practicality["KernelSHAP"],practicality["DiCE"],practicality["OSDT"]],
    "Total": [total_repr["LIME"], total_repr["KernelSHAP"], total_repr["DiCE"],total_repr["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f1

#%% F2 Structure (FIXED)

# =============================================================================
# Expressive power
# =============================================================================
C_predefined = {"decision_tree", "text_summary", "causal_diagram", "bar_plot", "analogy", "rule"}

#--------- LIME parameters
n_lime = 1  # One explanatory output: feature influence
F_lime = ["table_summary", "bar_plot", "rule"] #It has rule in the terms of if feature_1 < 10, class = 1
score_lime = metrics.f2_1(n_lime, F_lime, C_predefined)
print(f"LIME Expressive Power Score: {score_lime}")

#--------- KernelSHAP parameters
n_kernelshap = 1  # One explanatory output: Shapley values
F_kernelshap = ["bar_plot","table_summary"]  
score_kernelshap = metrics.f2_1(n_kernelshap, F_kernelshap, C_predefined)
print(f"KernelSHAP Expressive Power Score: {score_kernelshap}")

#------------DiCE
n_dice = 1 # => One explanatory output : counterfactuals explanations
F_dice = ["table_summary"] #
score_dice = metrics.f2_1(n_dice, F_dice, C_predefined)
print(f"DiCE Expressive Power Score: {score_dice}")

#------------ OSDT
n_osdt = 1
F_osdt = ["rule","decision_tree"]
score_osdt = metrics.f2_1(n_osdt,F_osdt, C_predefined)
print(f"OSDT Expressive Power Score: {score_osdt}") # 4


expressive_power = {"LIME": score_lime, "KernelSHAP": score_kernelshap,
                    "DiCE":score_dice,"OSDT" : score_osdt}


# =============================================================================
# Graphical Integrity
# =============================================================================
graphical_integrity = {"LIME": 1, "KernelSHAP": 1,"DiCE":0,"OSDT":0}

# They all show a positive/negative distinction with color-coded bars in tabular domain
# DICE not an attribution method

# =============================================================================
# # f2.3 Morpho clarity (diff more relevant from less )
# =============================================================================

morphological_clarity = {"LIME": 1, "KernelSHAP": 1,"DiCE":0,"OSDT":1}
# In all of them, longer bars indicate more relevant features in tabular domain

# =============================================================================
# # layer separation (shows OG inputs)
# =============================================================================
layer_separation = {"LIME": 0, "KernelSHAP": 0,"DiCE":0,"OSDT":0}
# In all of them, the feature values of the input instance are visibles by default

# =============================================================================
# ----- SUMMARIZE
# =============================================================================
total_structure = {key: expressive_power[key] + graphical_integrity[key] + morphological_clarity[key] + layer_separation[key] for key in expressive_power}

f2 = pd.DataFrame({
    "Expressive Power (F2.1)": [expressive_power["LIME"], expressive_power["KernelSHAP"],expressive_power["DiCE"],expressive_power["OSDT"]],
    "Graphical Integrity (F2.2)": [graphical_integrity["LIME"], graphical_integrity["KernelSHAP"],graphical_integrity["DiCE"],graphical_integrity["OSDT"]],
    "Morphological Clarity (F2.3)": [morphological_clarity["LIME"], morphological_clarity["KernelSHAP"],morphological_clarity["DiCE"],morphological_clarity["OSDT"] ],
    "Layer Separation (F2.4)": [layer_separation["LIME"], layer_separation["KernelSHAP"],layer_separation["DiCE"],layer_separation["OSDT"]],
    "Total": [total_structure["LIME"], total_structure["KernelSHAP"], total_structure["DiCE"],total_structure["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f2
#%% F3 selectivity 
#------- DICE

# =============================================================================
# s_dice =  fc.get_dice_s(correct_points = correct_points,
#                         X_dice_ds= X_dice_ds, dice_explainer= dice)
# =============================================================================
s_dice = 8 # with only the current amount of point in correct_points
s_osdt = 5 # it is the rounded average path length.

#add points to correct points 


'''
The metric compute it in a wrong way IMO. Shouldn't be penalized if s in [5,7]
'''

# Quantify if the number of explanation shown is relevan
explanation_size = {"LIME": metrics.f3(s=8, tunable = True), 
                    "KernelSHAP": metrics.f3(s=8, tunable = True),
                    "DiCE": metrics.f3(s=s_dice, tunable = False),
                    "OSDT" : metrics.f3(s = s_osdt, tunable = False)} # set it to true to truly reflect the note IMO
# In this case, both LIME and SHAP allow us to choose a max number of features to display, but there are certain methods that do not allow (e.g., PFI)


f3 = pd.DataFrame({
    "Total": [explanation_size["LIME"], explanation_size["KernelSHAP"], explanation_size["DiCE"],explanation_size["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f3

#%% F4
# ------------------------------------------------
# ------------------------- F4 Constrastivity
#------------------------------------------------

# =============================================================================
# Constrastivity levels
# =============================================================================
contrastivity_level = {"LIME": 1, "KernelSHAP": 1, "DiCE": 2,"OSDT":0}
# All are contrastive to a predefined baseline, which is the average value
# PDP, for example, simply show importance scores - score is 0
# CIU, for example, is contrastive to a predefined baseline (which can be chosen as the average or an other prediction) and to 
# the current instance (like counterfactual explanations) - score is 2
#%% Target sensitivity DICE


#%% F4.2 Target sensitivity
# SHAP Get the maximum distance out of 50 points

# =============================================================================
# t1 = time()
# kernel_shap_values = []
# 
# # That takes a long amount of time. Just run it once.
# 
# for i in range(len(background_raw)):
#    exp = kernel_shap_explainer(background_raw.iloc[i,:]) # 50 samples
#    kernel_shap_values.append(exp.values[:,0])
# t2 = time()
# print(round((t2 - t1)/60,2), "minutes") # 8.22 MINUTES
# 
# 
# # Now we need to compute d_max, but first we need all the pairwise distances
# # I computed it once, and I'm just using this result to avoid recomputations
# kernel_shap_distances = euclidean_distances(kernel_shap_values )
# =============================================================================



d_max_kernel = 0.4 #np.max(kernel_shap_distances)
print(round(d_max_kernel,1)) # 0.4

#%% LIME Get the maximum distance out of 50 points


# =============================================================================
# lime_values = []
# 
# for i in range(len(background_LIME)):
#     #label = mapped_labs[i]
#     exp = lime_explainer.explain_instance(background_LIME.iloc[i].values, mod.predict,top_labels=1)
#     pred_label = exp.top_labels[0]
#     lime_values.append([v for _, v in exp.as_list(label = pred_label)]) # => problem come from this line for some reasons.
# 
# # Compute Pairwise Distances and d_max
# lime_distances = euclidean_distances(lime_values)
# =============================================================================


d_max_lime =  0.9 #np.max(lime_distances)
print(round(d_max_lime,1)) #=> I have 0.9 now


#del(exp,pred_label,lime_values)


#%% Generate Nearest Counterfactual for each class

#dice_idx_query = dice_dic_idx[idx] # now problem here for some reasons ZZZZZZ

CATEGORICAL_COLS = ['month', 'imp_ind', 'app_noTattoo', 'cons_coffee', 'ses_dwelling', 'app_swag', 
                    'music', 'film', 'ses_ethn', 'act_transport', 'vehicule', 'cons_Smoke',
                    'cons_meat', 'cons_brand', 'animal', 'day', 'sport', 'alcohol', 'lang', 
                    'people_predict', 'gender', 'sex_ori', 'pays_qc', 'immigrant']

ORDINAL_COLS2 = {'ses_income' : 7,
                'ses_educ':6,
                'age':2, 
                'act_VisitsMuseumsGaleries':4,
                'act_Fishing':4, 'act_Hunting': 4,
                'act_MotorizedOutdoorActivities':4, 'act_Volunteering':4}

CONTINUOUS_COLS2 = {"lat": round(np.max(X_dice_ds["lat"]) -np.min( X_dice_ds["lat"]),4),
                   "long": round(np.max(X_dice_ds["long"]) -np.min( X_dice_ds["long"]),4),
                   'voting_probability':1}

#--- Generate one counterfactual for each class 
metricsf4_2_SHAP = []
metricsf4_2_LIME = []

for classe in correct_points.keys():
    
    for idx in correct_points[classe]:
        
        query_instance_dice = X_dice_ds.loc[[idx]] # needs shape (1,36) => DataFrame
        query_instance_for_LIME = X_lime.loc[idx] #needs shape (156,) => series

        classes = [0,1,2,3,4]
        for c in classes:
            if c != chosen_class:
                
                #---------------- Generate the CFs
                try :
                    counterfactual_f4 = dice.generate_counterfactuals(
                        query_instances=query_instance_dice,
                        total_CFs=3,
                        desired_class=c,
                        random_seed = 4243)
                except :
                    continue
                
                cf_instances = counterfactual_f4.cf_examples_list[0].final_cfs_df.drop("op_intent",axis = 1)
                
                
                
                # Safety check
                if cf_instances is None or cf_instances.shape[0] == 0:
                    continue
                
                #---------------- Choose the closest CF among the 5
                comp_ds = pd.concat([cf_instances,query_instance_dice],axis =0 )
                dist_matrix_cf = fc.gower_distance_matrix_fast(comp_ds, 
                                                            categorical_cols = CATEGORICAL_COLS ,
                                                            ordinal_cols = list(ORDINAL_COLS2.keys()) , 
                                                            continuous_cols = list(CONTINUOUS_COLS2.keys()), 
                                                            ordinal_ranges = ORDINAL_COLS2,
                                                            continuous_ranges = CONTINUOUS_COLS2)
                idx_instance = dist_matrix_cf.shape[0] -1
                dist_cf_to_x = dist_matrix_cf[idx_instance,:idx_instance]
                idx_closest = np.argmin(dist_cf_to_x)
                canonical_cf_f4 = cf_instances.iloc[[idx_closest]]
                
                
                
                #----------------SHAP explanations
                shap_value_og = kernel_shap_explainer(query_instance_dice) # error happens here
                shap_value_cf = kernel_shap_explainer(canonical_cf_f4)
                
                
                shap_value_og = shap_value_og.values[:,:,chosen_class][0]
                shap_value_cf = shap_value_cf.values[:,:,chosen_class][0]
                
                #-------------- LIME EXPLANATIONS OF CF
                lime_exp1 = lime_explainer.explain_instance(query_instance_for_LIME.values, mod.predict, labels= [classe],num_features=X_lime.shape[1]) # problem here
                #lime_val1 =[v for _, v in lime_exp1.as_list(label = classe)]
                lime_dict1 = dict(lime_exp1.as_list(label=classe))

                #---------------- Transform the instance into the encoded from for LIME
                cf_instance_LIME = fc.encode_for_nn(canonical_cf_f4, list(X_lime.columns.values))
                cf_instance_LIME = cf_instance_LIME.iloc[0].to_numpy()
                
                lime_exp2 = lime_explainer.explain_instance(cf_instance_LIME, mod.predict, labels = [classe],num_features=X_lime.shape[1])
                
                #----------------lime_val2 =[v for _, v in lime_exp2.as_list(label = classe)]
                lime_dict2 = dict(lime_exp2.as_list(label=classe))

                #---------------- align the vectors
                lime_vec1 = np.zeros(X_lime.shape[1])
                lime_vec2 = np.zeros(X_lime.shape[1])
                
                feature_to_idx = {f: i for i, f in enumerate(X_lime.columns)}
                
                #set(lime_dict2.keys()) == set(lime_dict1.keys())
                for feat, val in lime_dict1.items():
                    feat = feat.split("=")[0]
                    if feat in feature_to_idx:
                        lime_vec1[feature_to_idx[feat]] = val
                
                for feat, val in lime_dict2.items():
                    feat = feat.split("=")[0]
                    if feat in feature_to_idx:
                        lime_vec2[feature_to_idx[feat]] = val
                            
                
                #---------------- Compute the metric        
                metricsf4_2_SHAP.append(metrics.f4_2(shap_value_og, shap_value_cf, d_max=d_max_kernel, distance_metric=euclidean))
                metricsf4_2_LIME.append(metrics.f4_2(lime_vec1, lime_vec2, d_max=d_max_lime, distance_metric=euclidean))

        
    
    
avg_f4_2_SHAP = np.mean(metricsf4_2_SHAP) # 0.4
std_f4_2_SHAP = np.std(metricsf4_2_SHAP) # std of 0.237

avg_f4_2_LIME = np.mean(metricsf4_2_LIME) #0.3
std_f4_2_LIME = np.std(metricsf4_2_LIME) # std of 0.1233
print(f" SHAP mean metric F4.2 value :  {round(avg_f4_2_SHAP,1)} with standard deviation {round(std_f4_2_SHAP,4)}") # rounded ti 0.65
print(f" LIME mean metric F4.2 value :  {round(avg_f4_2_LIME,1)}  with standard deviation {round(std_f4_2_LIME,4)}") # rounded ti 0.65

# Rather use percentile
np.percentile(metricsf4_2_SHAP,[2.5,97.5]) # array([0.    , 0.7075])
np.percentile(metricsf4_2_LIME, [2.5,97.5]) #)array([0.1   , 0.5075])

np.median(metricsf4_2_SHAP) #0.4
np.median(metricsf4_2_LIME) # 0.3
import scipy
round(scipy.stats.iqr(metricsf4_2_SHAP),4) #0.4000
round(scipy.stats.iqr(metricsf4_2_LIME),4) #0.1750   
#%% Results F4
target_sensitivity = {"LIME": 0.4, "KernelSHAP": 0.3, "DiCE": 0, "OSDT":0.4}
contrastivity_level = {"LIME": 1, "KernelSHAP": 1, "DiCE": 2,"OSDT":0}
# =============================================================================
# ----- SUMMARIZE
# =============================================================================
total_contrast = {key: contrastivity_level[key] + target_sensitivity[key] for key in target_sensitivity.keys()}
f4 = pd.DataFrame({
    "Contrastivity Level (F4.1)": [contrastivity_level["LIME"], contrastivity_level["KernelSHAP"], contrastivity_level["DiCE"],contrastivity_level["OSDT"]],
    "Target Sensitivity (F4.2)": [target_sensitivity["LIME"], target_sensitivity["KernelSHAP"], target_sensitivity["DiCE"],target_sensitivity["OSDT"]],
    "Total": [total_contrast["LIME"], total_contrast["KernelSHAP"], total_contrast["DiCE"],total_contrast["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f4

#%% F5 (FIXED)
# ------------------------------------------------
# ------------------------- F5 INTERACTIVITY
#------------------------------------------------

# =============================================================================
# #--- Interaction Level
# =============================================================================

interaction_level = {"LIME": 1, "KernelSHAP": 1, "DiCE": 1,"OSDT":1} 
# In all of them: no built-in interactive tools, but API allows developers to implement
# interaction easily
# CF to check

# =============================================================================
# #--- Controllability
# =============================================================================

# limited control
controllability = {"LIME": 2, "KernelSHAP": 2, "DiCE": 2,"OSDT":1}
# Besides visual exploration, explanations can be refined by choosing specific features 
# or adjusting the number of features shown, offering limited control

# =============================================================================
# ----- SUMMARIZE
# =============================================================================
total_interactivity = {key: interaction_level[key] + controllability[key] for key in interaction_level}

f5 = pd.DataFrame({
    "Interaction Level (F5.1)": [interaction_level["LIME"], interaction_level["KernelSHAP"], interaction_level["DiCE"],interaction_level["OSDT"]],
    "Controllability (F5.2)": [controllability["LIME"], controllability["KernelSHAP"], controllability["DiCE"],controllability["OSDT"]],
    "Total": [total_interactivity["LIME"], total_interactivity["KernelSHAP"], total_interactivity["DiCE"],total_interactivity["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f5

#%% F6
# ------------------------------------------------
# ------------------------- F6 FIDELITY 
#------------------------------------------------


# =============================================================================
# #------- Surrogate Agreement SHAP
# =============================================================================

 # this is phi_0

kernel_preds = []
base_value = kernel_shap_explainer.expected_value[0] #

n_points = 100
random_points_idx = X_dice_ds.sample(n_points, random_state = 4243).index
rp_preds = predict_fn(X_dice_ds.loc[random_points_idx])
t1 = time()
for i in random_points_idx:
    exp = kernel_shap_explainer(X_dice_ds.loc[[i]])
    #base_value = exp.expected_value[0]
    surrogate_pred_shap = base_value + sum(exp.values[:,:,chosen_class][0]) # class 0
    kernel_preds.append([surrogate_pred_shap])

t2 = time()
print( round((t2 - t1)/60,2),"MINUTES" ) # 14 mins 20s !! now 11.85 mins SIIIIU
print(np.min(kernel_preds), np.max(kernel_preds)) # is this probability # yes
assert np.min(kernel_preds) >= 0 and np.max(kernel_preds) <= 1, "Not a probability"  
kernel_f6_2,kernel_std_f6_2 = metrics.f6_2(rp_preds[:,chosen_class], kernel_preds)
print(f"KernelSHAP Surrogate Agreement (F6.2): {kernel_f6_2} with standard deviation {round(kernel_std_f6_2,4)}")
# std 0.1769

#---- Confidence intertal
kernel_f6_2_CI = [round(kernel_f6_2 - 1.96*(kernel_std_f6_2/np.sqrt(100)),4) , round(kernel_f6_2 + 1.96*(kernel_std_f6_2/np.sqrt(100)),4) ]
#[0.7653, 0.8347]
#%% SURROGATE AGREEMENT LIME
lime_preds = []

for i in random_points_idx:
    exp = lime_explainer.explain_instance(X_lime.iloc[i].to_numpy(), mod.predict,top_labels=1)
    lime_preds.append( [exp.local_pred[0]] )

lime_f6_2,lime_std_f6_2 = metrics.f6_2(rp_preds[:,chosen_class], lime_preds)
print(f"LIME Surrogate Agreement (F6.2): {lime_f6_2} with standard deviation {round(lime_std_f6_2,4)}")
# std 0.1976
lime_f6_2_CI = [round(lime_f6_2 - 1.96*(lime_std_f6_2/np.sqrt(100)),4) , round(lime_f6_2 + 1.96*(lime_std_f6_2/np.sqrt(100)),4) ]
#[0.6617, 0.7383]


#%% SUMMARIZE
# about know the origin of the explanaion (surrogate and assumptions)

# =============================================================================
# ------- FIDELITY CHECK
# =============================================================================

fidelity_check = {"LIME": 0, "KernelSHAP": 0, "DiCE": 1,"OSDT":1}
# Surrogate model used in LIME 
#or linearity assumptions used kernelSHAP with default linear kernel

# =============================================================================
# ------ SUMMARIZE
# =============================================================================
surrogate_agreement = {"LIME": 0.7, "KernelSHAP": 0.8, "DiCE": 1,"OSDT":1}
#DiCE is 1 because no surrogate model is used


    
total_fidelity = {key: fidelity_check[key] + surrogate_agreement[key] for key in fidelity_check}

f6 = pd.DataFrame({
    "Fidelity Check (F6.1)": [fidelity_check["LIME"], fidelity_check["KernelSHAP"], fidelity_check["DiCE"],fidelity_check["OSDT"]],
    "Surrogate Agreement (F6.2)": [surrogate_agreement["LIME"], surrogate_agreement["KernelSHAP"], surrogate_agreement["DiCE"],surrogate_agreement["OSDT"]],
    "Total": [total_fidelity["LIME"], total_fidelity["KernelSHAP"], total_fidelity["DiCE"],total_fidelity["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f6

#%% F7 SEE DEDICATED FILE FOR DETAILS
incremental_deletion_lime = 0.8
incremental_deletion_dice = 0.5
incremental_deletion_kernel = 0.6 
incremental_deletion_osdt = 0.6
incremental_deletion = {"KernelSHAP": incremental_deletion_kernel,"LIME" : incremental_deletion_lime,"DiCE": incremental_deletion_dice,"OSDT":incremental_deletion_osdt}

#--- white box check score
white_box = {"LIME": 1, "KernelSHAP": 2, "DiCE": 0,"OSDT":2}
roar = {"KernelSHAP": None,"LIME": None,"DiCE":None,"OSDT":None}
total_faithfulness = {key: incremental_deletion[key] + white_box[key] for key in incremental_deletion}
f7 = pd.DataFrame({
    "Incremental Deletion (F7.1)": [incremental_deletion["LIME"], incremental_deletion["KernelSHAP"], incremental_deletion["DiCE"],incremental_deletion["OSDT"]],
    "ROAR (F7.2)": [roar["LIME"], roar["KernelSHAP"], roar["DiCE"],roar["OSDT"]],
    "White-Box Check (F7.3)": [white_box["LIME"], white_box["KernelSHAP"], white_box["DiCE"],white_box["OSDT"]],
    "Total": [total_faithfulness["LIME"], total_faithfulness["KernelSHAP"], total_faithfulness["DiCE"],total_faithfulness["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f7

#%% F8 (fixed)
# ------------------------------------------------
# ------------------------- F8 Truthfulness
#------------------------------------------------


# =============================================================================
# F.8.1 Reality check
# =============================================================================

'''
Checks whether the method prevents the generation of unrealistic data samples.
Use two sub-metric 
    A => Feature constraints consistency 
    B => Feature Correlation Consistency
'''


reality_check_A = {"LIME": 0, "KernelSHAP": 0,"DiCE" : 1,"OSDT":1}


# =============================================================================
# #---- Correlation check
#  also fixed
# =============================================================================

reality_check_B = {"LIME": 0, "KernelSHAP": 0,"DiCE":1,"OSDT":1}


reality_check = {"LIME": reality_check_A["LIME"] + reality_check_B["LIME"], 
                 "KernelSHAP": reality_check_A["KernelSHAP"] + reality_check_B["KernelSHAP"],
                 "DiCE" :reality_check_A["DiCE"] + reality_check_B["DiCE"],
                 "OSDT" :reality_check_A["OSDT"] + reality_check_B["OSDT"]
                 
                 }

#--------------------------------------------

# =============================================================================
# #---- Bias Detection
# =============================================================================


# check their code to understand the process they went through
bias_detection = {"LIME": 1, "KernelSHAP": 1,"DiCE" : 1,"OSDT":1}

total_truthfulness = {key: reality_check[key] + bias_detection[key] for key in reality_check}
f8 = pd.DataFrame({
    "Reality Check (F8.1)": [reality_check["LIME"], reality_check["KernelSHAP"],reality_check["DiCE"],reality_check["OSDT"]],
    "Bias Detection (F8.2)": [bias_detection["LIME"], bias_detection["KernelSHAP"],bias_detection["DiCE"],bias_detection["OSDT"]],
    "Total": [total_truthfulness["LIME"], total_truthfulness["KernelSHAP"],total_truthfulness["DiCE"],total_truthfulness["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f8

#%% F9 Stability

# ------------------------------------------------
# ------------------------- F9 Stability
#------------------------------------------------
'''
Test if the XAI method is robust to (small) changes in the input data.
'''


# =============================================================================
# # Similarity
# =============================================================================
'''
It evaluates the consistency of explanations for similar input samples (neighbors)
'''

#------ Get a distance metric

CATEGORICAL_COLS = ['month', 'imp_ind', 'app_noTattoo', 'cons_coffee', 'ses_dwelling', 'app_swag', 
                    'music', 'film', 'ses_ethn', 'act_transport', 'vehicule', 'cons_Smoke',
                    'cons_meat', 'cons_brand', 'animal', 'day', 'sport', 'alcohol', 'lang', 
                    'people_predict', 'gender', 'sex_ori', 'pays_qc', 'immigrant']

ORDINAL_COLS2 = {'ses_income' : 7,
                'ses_educ':6,
                'age':2, 
                'act_VisitsMuseumsGaleries':4,
                'act_Fishing':4, 'act_Hunting': 4,
                'act_MotorizedOutdoorActivities':4, 'act_Volunteering':4}

CONTINUOUS_COLS2 = {"lat": round(np.max(X_dice_ds["lat"]) -np.min( X_dice_ds["lat"]),4),
                   "long": round(np.max(X_dice_ds["long"]) -np.min( X_dice_ds["long"]),4),
                   'voting_probability':1}

print(len(ORDINAL_COLS2) + len(CONTINUOUS_COLS2))
#-----
chosen_class_name = conv_class[chosen_class]
string_col_name = "proba " + "C" + str(chosen_class)
mask_CAQ = (y_comp["true class"] == chosen_class) & (y_comp["correct"] == 1) & (y_comp[string_col_name] >= 0.6)
assert (X_dice_ds.index.values == y_comp.index.values).all()
distance_matrix = fc.gower_distance_matrix_fast(X_dice_ds[mask_CAQ],
                                   categorical_cols = CATEGORICAL_COLS ,
                                   ordinal_cols = list(ORDINAL_COLS2.keys()) , 
                                   continuous_cols = list(CONTINUOUS_COLS2.keys()), 
                                   ordinal_ranges = ORDINAL_COLS2,
                                   continuous_ranges = CONTINUOUS_COLS2)

#%% Get the neighboors

t1 = time()
# Define 30 random anchors
seed(4243)
anchors = sample(list(X_dice_ds[mask_CAQ].index.values), 20)

dic_dist_DS = {og_idx : idx_distance_matrix for idx_distance_matrix,og_idx in enumerate(X_dice_ds[mask_CAQ].index.values)}

dic_dist_DS[anchors[000]]
# number of nearest neighbors
k = 15
# 1 find the index of the nearest indiv for each anchor
# 2. translate it to the same index as X_dice that I can use
neighborhoods = { anchor :   list(X_dice_ds[mask_CAQ]
                                  .iloc[np.argsort(distance_matrix[dic_dist_DS[anchor ],:])[:k],:]
                                  .index.values)
                 for anchor in anchors
}


# =============================================================================
# PARALLEL ADVENTURE
# Get the same neighborhoods in abs_idx
# =============================================================================
# =============================================================================
# reverse_dice_dic = {value : key for key,value in dice_dic_idx.items() }
# print(neighborhoods)
# # neighborhoods in absolute idx 
# neighborhoods_abs = {}
# 
# for key in neighborhoods.keys():
#     
#     new_key = reverse_dice_dic[key]
#     
#     neighborhoods_abs[new_key] = [reverse_dice_dic[neighbor] for neighbor in neighborhoods[key]]
#     
# print(neighborhoods_abs)
# 
# =============================================================================

# =============================================================================
# 
# =============================================================================

##---- check if anchors in X_dice_ds idx

seum = 0
for i in range(len(anchors)):
    seum += int(all(idx in X_dice_ds[mask_CAQ].index.values
    for idx in neighborhoods[anchors[i]])) #perfect
    
seum == len(anchors)

#---- Before the following section check if key = items[0]
for key in neighborhoods.keys():
    #break
    #key in neighborhoods[key]
    if ( neighborhoods[key][0] == key):
        continue;
    else :
        print(f"Problem with key {key}")
          

#----- Sort by probability
for neighbors in neighborhoods.keys():
    true_idx = neighborhoods[neighbors]
    pred_prob = predict_fn(X_dice_ds.loc[true_idx])
    pred_class = pred_prob.argmax(axis = 1)
    
    #--- Check if predict the same class.
    proper_neighbors = [ i for i,classe in enumerate(pred_class) if classe == chosen_class]    
    proper_neighbors_idx = [ true_idx[j] for j in proper_neighbors]
    
    neighborhoods[neighbors] = fc.get_close_prediction(len(proper_neighbors), pred_prob[proper_neighbors], chosen_class, proper_neighbors_idx)

#---- Sort by global radius skipped


#%% Compute the similarity for LIME AND SHAP

#--- Get the explanations for each points
kernel_explanations = []
lime_explanations = []

#Feature order
feat_names = X_lime.columns.tolist()
feat_to_pos = {f:i for i,f in enumerate(feat_names)}


for key,items in neighborhoods.items():
    
    expl_indiv_kernel = kernel_shap_explainer(X_dice_ds.loc[items],silent= True)
    instance_lime_explanations = []
    for item in items:
        
    
        vec = np.zeros(len(feat_names),dtype = float)
        #enco_items = fc.encode_for_nn(X_dice_ds.loc[items], nn_features)
        assert mod.predict(X_lime.iloc[[item]]).argmax() == 0 
                
        # We take the explanations for ALL the variables this time
        expl_indiv_lime = lime_explainer.explain_instance(X_lime.loc[item].to_numpy(), mod.predict, top_labels=1, num_features=X_lime.shape[1])
        assert expl_indiv_lime.top_labels[0] == 0
        
        pred_labelF9 = expl_indiv_lime.top_labels[0]
        #lime_val =[v for _, v in expl_indiv_lime.as_list(label = pred_labelF9)]
        #instance_lime_explanations.append(lime_val)
         
        for feat_idx,w in expl_indiv_lime.local_exp[pred_labelF9] :#.as_list(label = pred_labelF9):
            vec[feat_idx] = w 
        
        # stock the lime explanations for each item
        instance_lime_explanations.append(vec)
        
    # Stock the explanations for each batch of neighboors
    kernel_explanations.append(expl_indiv_kernel.values[:,:,chosen_class])
    lime_explanations.append(instance_lime_explanations)

t2 = time()
run_time = t2 - t1
print("#################################################################")
print(f"it's over runtime of {run_time/60:.6f} minutes") # 51 minutes !!!
# Now I can use f9 metric. Let's try if it works #??? I'm here
#%% download it cuz it took way too long
import pickle

# =============================================================================
# #--- To download
# with open( path  + "/00. FINALS/kernel_explanations_bg50_sim_a20_k15.pkl", "wb") as f:
#     pickle.dump(kernel_explanations, f)
#     
# with open( path  + "/00. FINALS/lime_explanations_bg50_sim_a20_k15.pkl", "wb") as f:
#     pickle.dump(lime_explanations, f)
#     
# =============================================================================

# bg50 => background
# a20 => 20 anchors
# k15 => 15 nearest neighbors
# sim => similarity


#--- To load
with open(path  + "/00. FINALS/kernel_explanations_bg50_sim_a20_k15.pkl", "rb") as f:
    kernel_explanations = pickle.load(f)
    
with open(path  + "/00. FINALS/lime_explanations_bg50_sim_a20_k15.pkl", "rb") as f:
    lime_explanations = pickle.load(f)
    
    
    
#%% COmpute the similarity for DICE

def jaccard_bool(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # identical: no feature changed in either
    return intersection / union

def mean_pairwise_jaccard(change_vectors):
    sims = []
    n = len(change_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(jaccard_bool(change_vectors[i], change_vectors[j]))
    if len(sims) == 0:
        return 1.0  # singleton batch
    return np.mean(sims)

#-------------------------------------------------------------

arbitrary_chosen_other_class = 3 # QS
similarities = []
for key,items in neighborhoods.items():
    
    
    #canonical_cfs = []
    changed_vectors = []
    # Generate a canonical CF per instance and per class...
    for item in items:
        
        instance_item = X_dice_ds.loc[[item]]
        cf_item = dice.generate_counterfactuals(
            query_instances=instance_item,
            total_CFs=5,
            desired_class= arbitrary_chosen_other_class,
            random_seed = 4243)
        
        cf_df = cf_item.cf_examples_list[0].final_cfs_df.drop("op_intent",axis =1)
        
        # Safety check
        if cf_df is None or cf_df.shape[0] == 0:
            continue 
        # choose the closest CF among the 5
        comp_ds = pd.concat([cf_df,instance_item],axis =0 )
        dist_matrix_cf = fc.gower_distance_matrix_fast(comp_ds, 
                                                    categorical_cols = CATEGORICAL_COLS ,
                                                    ordinal_cols = list(ORDINAL_COLS2.keys()) , 
                                                    continuous_cols = list(CONTINUOUS_COLS2.keys()), 
                                                    ordinal_ranges = ORDINAL_COLS2,
                                                    continuous_ranges = CONTINUOUS_COLS2)
        idx_instance = dist_matrix_cf.shape[0] -1
        dist_cf_to_x = dist_matrix_cf[idx_instance,:idx_instance]
        idx_closest = np.argmin(dist_cf_to_x)
        canonical_cf = cf_df.iloc[[idx_closest]]
        #canonical_cfs.append(canonical_cf)
        
        # Measure and stock the change
        change_vec = (canonical_cf.values[0] != instance_item.values[0])
        changed_vectors.append(change_vec)
    
    # Now we have all the canonical_cf for the batch.
    similarities.append(mean_pairwise_jaccard(changed_vectors))
    
avg_dice_similarity = np.mean(similarities) #
std_dice_similarity = np.std(similarities) #
print(f"DiCE similarity : {round(avg_dice_similarity,1)} with standard deviation {std_dice_similarity}") #0.4
    

#%% Use the f9 score metric

k_sim, k_sim_std = metrics.f9_score(kernel_explanations, euclidean) 
l_sim, l_sim_std = metrics.f9_score(lime_explanations, euclidean)

similarity = {"KernelSHAP": k_sim,#0.1 # from 0.2 => 0.1
              "LIME": l_sim, #0.1 # from 0.2 => 0.1
              "DiCE": 0.4, #0.4
              "OSDT":"??"} # results see below 

similarity

#%% Identity check SHAP and LIME
'''
the degree to
which the generated explanations remain consistent across different runs of the XAI method for
the same instance, i.e., identical input values rather than similar ones as in F9.1.
'''


exp_list_kernel = []
exp_list_lime = []
# Random sample to test
individuals_idx = X_dice_ds[mask_CAQ].sample(20, random_state = 4243).index.values


# The number of repetitions
r = 10 # Was 10 in their papers

for i in individuals_idx:
    
    instance_expl_kernel = []
    instance_expl_lime = []
    for j in range(r):
        
        seed(int(4243 + i + j))
        vec = np.zeros(len(X_lime.columns.tolist()),dtype = float)

        #-------SHAP
        expl_shap = kernel_shap_explainer(X_dice_ds.loc[[i]], silent= True)
        instance_expl_kernel.append(expl_shap.values[:,:,chosen_class][0])
        
        #-----LIME
        expl_limeF92 = lime_explainer.explain_instance(X_lime.loc[i].to_numpy(),mod.predict, top_labels=1)
        
        pred_labelF92 = expl_limeF92.top_labels[0]
        #lime_val =[v for _, v in expl_indiv_lime.as_list(label = pred_labelF9)]
        lime_explF92 = [ v for _,v in expl_limeF92.as_list(label = pred_labelF92)]
        
        # aligns LIME
        for feat_idx,w in expl_limeF92.local_exp[pred_labelF92] :#.as_list(label = pred_labelF9):
            vec[feat_idx] = w 
        
        # append instance result
        instance_expl_lime.append(vec)
    
    # Append the list of explanations for this instance
    exp_list_kernel.append(instance_expl_kernel)
    exp_list_lime.append(instance_expl_lime)

#%% download it too
with open( path  + "/00. FINALS/kernel_explanations_bg50_identity_s20_r10.pkl", "wb") as f:
    pickle.dump(exp_list_kernel, f)
    
with open( path  + "/00. FINALS/lime_explanations_bg50_identity_s20_r10.pkl", "wb") as f:
    pickle.dump(exp_list_lime, f)


# =============================================================================
# #------ to load it 
# with open(path  + "/00. FINALS/kernel_explanations_bg50_identity_s20_r10.pkl", "rb") as f:
#     exp_list_kernel = pickle.load(f)
#     
# with open(path  + "/00. FINALS/lime_explanations_bg50_identity_s20_r10.pkl", "rb") as f:
#     exp_list_lime = pickle.load(f)
# =============================================================================
    
#%% Dice identity metric computation 


def mean_pairwise_gower(cfs, gower_fn):
    dists = []
    n = len(cfs)
    for i in range(n):
        for j in range(i + 1, n):
            comp = pd.concat([cfs[i], cfs[j]], axis=0)
            D = gower_fn(
                comp,
                categorical_cols=CATEGORICAL_COLS,
                ordinal_cols=list(ORDINAL_COLS2.keys()),
                continuous_cols=list(CONTINUOUS_COLS2.keys()),
                ordinal_ranges=ORDINAL_COLS2,
                continuous_ranges=CONTINUOUS_COLS2
            )
            dists.append(D[0, 1])
    if len(dists) == 0:
        return np.nan
    return np.mean(dists)


# set the number of repetition per sample
R = 10

individuals_idx = X_dice_ds[mask_CAQ].sample(20, random_state = 4243).index.values
  
test = X_dice_ds.loc[individuals_idx]
canonical_cfs = {idx: [] for idx in individuals_idx}
changed_vec = {idx: [] for idx in individuals_idx}
for r in range(R):
    
    cf_batch = dice.generate_counterfactuals(
        query_instances = test,
        total_CFs = 5,
        desired_class= arbitrary_chosen_other_class,
        random_seed = 4243 + r)
    
    # Transform into a dataframe
    
    for i, idx in enumerate(individuals_idx):
        
        cf_ds_id = cf_batch.cf_examples_list[i].final_cfs_df.drop("op_intent",axis =1)
        
        # Check if cf were generated
        if cf_ds_id is None or cf_ds_id.shape[0] == 0:
            continue
        
        instance_item = test.iloc[[i]]
        comp_ds = pd.concat([cf_ds_id, instance_item], axis=0)
        dist_matrix_cf = fc.gower_distance_matrix_fast(comp_ds, 
                                                    categorical_cols = CATEGORICAL_COLS ,
                                                    ordinal_cols = list(ORDINAL_COLS2.keys()) , 
                                                    continuous_cols = list(CONTINUOUS_COLS2.keys()), 
                                                    ordinal_ranges = ORDINAL_COLS2,
                                                    continuous_ranges = CONTINUOUS_COLS2)
        
        idx_instance = dist_matrix_cf.shape[0] -1
        dist_cf_to_x = dist_matrix_cf[idx_instance,:idx_instance]
        idx_closest = np.argmin(dist_cf_to_x)
        canonical_cf = cf_ds_id.iloc[[idx_closest]]
        canonical_cfs[idx].append(canonical_cf)
        
        # Store the changed vector that will serve as representation
        original_input = X_dice_ds.loc[[idx]]
        changed_vec[idx].append(canonical_cf.values[0] != original_input.values[0])
        
        
# compute one score
identity_set_scores = []

for idx in individuals_idx:
    vecs = changed_vec[idx]
    if len(vecs) < 2:
        continue  # need at least 2 runs

    id_set = mean_pairwise_jaccard(vecs)
    identity_set_scores.append(id_set)

F9_2_feature_identity = np.nanmean(identity_set_scores)

# Compute the other one 
identity_action_scores = []

for idx in individuals_idx:
    cfs = canonical_cfs[idx]
    if len(cfs) < 2:
        continue

    mean_dist = mean_pairwise_gower(cfs, fc.gower_distance_matrix_fast)
    id_action = 1 / (1 + mean_dist)
    identity_action_scores.append(id_action)

F9_2_action_identity = np.nanmean(identity_action_scores)

dice_F9_2_identity = round(0.5 * F9_2_feature_identity + 0.5 * F9_2_action_identity,1) #0.9


#%% I can use the function

met_k, std_k = metrics.f9_score(exp_list_kernel, euclidean,metric = "identity")# 0.2

# maybe need some modification for lime
met_l, std_l = metrics.f9_score(exp_list_lime, euclidean,metric = "identity") #0.1

#%% Some plots would be nice
metrics.scatter_feature_values(exp_list_kernel, instance_idx = [3], metric="identity")
# not so good, let's make them myself later on.
#%% F9 RESULTS

similarity = {"KernelSHAP" : 0.1, "LIME" :0.1 ,"DiCE": 0.4,"OSDT":0.8}
identity = {"KernelSHAP" : 0.2, "LIME": 0.2 ,"DiCE": 0.9,"OSDT":1}

total_stability = {key: similarity[key] + identity[key] for key in similarity}
f9 = pd.DataFrame({
    "Similarity (F9.1)": [similarity["LIME"], similarity["KernelSHAP"], similarity["DiCE"],similarity["OSDT"]],
    "Identity (F9.2)": [identity["LIME"], identity["KernelSHAP"], identity["DiCE"],identity["OSDT"]],
    "Total": [total_stability["LIME"], total_stability["KernelSHAP"], total_stability["DiCE"],total_stability["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f9

#%% F10 CERTAINTY
# ------------------------------------------------
# ------------------------- F10 Certainty
#------------------------------------------------

c1 = {"LIME": 1, "KernelSHAP": 1, "DiCE": 1,"OSDT": 0}
# All methods report confidence in the black-box model's output - display the probability of the outcome (for classification)
c2 = {"LIME": 0, "KernelSHAP": 0, "DiCE": 0,"OSDT":0}
# None of them report explanation confidence values
c3 = {"LIME": 0, "KernelSHAP": 0, "DiCE": 0,"OSDT": 1}
# LIME and kernelSHAP are not deterministic, and this is not disclosured with the explanation
# treeSHAP displays feature perturbation method
c4 = {"LIME": 0, "KernelSHAP": 0, "DiCE": 0,"OSDT":0}
# None of them shows instance distribution relatively to training set
c5 = {"LIME": 1, "KernelSHAP": 0,"DiCE": 0,"OSDT":0}
# LIME can indicate the r2 score, even it is bad it shows the "wrong" explanation
# kernelSHAP provides kernel weights

# Combine all into a list of dictionaries
categories = [c1, c2, c3, c4, c5]

# Sum up values for each method
final_metric_f10 = {key: sum(cat.get(key, 0) for cat in categories) for key in c1.keys()}

f10 = pd.DataFrame({
    "Total": [final_metric_f10["LIME"], final_metric_f10["KernelSHAP"],final_metric_f10["DiCE"],final_metric_f10["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f10

#%% F11 Speed
# ------------------------------------------------
# ------------------------- F11 Speed
#------------------------------------------------

# =============================================================================
# #------------------ SHAP
# =============================================================================

start_time = time()  # Start time
# Initialize SHAP

kernel_shap_explainer = shap.KernelExplainer(
    predict_fn,
    X_dice_ds.sample(50, random_state = 4243), # inititally 50 points but I need to change it=> NO YOU DONT
)
# Kernel SHAP explanation
instance = X_dice_ds.sample(1, random_state = 4243)
kernel_shap_values = kernel_shap_explainer(instance)
shap.initjs()
# Showing explanation for diabetes probability
shap.force_plot(kernel_shap_explainer.expected_value[1], kernel_shap_values.values[:,:,1], instance, feature_names=X_dice_ds.columns)

end_time = time()  # End time
runtime_kernel = end_time - start_time  # Compute runtime
print(f"Runtime kernelSHAP: {runtime_kernel:.6f} seconds")

# =============================================================================
# #------------------ LIME
# =============================================================================


start_time = time()

# initialize LIME
X_lime_continuous = ["age","income","educ",
                     "act_VisitsMuseumsGaleries","act_Fishing","act_Hunting","act_MotorizedOutdoorActivities",
                     "act_Volunteering","voting_probability",
                     "lat_scaled","long_scaled"]

X_lime_categorical = [ col for col in X_lime.columns.values if col not in X_lime_continuous]

# categorical_features => list of indices corresponding to categrical columns
# categorical_names => map from in to list of names
map_cat_names = { i:col for i,col in enumerate(X_lime.columns.values) if col in X_lime_categorical}

map_cat_names2 = { i:np.array(X_lime[col]) 
                  for i,col in enumerate(X_lime.columns.values) 
                  if col in X_lime_categorical
                  }
categorical_names = {}

for col in X_lime_categorical:
    
    idx = X_lime.columns.get_loc(col)

    # Sanity check: binary feature

    # Build human-readable names
    #base_name = col.split("_", maxsplit = 1)  # optional cleanup
    nunique = X_lime[col].nunique()
    stock = {} # [0]*nunique
    
    stock[0] = f"no_{col}"
    stock[1] = col

    # LIME expects a LIST where index == value
    categorical_names[idx] = stock

raw["month"].value_counts()
# month 
categorical_names[149] = {9 : "sep",10:"oct"}

#---- Map the 0,1 to categorical names

for key, val in map_cat_names2.items():
    map_cat_names2[key] =  np.vectorize(categorical_names[key].get)(val)  

lime_explainer = LimeTabularExplainer( training_data = X_lime.to_numpy(), # need numpy + no string. Encoded DS
                                      
                                      mode='classification', 
                                      
                                      # This I still need to change
                                      class_names =["CAQ","PQ","PLQ","QS","PCQ"], # same order as the model output
                                      
                                      categorical_features= list(map_cat_names.keys()), # need list of int
                                      categorical_names= map_cat_names2,#•categorical_names,#map_cat_names2,#categorical_names ,#map_cat_names,
                                      
                                      feature_names=X_lime.columns.tolist(), 
                                      
                                      discretize_continuous = False, # to be avoided for Ordinal var.
                                      random_state = 4243)

#--- test it once
instance_lime = X_lime.sample(1, random_state = 4243)
lime_explanation = lime_explainer.explain_instance(instance_lime.iloc[0].to_numpy(), mod.predict,num_features=7)

end_time = time()

runtime_lime = end_time - start_time
print(f"Runtime LIME: {runtime_lime:.6f} seconds")

# =============================================================================
# #------------------ DiCE
# =============================================================================
start_time = time()


#---------------- Initialization
from dice_ml import Data, Model, Dice

# function and class already imported as fc

nn_features = X_test.drop(columns = ["absolute_idx"]).columns.tolist()

# Define the wrapper.
Dice_wrapper = fc.DiceNNWrapper(mod, nn_features)
    
# Define the dataset
dice_ds,dice_dic_idx, continuous_col,categories_col = fc.generate_dice_ds(raw)
# dice_ds and X_test are aligned since raw and X_test were aligned

dice_model = Model(
    model=Dice_wrapper,
    backend="sklearn"
)
# Define dice data

dice_data = Data(
    dataframe = dice_ds,
    continuous_features = continuous_col,
    categorical_features = categories_col,
    outcome_name ="op_intent"
    )

# Create the DiCE explainer
dice = Dice(
    dice_data,
    dice_model,
    method = "random"
    )
#--------------- Testing 
instance_test_dice = dice_ds.drop(columns = "op_intent").sample(1, random_state = 4243)
predict_fn(instance_test_dice) #4

counterfactual = dice.generate_counterfactuals(
    query_instances=instance_test_dice,
    total_CFs=3,
    desired_class=0,
    random_seed = 4243)

cf_test = counterfactual.cf_examples_list[0].final_cfs_df.drop("op_intent",axis = 1)
counterfactual.visualize_as_dataframe(show_only_changes=True)



end_time = time()

runtime_dice = end_time - start_time
print(f"Runtime DiCE : {runtime_dice:.6f} seconds")

#%% RESULT F11 SPEED
speed = {"LIME": 3, # 3 : metrics.calculate_speed_score(runtime_lime)
         "KernelSHAP": 1,# 1 : metrics.calculate_speed_score(runtime_kernel), 
         "DiCE": 2, # 2 : metrics.calculate_speed_score(runtime_dice)
         "OSDT":0}

f11 = pd.DataFrame({
    "Total": [speed["LIME"], speed["KernelSHAP"],speed["DiCE"],speed["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f11


#%% Table to latex
#F2
fc.get_latex_format(f2, "F2 Structure scores", "f2_structure")

#F3
fc.get_latex_format(f3, "F3 Selectivity scores ")

#F4
fc.get_latex_format(f4, "F4 Contrastivity scores")

#F5
f5
fc.get_latex_format(f5, "F5 Interactivity scores")

#F6
f6
fc.get_latex_format(f6,"F6 Fidelity scores")

#f7
f7
fc.get_latex_format(f7, "F7 Faithfulness scores")

# f8
f8
fc.get_latex_format(f8, "F8 Truthfulness scores")


#f9
f9
fc.get_latex_format(f9, "F9 Stability scores")

#f10
f10
fc.get_latex_format(f10, "F10 (Un)Certainty scores")

# f11
f11
fc.get_latex_format(f11, "F11 Speed scores")
"|l" +"c"*len(f9.columns.tolist())+  "|"


#%% Plots to vizualize everything