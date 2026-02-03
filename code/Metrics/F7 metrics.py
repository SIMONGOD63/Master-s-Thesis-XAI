'''
This files exclusively computes the f7 metrics.
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

# does not work anymore for some reasons
boolean,_ = fc.same_individual(ind_raw, ind_enco)

if boolean:
    print("It works. I can find individuals automatically ")

del(boolean,ind_raw,ind_enco)
#%% Define DICE

from dice_ml import Data, Model, Dice

# function and class already imported as fc

nn_features = X_test.drop(columns = ["absolute_idx"]).columns.tolist()

# Define the wrapper.
Dice_wrapper = fc.DiceNNWrapper(mod, nn_features)
    
# Define the dataset
dice_ds,dice_dic_idx, continuous_col,categories_col = fc.generate_dice_ds(raw)
# dice_ds and X_test are aligned since raw and X_test were aligned

print(dice_dic_idx.values())


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


instance = X_lime.loc[[idx]]
print(y_test.loc[[idx]]) # class 0 => 

#%% F7 Incremental Deletion
# ------------------------------------------------
# ------------------------- F7 FAITHFULNESS
#------------------------------------------------
#----- Incremental Deletion
'''
How the progressive removal of input features identified as relevant by the method impacts the 
predictive model's output f
'''
# Code it by hand to see what it does 
from sklearn.metrics import auc
# get_probs AUC

# ====================================================================================================
# ====================================================================================================
# # IN THE FOLLOWING SECTION WE CREATE EVERYTHING NECESSARY TO BUILD INCREMENTAIL DELETION
# ====================================================================================================
# ====================================================================================================

# Select 30 instances where class 1 proba > 0.6 (can be any class) + correct idx
string_col_name = "proba " + "C" + str(chosen_class)
mask_CAQ = (y_comp["correct"] == chosen_class) & (y_comp["true class"] == 1) & (y_comp[string_col_name] >= 0.6)

sum(mask_CAQ) # only 18

selected_instance_idx = (y_comp.loc[mask_CAQ,"absolute_idx"].sample( np.minimum(30,sum(mask_CAQ)), random_state = 4243))
selected_instance_idx = selected_instance_idx.map(dice_dic_idx).tolist()
# Could merge the two precedings lines. Just there to debug
#print(selected_instance_idx.values)

if (np.minimum(30,sum(mask_CAQ)) < 30):
    print(f"The number of individuals selected is {np.minimum(30,sum(mask_CAQ))}")

selected_instances_kernel = X_dice_ds.loc[selected_instance_idx]
selected_instances_lime  = X_lime.loc[selected_instance_idx]
#### Get the base instance for class 0 (CAQ) => align with chosen_class

chosen_class_name = conv_class[chosen_class]
mask_CAQ_global = y_dice_ds["op_intent"] == chosen_class_name

base_instance_kernel = X_dice_ds[mask_CAQ_global].mode().loc[[0]]
base_instance_LIME = X_lime[mask_CAQ_global].mode().loc[[0]]

# idk if the reshaping is necessary tho
#base_instance_array_kernel = np.array([base_instance_kernel[feature] for feature in X_dice_ds.columns]).reshape(1, -1)
base_instance_array_LIME = np.array([base_instance_LIME[feature] for feature in X_lime.columns]).reshape(1, -1)

instance_test_kernel = selected_instances_kernel.loc[[selected_instance_idx[0]]]
instance_test_lime = selected_instances_lime.loc[[selected_instance_idx[0]]]


######## Test kernel
kernel_shap_test = kernel_shap_explainer(instance_test_kernel)
kernel_shap_feature_ranking = np.argsort(-np.abs(kernel_shap_test.values[:,:,chosen_class][0]))  # Ranking by absolute Shapley values
kernel_feature_ranking = [X_dice_ds.columns[index] for index in kernel_shap_feature_ranking]

##### Test LIME
lime_test = lime_explainer.explain_instance( instance_test_lime.iloc[0].to_numpy(), mod.predict, top_labels=1, num_features= 35 )
# Extract the feature indices
feature_indices = [index for index, _ in lime_test.as_map()[chosen_class]]
lime_feature_ranking = [X_lime.columns[index] for index in feature_indices]

#%% TEST KERNEL : Perform Incremental Deletion  (NOT CORE)
# it mimics the function metrics.f7_1

perturbed_instance = instance_test_kernel.copy()

# For NN predict_proba(.) => predict(.), in my case simply perdict_fn(data_raw_format)


probabilities = [predict_fn(perturbed_instance)[0][chosen_class]]  # Initial probability

for feature in kernel_feature_ranking:
    
    # Pick the mode value for this class to remove
    L = X_dice_ds[feature].unique().tolist()
    L.remove(base_instance_kernel[feature].values[0])
    
    # Choice the value that will perturb my point away from optimal class value for that feature
    seed(4243)
    change_to = sample(L,1)
    
    # Change the value in perturbted_instance
    perturbed_instance[feature] = change_to[0]
        
    # Predict the probability for class 1
    prob = predict_fn( perturbed_instance)[0][chosen_class]
    probabilities.append(prob)
    

# Calculate AUC for probability decay
auc_score = auc(range(len(probabilities)), probabilities)

#%% TEST LIME : INCREMENTAL DELETION (NOT CORE just a draft)
perturbed_instance = instance_test_lime.copy()
perturbed_instance = perturbed_instance.iloc[0].values.reshape(1, -1)
perturbed_instance.shape

probabilities = [mod.predict(perturbed_instance)[0][chosen_class]]  # Initial probability
#def f7_1_get_probs_auc_lime(model, instance, base_instance, feature_ranking, X,classe):


for feature in lime_feature_ranking:
    
    # Replace the feature with its optimal value
    feature_index = X_lime.columns.get_loc(feature)
    
    # perturn the base feature
    if base_instance_array_LIME[0][feature_index]  == 0 :
        perturbed_instance[0][feature_index] = 1
    else:        
     perturbed_instance[0][feature_index] = 0
    
    # Predict the probability for class 1
    
    prob = mod.predict(perturbed_instance.reshape(1, -1))[0][chosen_class]
    probabilities.append(prob)

# Calculate AUC for probability decay
auc_score = auc(range(len(probabilities)), probabilities)
print(auc_score)

lime_probs, lime_auc = metrics.f7_1_get_probs_auc_lime(mod, instance_test_lime, base_instance_array_LIME, lime_feature_ranking , X_lime, chosen_class)

#%% This would be the normal use of the function
kernel_probs, kernel_auc = metrics.f7_1_get_probs_auc( predict_function= predict_fn, instance = instance_test_kernel,
                                                      base_instance= base_instance_kernel ,feature_ranking= kernel_feature_ranking,
                                                      classe= chosen_class, data_X= X_dice_ds)

lime_probs, lime_auc = metrics.f7_1_get_probs_auc_lime(mod, instance_test_lime,
                                                       base_instance_array_LIME, lime_feature_ranking,
                                                       X_lime, chosen_class)

instance_test = base_instance_kernel.copy()

seed(4243)
random_feature_ranking = np.random.permutation(X_dice_ds.columns).tolist()
random_probs, random_auc = metrics.f7_1_get_probs_auc(predict_function= predict_fn, instance=instance_test , 
                                                      base_instance=base_instance_kernel, feature_ranking= random_feature_ranking , 
                                                      classe= chosen_class,data_X= X_dice_ds )
#------ Plot the results
# Plot probability decay curves
plt.plot(range(len(lime_probs)), lime_probs, label='LIME', marker='o')
plt.plot(range(len(kernel_probs)), kernel_probs, label='KernelSHAP', marker='x')
plt.plot(range(len(random_probs)), random_probs, label='Random Explainer', marker='d')
#plt.plot(range(len(dice_probs_instance[0][0])), dice_probs_instance[0][1], label ='DiCE',marker ="v")


# Customize plot
plt.xlabel('Number of features replaced', fontsize = 12)
plt.ylabel(f'Predicted probability for class {chosen_class}',fontsize = 12)
plt.title('Prob. Decay: SHAP,DiCe and LIME', fontweight = "bold",fontsize = 15 )
plt.legend()
plt.grid(True)
plt.show()
#%% Now do it for all the samples

# Initialize lists to store AUC differences for each method
kernel_scores_auc = []
lime_scores_auc = []
# Loop over all instances in selected_instances
for index, row in selected_instances_kernel .iterrows():
    
    instance_test_kernel = X_dice_ds.loc[[index]]
    
    
    # KernelSHAP feature ranking
    kernel_shap_test = kernel_shap_explainer(instance_test_kernel, silent=True)
    kernel_shap_feature_ranking = np.argsort(-np.abs(kernel_shap_test.values[:,:,1][0]))
    kernel_feature_ranking = [X_dice_ds .columns[index] for index in kernel_shap_feature_ranking]

    # random 
    seed(4243 + index) 
    random_feature_ranking = np.random.permutation(X_dice_ds.columns).tolist()
    
    
    random_probs, random_auc = metrics.f7_1_get_probs_auc(predict_function= predict_fn, instance=instance_test_kernel  , 
                                                          base_instance=base_instance_kernel, feature_ranking= random_feature_ranking , 
                                                          classe= chosen_class,data_X= X_dice_ds )


    # Perform Incremental Deletion
    kernel_probs, kernel_auc = metrics.f7_1_get_probs_auc(predict_function= predict_fn, instance = instance_test_kernel ,
                               base_instance= base_instance_kernel,feature_ranking= kernel_feature_ranking,
                               classe= chosen_class, data_X= X_dice_ds)
    # Calculate normalized F7.1 scores
    kernel_scores_auc.append(metrics.f7_1_result(kernel_auc, random_auc))
    
# COmpute the Incremental Deletion f7.1 score
incremental_deletion_kernel = round(np.mean(kernel_scores_auc),1)
incremental_deletion_kernel_std = np.std(kernel_scores_auc)

print("Incremental Deletion F7.1 Score (KernelSHAP):", incremental_deletion_kernel) # 0.6 okay resutlt !

# range :
    # Score 0: : The XAI method performs no better than random feature selection.
    # Score 1: The XAI method perfectly identifies the most impactful features,
    #          leading to a steeper probability decay than the random explainer.

    
#%% DiCE Incremental deletion 
dice_auc_score = []
auc_dice = []
probs_dice = []
for index, row in selected_instances_kernel .iterrows():
    
    instance_test_dice = X_dice_ds.loc[[index]]
    classes = [0,1,2,3,4]
    classes.remove(chosen_class)
    auc_instance_lvl = []
    dice_probs_instance = []
    
    seed(4243 + 4 + index) 
    random_feature_ranking = np.random.permutation(X_dice_ds.columns).tolist()
    random_probs, random_auc_dice = metrics.f7_1_get_probs_auc(predict_function= predict_fn, instance=instance_test_dice  , 
                                                          base_instance=base_instance_kernel, feature_ranking= random_feature_ranking , 
                                                          classe= chosen_class,data_X= X_dice_ds )
    
    for c in classes:
        
        counterfactualF7 = dice.generate_counterfactuals(
                query_instances=instance_test_dice,
                total_CFs=2,
                desired_class=c,
                random_seed = 4243)
        
        # Now I generate a feature ranking 
        cf = counterfactualF7.cf_examples_list[0].final_cfs_df.drop("op_intent",axis = 1)
        
        # Select the features that changed
        #feature = cf.columns.tolist()[0]
        ranked_features_dice1 = [feature for feature in cf.columns.tolist() if cf[feature].values[0] != instance_test_dice[feature].values[0] ]
        ranked_features_dice2 = [feature for feature in cf.columns.tolist() if cf[feature].values[1] != instance_test_dice[feature].values[0] ]

        other_features1 = [feature for feature in cf.columns.tolist() if feature not in ranked_features_dice1 ]
        other_features2 = [feature for feature in cf.columns.tolist() if feature not in ranked_features_dice2 ]

        ranked_features_dice1.extend(other_features1)
        ranked_features_dice2.extend(other_features2)
        
        #---- now the get the probability
        probs1 = [ predict_fn(cf.iloc[0])[0][c] ]
        probs2 = [ predict_fn(cf.iloc[1])[0][c] ]
        
        perturbed_instance_dice1 = cf.iloc[0].copy()
        perturbed_instance_dice2 = cf.iloc[1].copy()
        
        for feat in ranked_features_dice1:
            
            perturbed_instance_dice1[feat] = instance_test_dice[feat].values[0]  # revert
            probs1.append( predict_fn(perturbed_instance_dice1)[0][c])
            
        for feat in ranked_features_dice2:
            
            perturbed_instance_dice2[feat] = instance_test_dice[feat].values[0]  # revert
            probs2.append( predict_fn(perturbed_instance_dice2)[0][c])
        
        
        dice_probs_instance.append((probs1,probs2))
        auc_dice = (auc(range(len(probs1)),probs1) + auc(range(len(probs2)),probs2))/2
        
        auc_instance_lvl.append(auc_dice)
        
    
    #auc_dice.append(np.mean(auc_instance_lvl))
    #probs_dice.append(dice_probs_instance)
    
    final_auc = np.mean(auc_instance_lvl)
    dice_auc_score .append(metrics.f7_1_result(final_auc, random_auc_dice))
    
dice_probs_instance[0][0]
# COmpute the Incremental Deletion f7.1 score
incremental_deletion_dice = round(np.mean(dice_auc_score ),1)
print("Incremental Deletion F7.1 Score (DiCE):", incremental_deletion_dice) #0.5

#%% LIME for all selected instances
del(instance_test_lime)
del(index)
del(row)

lime_scores_auc = []
for index, row in selected_instances_lime .iterrows():
    
    instance_test_lime = X_lime.loc[[index]]
    
    
    #---- Lime feature ranking
    lime_test = lime_explainer.explain_instance( instance_test_lime.iloc[0].to_numpy(), mod.predict, top_labels=1, num_features=X_lime.shape[1])
    
    # Extract the feature indices
    feature_indices = [index for index, _ in lime_test.as_map()[chosen_class]]
    lime_feature_ranking = [X_lime.columns[index] for index in feature_indices]
    
    # Random feature ranking
    seed(4243 + index)
    random_feature_ranking = np.random.permutation(X_lime.columns).tolist()
    # def f7_1_get_probs_auc_lime(model, instance, base_instance_array, feature_ranking, X,classe):

    random_probs, random_auc = metrics.f7_1_get_probs_auc_lime(model = mod, instance=instance_test_lime , 
                                                          base_instance_array=base_instance_array_LIME ,
                                                          feature_ranking = random_feature_ranking , 
                                                          X = X_lime,classe= chosen_class)
    #---- Perform Incremental Deletion
    lime_probs, lime_auc = metrics.f7_1_get_probs_auc_lime(mod, instance_test_lime,
                                                           base_instance_array_LIME, lime_feature_ranking,
                                                           X_lime, chosen_class)
        
    lime_scores_auc.append(metrics.f7_1_result(lime_auc, random_auc))

# final result
incremental_deletion_lime = round(np.mean(lime_scores_auc),1)
print(incremental_deletion_lime) # 0.8 
print("Incremental Deletion F7.1 Score (LIME):", incremental_deletion_lime)

#%% SOLUTION MA GUEULE

incremental_deletion = {"KernelSHAP": incremental_deletion_kernel,"LIME" : incremental_deletion_lime}
print(incremental_deletion)

#%% F7.2 ROARf
'''
=> Mostly for GLOBAL explanation technique
The impact of feature deletion based on the relevance defined the XAI method
on the model accuracy AFTER RETRAINING
'''
roar = {"KernelSHAP": None,"LIME": None,"DiCE":None}

#%% F7.3 WHITE BOX CHECK (setup)
'''
This is the code re-used from the framework example of the Paper :
    "A Functionally-Grounded Benchmark Framework for XAI
    Methods: Insights and Foundations from a Systematic
    Literature Review"
Can be found here : https://github.com/DCanha/FUNCXAI-11/blob/main/framework_example.ipynb
I'm not trying to appropriate work that isn't mine. I'm simply reusing results and write down it for the
transparency of my thesis.
'''
# =============================================================================
# =============================================================================
# #-------------------  F7.3 WHITE BOX CHECK
# =============================================================================
# =============================================================================
# Define the linear function
def linfunc(inputs):
    """
    Linear function: Weighted sum of inputs.
    Coefficients: 0.4, 0.3, 0.2, 0.1 # just an example, it can be changed
    """
    if isinstance(inputs, pd.DataFrame):
        X = inputs.values
    else:
        X = inputs
    return 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * X[:, 3]

# Generate the synthetic dataset
x1 = x2 = x3 = x4 = np.linspace(0, 1, 21)  # Sequence from 0 to 1, step 0.05
pm = np.array(np.meshgrid(x1, x2, x3, x4)).T.reshape(-1, 4)  # Create the grid

# Ground-truth outputs
y = linfunc(pm)


# Define instances to compare with explanations
c = np.array([[1, 0.66, 0.33, 0]])        # Specific instance
c1 = np.array([[0.5, 0.5, 0.5, 0.5]])     # All values are average
c2 = np.array([[1, 1, 1, 1]])             # All values are maximum

# Define the linear function coefficients
coefficients = [0.4, 0.3, 0.2, 0.1]
E_X = np.mean(y)  # 0.5 - Average prediction for all features

# Define a function to calculate feature influence
def compute_influence(coefficients, feature_values, E_X=0.5):
    """
    
    Basically it computes :
        "how much each feature contributes to moving the prediction 
        away from the average-input baseline.""
    
    #------------------------------------------------------
    Compute feature influence for an instance. - from decision theory - more in https://doi-org.proxy.bnl.lu/10.1007/978-3-031-44064-9_14
    
    Parameters:
        coefficients (list): Linear coefficients for the features (w_i).
        feature_values (list): Values of the features for a given instance (x_i).
        E_X (float): Expected value of each feature (default is 0.5).
    
    Returns:
        list: Feature influence values.
    """
    return [w * x - w * E_X for w, x in zip(coefficients, feature_values)]

# Define a function to construct the table
def construct_table(instance, coefficients, xai_explanations, feature_names):
    """
    Construct a table with feature values, global importance, feature influence, and XAI coefficients.
    
    Parameters:
        instance (array): Feature values for the instance (x_i).
        coefficients (list): Linear coefficients for the features (w_i).
        xai_explanations (dict): Explanation coefficients from different XAI methods (e.g., LIME, SHAP).
        feature_names (list): Names of the features.
    
    Returns:
        DataFrame: Table summarizing the values.
    """
    # Compute feature influence
    influences = compute_influence(coefficients, instance)
    
    # Create the table
    data = {
        "Feature": feature_names,
        "Feature Value (x_i)": instance,
        "Global Importance (w_i)": coefficients,
        "Feature Influence": influences,
    }
    
    # Add XAI coefficients
    for method, explanation in xai_explanations.items():
        data[f"{method} Coefficient"] = explanation
    
    return pd.DataFrame(data)

#%%# Create explainers for linear function
lime_explainer_linear = LimeTabularExplainer(pm, feature_names=["x1", "x2", "x3", "x4"], mode='regression')
kernel_shap_explainer_linear = shap.KernelExplainer(linfunc, shap.sample(pm, 1000)) # I am sampling to reduce the running time - limitation in kernelSHAP!

# For instance c:
lime_exp_c = lime_explainer_linear.explain_instance(c[0], linfunc)
lime_explanation_c = np.array([value for _, value in lime_exp_c.as_list()])  # Extract feature influence
shap_values_c = kernel_shap_explainer_linear.shap_values(c)
kernelshap_explanation_c = shap_values_c[0]

# For instance c1:
lime_exp_c1 = lime_explainer_linear.explain_instance(c1[0], linfunc)
lime_explanation_c1 = np.array([value for _, value in lime_exp_c1.as_list()])  # Extract feature influence
shap_values_c1 = kernel_shap_explainer_linear.shap_values(c1)
kernelshap_explanation_c1 = shap_values_c1[0]

# For instance c2:
lime_exp_c2 = lime_explainer_linear.explain_instance(c2[0], linfunc)
lime_explanation_c2 = np.array([value for _, value in lime_exp_c2.as_list()])  # Extract feature influence
shap_values_c2 = kernel_shap_explainer_linear.shap_values(c2)
kernelshap_explanation_c2 = shap_values_c2[0]

#%% Analyze c
feature_names = ["x1", "x2", "x3", "x4"]
# Explanation for c
xai_explanations_c = {
    "LIME": lime_explanation_c,
    "KernelSHAP": kernelshap_explanation_c
}

# Construct the table
table_c = construct_table(c[0], coefficients, xai_explanations_c, feature_names)
table_c

# for c
print(metrics.f7_3_compute_agreement(lime_explanation_c, table_c["Feature Influence"])) #0.73
print(metrics.f7_3_compute_agreement(kernelshap_explanation_c, table_c["Feature Influence"])) # 0.97


#%% Analyze c1
# Explanation for c1
xai_explanations_c1 = {
    "LIME": lime_explanation_c1,
    "KernelSHAP": kernelshap_explanation_c1
}

# Construct the table
table_c1 = construct_table(c1[0], coefficients, xai_explanations_c1, feature_names)
table_c1
# for c1
print(metrics.f7_3_compute_agreement(lime_explanation_c1, table_c1["Feature Influence"]))
print(metrics.f7_3_compute_agreement(kernelshap_explanation_c1, table_c1["Feature Influence"]))

#%% Analyze c2
# Explanation for c2
xai_explanations_c2 = {
    "LIME": lime_explanation_c2,
    "KernelSHAP": kernelshap_explanation_c2
}

# Construct the table
table_c2 = construct_table(c2[0], coefficients, xai_explanations_c2, feature_names)
table_c2

print(metrics.f7_3_compute_agreement(lime_explanation_c2, table_c2["Feature Influence"]))
print(metrics.f7_3_compute_agreement(kernelshap_explanation_c2, table_c2["Feature Influence"]))

#%% Initiliaze dice
from dice_ml import Data, Model, Dice
feature_names = ["x1", "x2", "x3", "x4"]
#------------------------------------------------
def linfunc_classifier(inputs):
    y = linfunc(inputs)
    return (y > 0.5).astype(int)

def linfunc_proba(inputs):
    y = linfunc(inputs)
    p1 = np.clip(y, 0, 1)
    p0 = 1 - p1
    return np.vstack([p0, p1]).T
#------------------------------------------------
class LinearWrapper:
    def predict(self, X):
        return linfunc_classifier(X)

    def predict_proba(self, X):
        return linfunc_proba(X)
#------------------------------------------------
# Create dataframe for DiCE
df_dice = pd.DataFrame(pm, columns=feature_names)
df_dice["y"] = linfunc_classifier(pm)

dice_data = Data(
    dataframe=df_dice,
    continuous_features=feature_names,
    outcome_name="y"
)

dice_model = Model(
    model=LinearWrapper(),
    backend="sklearn",
    model_type="classifier"
)

dice = Dice(dice_data, dice_model, method="random")

#%% TEST DiCE in this context

query_instance = pd.DataFrame( [pm[1] ], columns=feature_names)

pm[[1]]
df_dice.drop(columns = ["y"]).iloc[[1]].to_numpy()
linfunc(df_dice.drop(columns = ["y"]).iloc[[1]].to_numpy())
linfunc_classifier(df_dice.drop(columns = ["y"]).iloc[[1]].to_numpy())
linfunc_proba(df_dice.drop(columns = ["y"]).iloc[[1]].to_numpy())

cf = dice.generate_counterfactuals(
    query_instance,
    total_CFs=10,
    desired_class=1
)

cf_df = cf.cf_examples_list[0].final_cfs_df


#%% Compare the influence scores for 100 instances
feature_names = ["x1", "x2", "x3", "x4"]

# Select 100 random indices
random_indices = np.random.choice(range(len(pm)), 100, replace=False)

# Initialize lists to store agreements
lime_agreement = []
kernel_agreement = []
dice_agreement = []
for i in random_indices:
    break
    # LIME Explanation
    lime_exp = lime_explainer_linear.explain_instance(pm[i], linfunc)
    lime_explanation = np.array([value for _, value in lime_exp.as_list()])  # Extract feature influence
    
    # KernelSHAP Explanation
    shap_values = kernel_shap_explainer_linear.shap_values(pm[i:i+1], silent = True)  # KernelSHAP expects 2D input
    kernelshap_explanation = shap_values[0]  # Use the first output (assumes single target)
    
    #---------- DiCE
    query_instance = pd.DataFrame([pm[i]], columns=feature_names)

    cf = dice.generate_counterfactuals(
        query_instance,
        total_CFs=10,
        desired_class=1
    )

    cf_df = cf.cf_examples_list[0].final_cfs_df
    dice_relevance = []
    x_orig = query_instance.iloc[0].values
    
    # for each feature, get whether or not its label as "dice relevant"
    for j, feat in enumerate(feature_names):
        
        # the condition is that in more that half cf changed this variables, meaning identify it as relevant in the DiCE sense.
        changed = (cf_df[feat].values != x_orig[j]).sum() >= round((len(cf_df)/2))
        #changed = np.abs(cf_df[feat].values - x_orig[j])
        #dice_explanation.append(changed.mean())
        dice_relevance.append(changed)
        
    dice_relevance = np.array(dice_relevance)
    

    
    # Construct XAI explanations dictionary
    xai_explanations = {
        "LIME": lime_explanation,
        "KernelSHAP": kernelshap_explanation,
    }

    # Construct the table
    table = construct_table(pm[i], coefficients, xai_explanations, feature_names)
    
    # Compute agreement scores and store
    lime_agreement.append(metrics.f7_3_compute_agreement(lime_explanation, table["Feature Influence"]))
    kernel_agreement.append(metrics.f7_3_compute_agreement(kernelshap_explanation, table["Feature Influence"]))
    
    #---- COmpute DiCE metric
    threshold = np.percentile(
        np.abs(table["Feature Influence"]),
        50  # median importance
    )

    true_relevance = np.abs(table["Feature Influence"].values) > threshold

    dice_f7_3 = np.mean(dice_relevance == true_relevance) # proportion of correctly identified relevant feature

    
    dice_agreement.append(dice_f7_3)
    
    
# Compute mean agreement
mean_lime_agreement = np.mean(lime_agreement)
mean_kernel_agreement = np.mean(kernel_agreement)
mean_dice_agreement = np.mean(dice_agreement)

print(f"Mean Agreement for LIME: {mean_lime_agreement:.2f}") #0.62
print(f"Mean Agreement for KernelSHAP: {mean_kernel_agreement:.2f}") #0.91
print(f"Mean Agreement for DiCE: {mean_dice_agreement:.2f}") # 0.53

# =============================================================================
# The final metric
# =============================================================================

white_box = {"LIME": metrics.f7_3_score(mean_lime_agreement), "KernelSHAP": metrics.f7_3_score(mean_kernel_agreement), "DiCE" : metrics.f7_3_score(mean_dice_agreement)}
print(white_box)
#%% FINAL F7 SCORE

# set the correct score without re run
incremental_deletion_lime = 0.8
incremental_deletion_dice = 0.5
incremental_deletion_kernel = 0.6 
incremental_deletion = {"KernelSHAP": incremental_deletion_kernel,"LIME" : incremental_deletion_lime,"DiCE": incremental_deletion_dice}
roar = {"KernelSHAP": None,"LIME": None,"DiCE":None}


total_faithfulness = {key: incremental_deletion[key] + white_box[key] for key in incremental_deletion}
f7 = pd.DataFrame({
    "Incremental Deletion (F7.1)": [incremental_deletion["LIME"], incremental_deletion["KernelSHAP"], incremental_deletion["DiCE"]],
    "ROAR (F7.2)": [roar["LIME"], roar["KernelSHAP"], roar["DiCE"]],
    "White-Box Check (F7.3)": [white_box["LIME"], white_box["KernelSHAP"], white_box["DiCE"]],
    "Total": [total_faithfulness["LIME"], total_faithfulness["KernelSHAP"], total_faithfulness["DiCE"]]
}, index=["LIME", "KernelSHAP", "DiCE"])
f7
#%%
