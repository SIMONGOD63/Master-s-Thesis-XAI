'''
TABLE FILE
'''

#%% IMPORTS

import pandas as pd ; import numpy as np
import os
path2 = r"C:\\Users\\simeo\\A. Data Science\\Master thesis\\02. Code\\Techniques"
os.chdir(path2)

import metrics


#%% GENERATE ALL THE TABLES
#% F1 Metric (FIXED)

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
    "Total_f1": [total_repr["LIME"], total_repr["KernelSHAP"], total_repr["DiCE"],total_repr["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f1

#% F2 Structure (FIXED)

# =============================================================================
# Expressive power
# =============================================================================
C_predefined = {"decision_tree", "text_summary", "causal_diagram", "bar_plot", "analogy", "rule"}

#--------- LIME parameters
n_lime = 1  # One explanatory output: feature influence
F_lime = ["table_summary", "bar_plot"] # unsure about rues It has rule in the terms of if feature_1 < 10, class = 1
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
    "Total_f2": [total_structure["LIME"], total_structure["KernelSHAP"], total_structure["DiCE"],total_structure["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f2

#%F3

s_dice = 0.9
print(s_dice)

s_osdt = 5 # it is the rounded average path length.
'''
The metric compute it in a wrong way IMO. Shouldn't be penalized if s in [5,7]
'''

# Quantify if the number of explanation shown is relevan
explanation_size = {"LIME": metrics.f3(s=8, tunable = True), 
                    "KernelSHAP": metrics.f3(s=8, tunable = True),
                    "DiCE":0.9,
                    "OSDT" : metrics.f3(s = s_osdt, tunable = False)} # set it to true to truly reflect the note IMO
# In this case, both LIME and SHAP allow us to choose a max number of features to display, but there are certain methods that do not allow (e.g., PFI)


f3 = pd.DataFrame({
    "Total_f3": [explanation_size["LIME"], explanation_size["KernelSHAP"], explanation_size["DiCE"],explanation_size["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f3

#% Results F4
target_sensitivity = {"LIME": 0.3, "KernelSHAP": 0.4, "DiCE": 0, "OSDT":0.5}
contrastivity_level = {"LIME": 1, "KernelSHAP": 1, "DiCE": 2,"OSDT":0}
# =============================================================================
# ----- SUMMARIZE
# =============================================================================
total_contrast = {key: contrastivity_level[key] + target_sensitivity[key] for key in target_sensitivity.keys()}
f4 = pd.DataFrame({
    "Contrastivity Level (F4.1)": [contrastivity_level["LIME"], contrastivity_level["KernelSHAP"], contrastivity_level["DiCE"],contrastivity_level["OSDT"]],
    "Target Sensitivity (F4.2)": [target_sensitivity["LIME"], target_sensitivity["KernelSHAP"], target_sensitivity["DiCE"],target_sensitivity["OSDT"]],
    "Total_f4": [total_contrast["LIME"], total_contrast["KernelSHAP"], total_contrast["DiCE"],total_contrast["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f4

#% F5 (FIXED)
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
    "Total_f5": [total_interactivity["LIME"], total_interactivity["KernelSHAP"], total_interactivity["DiCE"],total_interactivity["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f5

#% F6 SUMMARIZE
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
    "Total_f6": [total_fidelity["LIME"], total_fidelity["KernelSHAP"], total_fidelity["DiCE"],total_fidelity["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f6

#% F7 SEE DEDICATED FILE FOR DETAILS
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
    "Total_f7": [total_faithfulness["LIME"], total_faithfulness["KernelSHAP"], total_faithfulness["DiCE"],total_faithfulness["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f7

#% F8 (fixed)
# ------------------------------------------------
# ------------------------- F8 Truthfulness
#------------------------------------------------

# F.8.1 Reality check
'''
Checks whether the method prevents the generation of unrealistic data samples.
Use two sub-metric 
    A => Feature constraints consistency 
    B => Feature Correlation Consistency
'''


reality_check_A = {"LIME": 0, "KernelSHAP": 0,"DiCE" : 1,"OSDT":1}


#---- Correlation check
# also fixed

reality_check_B = {"LIME": 0, "KernelSHAP": 0,"DiCE":0,"OSDT":1}


reality_check = {"LIME": reality_check_A["LIME"] + reality_check_B["LIME"], 
                 "KernelSHAP": reality_check_A["KernelSHAP"] + reality_check_B["KernelSHAP"],
                 "DiCE" :reality_check_A["DiCE"] + reality_check_B["DiCE"],
                 "OSDT" :reality_check_A["OSDT"] + reality_check_B["OSDT"]}

#--------------------------------------------

#---- Bias Detection
# check their code to understand the process they went through
bias_detection = {"LIME": 1, "KernelSHAP": 1,"DiCE" : 1,"OSDT":1}

total_truthfulness = {key: reality_check[key] + bias_detection[key] for key in reality_check}
f8 = pd.DataFrame({
    "Reality Check (F8.1)": [reality_check["LIME"], reality_check["KernelSHAP"],reality_check["DiCE"],reality_check["OSDT"]],
    "Bias Detection (F8.2)": [bias_detection["LIME"], bias_detection["KernelSHAP"],bias_detection["DiCE"],bias_detection["OSDT"]],
    "Total_f8": [total_truthfulness["LIME"], total_truthfulness["KernelSHAP"],total_truthfulness["DiCE"],total_truthfulness["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f8
#%F9 RESULTS

similarity = {"KernelSHAP" : 0.1, "LIME" :0.1 ,"DiCE": 0.4,"OSDT":0.8}
identity = {"KernelSHAP" : 0.2, "LIME": 0.2 ,"DiCE": 0.9,"OSDT":1}

total_stability = {key: similarity[key] + identity[key] for key in similarity}
f9 = pd.DataFrame({
    "Similarity (F9.1)": [similarity["LIME"], similarity["KernelSHAP"], similarity["DiCE"],similarity["OSDT"]],
    "Identity (F9.2)": [identity["LIME"], identity["KernelSHAP"], identity["DiCE"],identity["OSDT"]],
    "Total_f9": [total_stability["LIME"], total_stability["KernelSHAP"], total_stability["DiCE"],total_stability["OSDT"]]
}, index=["LIME", "KernelSHAP", "DiCE","OSDT"])
f9

# F10 CERTAINTY
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
c4 = {"LIME": 0, "KernelSHAP": 0, "DiCE": 1,"OSDT":0}
# None of them shows instance distribution relatively to training set
c5 = {"LIME": 1, "KernelSHAP": 0,"DiCE": 0,"OSDT":0}
# LIME can indicate the r2 score, even it is bad it shows the "wrong" explanation
# kernelSHAP provides kernel weights

# Combine all into a list of dictionaries
categories = [c1, c2, c3, c4, c5]

# Sum up values for each method
final_metric_f10 = {key: sum(cat.get(key, 0) for cat in categories) for key in c1.keys()}

f10 = pd.DataFrame({
    "Total_f10": [final_metric_f10["LIME"], final_metric_f10["KernelSHAP"],final_metric_f10["DiCE"],final_metric_f10["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f10

# RESULT F11 SPEED
speed = {"LIME": 3, # 3 : metrics.calculate_speed_score(runtime_lime)
         "KernelSHAP": 1,# 1 : metrics.calculate_speed_score(runtime_kernel), 
         "DiCE": 2, # 2 : metrics.calculate_speed_score(runtime_dice)
         "OSDT":0}

f11 = pd.DataFrame({
    "Total_f11": [speed["LIME"], speed["KernelSHAP"],speed["DiCE"],speed["OSDT"]]
}, index=["LIME", "KernelSHAP","DiCE","OSDT"])
f11

keep_vars = {
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11","path"
}

for var in list(globals().keys()):
    if var not in keep_vars and not var.startswith("_"):
        del globals()[var]

print("All tables were generated !")
#%% Check the tables' score
import pandas as pd
import numpy as np
# Build a general tables 

tab = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11], axis = 1)

methods_names = tab.index.tolist()

#----- Rename the columns 
import re

def rename_col(col):
    # Keep total columns unchanged
    if col.startswith("Total_"):
        return col
    
    # Match pattern like "(F1.1)", "(F10.2)", etc.
    match = re.search(r"\(F(\d+)\.(\d+)\)", col)
    if match:
        i, j = match.groups()
        return f"f{i}_{j}"
    
    # Otherwise keep original name
    return col


tab = tab.rename(columns=rename_col)


#------ Normalize all the metrics 
tab.loc["Min_score"] = tab.apply(np.min, axis = 0)
tab.loc["Max_score"] = tab.apply(np.max, axis = 0)
tab.drop("f7_2", axis = 1, inplace = True)
normalized_df = tab.drop(["Min_score","Max_score"], axis = 0).copy()



# Normalize the df
for col in  tab.columns.tolist():
    min_val = tab.loc["Min_score", col]
    max_val = tab.loc["Max_score", col]
    
    if max_val == min_val:
        # Non-discriminative metric → neutral score
        normalized_df[col] = 0.5
    else:
        normalized_df[col] = (
            normalized_df[col] - min_val
        ) / (max_val - min_val)

#--- Check for silent NaN
assert normalized_df.isna().any(axis = 1).sum() == 0

del(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11)
#%% Defined what's necessary for plots 
path_image = r"C:\Users\simeo\A. Data Science\Master thesis\03. Image\results"
import matplotlib.pyplot as plt

# fixed data type and black box dependent scores
categories_fixed = ["f1_1", "f1_2", "f1_3", "f1_4", "f5_1", "f5_2", "f6_1", "f8_1", "Total_f10"]
categories_dt = ["f2_1", "f2_2", "f2_3", "f2_4", "f4_1"]
categories_bb = ["Total_f3", "f4_2", "f6_2", "f7_1", "f7_3", "f8_2", "f9_1", "f9_2", "Total_f11"]

# All categories and only subcategories
categories_excluding_total = [col for col in normalized_df.columns if not col.startswith("Total_")]
categories_excluding_total.extend(["Total_f10","Total_f11","Total_f3"])
categories_total_only = [col for col in normalized_df.columns if col.startswith("Total_")]

METHOD_COLORS = {
    "LIME": "#1f77b4",        # blue
    "KernelSHAP": "#ff7f0e",  # orange
    "DiCE": "#2ca02c",        # green
    "OSDT": "#d62728",        # red
}

METHOD_COLORS.get("LIME")
METHOD_COLORS["LIME"]

def plot_spider(df, categories, title,save_path, allowed_methods = ['LIME', 'KernelSHAP', 'DiCE', 'OSDT'], method_colors = None, ):
    
    # Specify fixed colors
    if method_colors is None:
        method_colors = METHOD_COLORS
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for method_name, row in df.iterrows():
        
        if method_name in allowed_methods:
            values = row[categories].values.flatten().tolist()
            values += values[:1]
            color = method_colors[method_name] #.get(method_name,"gray")
            
            ax.plot(angles, values,label = method_name, color = color)
            ax.fill(angles, values, alpha=0.1, color = color)

    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Save the plot as a PDF
    plt.savefig(save_path, bbox_inches="tight",dpi = 300)
    plt.show()
    
#%%

zz_fixed_scored_df = normalized_df[categories_fixed].copy()
zz_dt_scored_df = normalized_df[categories_dt].copy()
zz_bb_scored_df = normalized_df[categories_bb].copy()

plot_spider(normalized_df, categories_fixed, "Fixed Score")


#%% data type
plot_spider(normalized_df, categories_dt, "Tabular Domain Score")

#%% BLACK BOX

plot_spider(normalized_df, categories_bb, "Non-Fixed Score")

#%%  TOTAL ONLY

plot_spider(normalized_df, categories_total_only,title= "Property Total Scores", save_path= path_image +"/Property Total Scores")


#%% SUB PROPERTY ONLY

plot_spider(normalized_df, categories_excluding_total, "Sub-property Scores")

#%% Methods 2 by 2 
#---- KERNELSHAP
["LIME","KernelSHAP"]

#---data type
plot_spider(normalized_df, categories_dt,title = "Tabular Domain Score", allowed_methods= ["LIME","KernelSHAP"],
            save_path= path_image +"/SHAP_LIME_data_score")

#---- black box
plot_spider(normalized_df, categories_bb, "Non-Fixed Score",["LIME","KernelSHAP"])

#TOTAL ONLY
plot_spider(normalized_df, categories_total_only, title = "LIME, SHAP Property Score", allowed_methods = ["LIME","KernelSHAP"],
            save_path= path_image +"/SHAP_LIME_total_score")

# All categories
plot_spider(normalized_df, categories_excluding_total, "Sub-property Scores",["LIME","KernelSHAP"])

#%% OSDT,DiCE
["DiCE","OSDT"]
#---data type
plot_spider(normalized_df, categories_dt, "Tabular Domain Score",["DiCE","OSDT"])

#---- black box
plot_spider(normalized_df, categories_bb, "Non-Fixed Score",["DiCE","OSDT"])

#TOTAL ONLY
plot_spider(normalized_df, categories_total_only, "DiCE,OSDT Property Score", allowed_methods = ["DiCE","OSDT"],
            save_path =path_image +"/DiCE_OSDT_total_score" )

# All categories
plot_spider(normalized_df, categories_excluding_total, "Sub-property Scores",["DiCE","OSDT"])


#%% Individual plot metrics not very interesting, too few different metrics

# not interesting
cat_f1 = [col for col in normalized_df.columns if col.startswith("f1")]

plot_spider(normalized_df, cat_f1, "F1 Property Score",
            save_path =path_image +"/f1_score" )
