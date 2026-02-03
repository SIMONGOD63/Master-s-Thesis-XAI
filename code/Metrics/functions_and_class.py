'''
File dedicated to additional functions used to compute  the 11 metrics.
'''

#%% imports
from random import sample, seed
import pandas as pd
import numpy as np
#---  Define the mapping dictionnonary
MAP_INCOME = {
    "None": 0, "i1to30": 1, "i31to60": 2, "i61to90": 3,
    "i91to110": 4, "i111to150": 5, "i151to200": 6, "i201toInf": 7
}

MAP_EDUC = {
    "None": 0, "Prim": 1, "Sec": 2, "Coll": 3,
    "Bacc": 4, "Master": 5, "PhD": 6
}

MAP_DAY = {"<11": 0, "11-24": 1, ">24": 2}
MAP_AGE = {"34m": 0, "3554": 1, "55p": 2}


MAP_SEXE= {
    "ses_hetero": "hetero",
    "ses_gai" :"gai",
    "ses_bisex": "bisex",
    "ses_sexOri_other": "sexOri_other",
    "ses_Pansexual" : "Pansexual",
    "ses_Asexual" : "Asexual",
    "ses_Queer": "Queer",
    "ses_Questionning":"Questionning"
    }

ORDINAL_MAPS = {
    "educ": {
        "None":0, "Prim": 1, "Sec": 2, "Coll": 3,
        "Bacc": 4, "Master": 5, "PhD": 6
    },
    "income": {
        "None": 0, "i1to30": 1, "i31to60": 2,
        "i61to90": 3, "i91to110": 4,
        "i111to150": 5, "i151to200": 6,
        "i201toInf": 7
    },
    "age":{
        "34m":0, "3554":1, "55p":2
    }
}
# from : dice_ds.columns.tolist()

RAW_FEATURES_NAMES = ['month', 'imp_ind', 'app_noTattoo', 'cons_coffee', 'ses_income',
                      'ses_dwelling', 'ses_educ', 'app_swag', 'music', 'film', 'ses_ethn',
                      'act_transport', 'vehicule', 'cons_Smoke', 'cons_meat', 'cons_brand',
                      'animal', 'day', 'sport', 'alcohol', 'age', 'lang', 'people_predict', 
                      'gender', 'sex_ori', 'act_VisitsMuseumsGaleries', 'act_Fishing', 
                      'act_Hunting', 'act_MotorizedOutdoorActivities', 'act_Volunteering', 'pays_qc',
                      'immigrant', 'lat', 'long', 'voting_probability'] # 'op_intent',

'''
# from : raw.columns.tolist()

['absolute_idx', 'month', 'imp_ind', 'app_noTattoo', 'cons_coffee', 
'ses_income', 'ses_dwelling', 'ses_educ', 'app_swag', 'music', 'film',
'ses_ethn', 'act_transport', 'vehicule', 'cons_Smoke', 'cons_meat', 
'cons_brand', 'animal', 'day', 'sport', 'alcohol', 'age', 'lang', 
'people_predict', 'op_intent', 'gender', 'sex_ori', 'act_VisitsMuseumsGaleries', 
'act_Fishing', 'act_Hunting', 'act_MotorizedOutdoorActivities', 'act_Volunteering', 
'pays_qc', 'immigrant', 'lat', 'long', 'lat_scaled', 'long_scaled', 
'voting_probability']

'''

#%% Dice related

#raw_df = cf_instance
#train_features = X_test.columns.values

def encode_for_nn(raw_df: pd.DataFrame, train_features: list) -> pd.DataFrame:
    """
    Encode a raw semantic DataFrame (dice_ds format) into the exact
    numeric feature space expected by the neural network.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Same schema as dice_ds (semantic features).
    train_features : list
        Ordered list of NN input features (length = 156).

    Returns
    -------
    X_enc : pd.DataFrame
        Encoded DataFrame, columns aligned exactly to train_features.
    """
    
    # -------------------------
    #------ 1. Check datatype phase
    # -------------------------
    
    # --- SHAP compatibility: ndarray → DataFrame
    if isinstance(raw_df, np.ndarray):
        raw_df = pd.DataFrame(raw_df, columns=RAW_FEATURES_NAMES)

    # --- Series → DataFrame (already known)
    if isinstance(raw_df, pd.Series):
        raw_df = raw_df.to_frame().T

    # -------------------------
    #------ 0. Setup phase
    # -------------------------
    
    df = raw_df.copy()
    columns = df.columns
    
    #If series => transform to dataframe
    

    # ----------------------------
    # 1. Ordinal variables
    # ----------------------------
    
    if "day" in columns:
        df["day"] = df["day"].map(MAP_DAY)
        #print("day entered.")
    
    if "sex_ori" in columns:
        df["ses"] = df["sex_ori"].map(MAP_SEXE)
        df.drop(columns= ["sex_ori"],inplace = True)
        #print("sex entered.")
    
    # I don't know if this part is required. => nop already clipped but the created point will need to be clipped
    if "voting_probability" in df.columns:
        df["voting_probability"] = (
            pd.to_numeric(df["voting_probability"], errors="coerce")
            .clip(0, 1)
            .round(1)
        )
        #print("VP entered.")
        
    # ----------------------------
    # 2. Transform some column name to have an aligned pre fixe
    # ----------------------------
        
    rename_dic = {"people_predict" : "people_pred",
                  "alcohol" : "cons",
                  "sport" :"act",
                  "ses_income":"income",
                  "ses_educ":"educ"}

    df.rename(columns = rename_dic,inplace = True)

    # ----------------------------
    # 3. Encode the variable
    # ----------------------------
    df.dtypes
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    df_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)
    df_num = df.drop(columns=cat_cols)
    
    
    
    # ----------------------------
    # 4. Necessary renaming
    # ----------------------------
    
    rename2 = { "lang_Fr":"langFr" ,
               "lang_En":"langEn",
               "gender_female":"female",
               "gender_male":"male",
               "music_genre_": "music_genre_music_other"
        }

    df_cat.rename(columns = rename2,inplace = True)
    
    # ----------------------------
    # 5. Removes the columns I removed by hand (if they match by name)
    # ----------------------------
    cat_to_remove = ['cons_coffee_None', 'ses_dwelling_Other', 'app_swag_Other',
     'music_no', 'film_no', 'ses_ethn_Other', 'act_transport_PublicTransportation', 
     'vehicule_VUS', 'cons_Smoke_few_times_day', 'cons_meat_few_daily', 
     'cons_brand_Other', 'animal_noPet']

    remove_by_hand = ['educCollege', 'educUniv', 'educBHS', 'act_None', 'cons_noDrink',
      'gender_agender', 'gender_trans_female', 'gender_trans_male', 'gender_nonbinary',
       'gender_queer', 'genderOther', 'ses_sexOri_other', 'ses_languageOther','gender_genderOther','lang_Other','people_pred_PCQ']

    removed = []
    for col in df_cat.columns:
        if (col in cat_to_remove) or (col in remove_by_hand):
            df_cat.drop(columns = [col], inplace = True)
            removed.append(col)

    df_num.drop(columns = ["lat","long"], errors="ignore",inplace = True)
    #♦print(f"This step removed {len(removed)} columns") # 15 => 17
    
    # ----------------------------
    # 6. Create X_enco
    # ----------------------------
    
    X_enc = pd.concat([df_num, df_cat], axis=1)
    #assert (X_enc.index.values == X_enc["absolute_idx"].values).all() # synthetic points does not have abs_idx
    
    # Drop the unnecessary index
    #X_enc.drop(columns = "absolute_idx",inplace = True)
    
    # ----------------------------
    # 7. Handle response variable
    # ----------------------------
    to_del = [col for col in X_enc.columns if col.startswith("op_intent")]
    #to_del = deepcopy(op_intent)
    #op_intent.remove("op_intent_Did not vote")
    #op_intent.remove("op_intent_Other")
    #y_var = X_enc[op_intent]
    X_enc.drop(columns = to_del,inplace = True)


    # ----------------------------
    # 8. Final safety checks
    # ----------------------------
    # Add missing columns expected by the model
    #for col in train_features:
    #    if col not in X_enc.columns:
    #       X_enc[col] = 0
            
    missing_cols = list(set(train_features) - set(X_enc.columns))

    if missing_cols:
        zeros = pd.DataFrame(
            0,
            index=X_enc.index,
            columns=missing_cols
        )
        X_enc = pd.concat([X_enc, zeros], axis=1)


    X_enc = X_enc[train_features]
    
    assert X_enc.columns.tolist() == list(train_features)
    assert X_enc.shape[1] == len(train_features), "Feature count mismatch"
    assert not X_enc.isna().any().any(), "NaNs introduced during encoding"

    return X_enc

#%% Wrapper class for dice

class DiceNNWrapper:
    def __init__(self, keras_model, train_features):
        self.model = keras_model
        self.train_features = train_features

    def predict_proba(self, raw_df):
        X = encode_for_nn(raw_df, self.train_features)
        return self.model.predict(X.to_numpy())

#%% Generate dataset for dice dice_ds

def generate_dice_ds(raw_df):
    dice_ds = raw_df.copy()
    
    #--- Drop useless columns
    to_drop = ["lat_scaled","long_scaled","absolute_idx"]
    
    dice_idx = {val : i for i,val in enumerate( dice_ds["absolute_idx"].copy().tolist() )}
    
    dice_ds.drop(columns = to_drop,inplace = True)

    #--- Reset index to fix internal bugs
    dice_ds.reset_index(drop = True,inplace = True)

    del(to_drop)

    #---- Put the activity into ordinals
    act_col = [col for col in dice_ds.columns if col.startswith("act_")]
    act_col.remove("act_transport")

    dice_ds[act_col] = dice_ds[act_col]*4

    # Change Income to ordinals variables 

    map_inc = {"None":0,
               "i1to30": 1,
               "i31to60": 2,
               'i61to90': 3,
               "i91to110": 4,
              "i111to150" : 5,
               "i151to200": 6,
               "i201toInf" : 7}
    dice_ds["ses_income"] = dice_ds["ses_income"].map(map_inc)

    # change Educ to ordinal variable
    map_educ = {
        "None":   0,  # no formal education completed
        "Prim":   1,  # primary school
        "Sec":    2,  # secondary school (high school)
        "Coll":   3,  # college / CEGEP / technical diploma
        "Bacc":   4,  # bachelor's degree
        "Master": 5,  # master's degree
        "PhD":    6   # doctoral degree
    }
    dice_ds["ses_educ"] = dice_ds["ses_educ"].map(map_educ)

    # change Age to ordinal form
    map_a = {"34m":0,
             "3554":1,
             "55p":2}
    dice_ds["age"] = dice_ds["age"].map(map_a)

    #--- Select the continuous feature
    continuous_col = ["lat","long","ses_income","ses_educ","age","voting_probability"]
    continuous_col.extend(act_col)

    #--- Select the categorical features
    categories_col = [col for col in list(dice_ds.columns) if col not in continuous_col]

    return dice_ds,dice_idx, continuous_col,categories_col

#%% Generate data instance of interest

def get_well_class_points(YCOMP):
    '''
    Parameters
    ----------
    YCOMP : A pd.DataFrame object 
        It must contain a columns representing the true class label, named "true class" and a column stating whether 
        or not the point was correctly classify by the model, named "correct"

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    out : dictionay
        It returns a dictionnary with the number of the class as key and the  absolute_idx (unique identifier across datasets)
        of the CORRECTLY classify points as values

    '''
    # Seeding here so each time same "randomness" activated
    seed(4243)
    classes = [0,1,2,3,4]
    out = {}
    for c in classes:
        class_name = "proba C" + str(c)
        mask = (YCOMP["true class"] == c) & (YCOMP["correct"] == 1) & (YCOMP[class_name] > 0.3)
        abs_idx = YCOMP.index[mask]
        try:
            out[c] = sample(list(abs_idx), 2) # no replacements
        except:
            raise Exception("Something is wrong, maybe not enough points")
    
    return out


def get_missclassified_points(YCOMP): 
    '''
    Parameters
    ----------
    YCOMP : A pd.DataFrame object 
        It must contain a columns representing the true class label, named "true class" and a column stating whether 
        or not the point was correctly classify by the model, named "correct"

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    out : dictionay
        It returns a dictionnary with the number of the class as key and the absolute_idx (unique identifier across datasets)
        of the MISSCLASSIFIEDpoints as values

    '''
    # Seeding here so each time same "randomness" activated
    seed(4243)
    classes = [0,1,2,3,4]
    out = {}
    for c in classes:
        mask = (YCOMP["true class"] == c) & (YCOMP["correct"] == 0)
        abs_idx = YCOMP.index[mask]
        try:
            out[c] = sample(list(abs_idx), 2) # no replacements
        except:
            raise Exception("Something is wrong, maybe not enough points")
    
    return out

#%% Compare the indexes of two individuals


def same_individual(ind_raw, ind_enco, tol=1e-6):
    '''

    Parameters
    ----------
    ind_raw : pd.Series
        a row representing an individual X with raw categorical value
    ind_enco : pd.Series
        a row representing individual X with one hot encoded categories
    tol : int or float
        DESCRIPTION. Tolereance threshold. The default is 1e-6.

    Returns
    -------
    bool
        Specifying whether or not the two individuals are identical.
    str
        Where the mismatch occured.

    '''
    # Check shape 
    #if (ind_raw.shape !=  (37,)):  # I added voting_proba that I had forgot.
    #    raise Exception(f"Not the good shapes. \nExpected : (37,).\tReceived {ind_raw.shape}")
        
    #if (ind_enco.shape !=  (157,)):
    #    raise Exception(f"Not the good shapes. \nExpected : (157,).\tReceived {ind_enco.shape}")

    
    
    #0. absolute_idx match
    
    if ("absolute_idx" in ind_enco.columns.values) & (("absolute_idx" in ind_raw.columns.values)):
        if (ind_enco["absolute_idx"] !=ind_raw.name):
            return False, "absolute_idx mismatch"
        
    # 1. Strong numeric invariants
    if ("lat_scaled" in ind_enco.columns.values) & ("long_scaled" in ind_raw.columns.values):
        for col in ["lat_scaled", "long_scaled"]:
            if abs(float(ind_raw[col]) - float(ind_enco[col])) > tol:
                return False, f"Mismatch in {col}"

    # 2. Ordinal variables
    if ORDINAL_MAPS["educ"][ind_raw["ses_educ"]] != int(ind_enco["educ"]):
        return False, "Education mismatch"

    if ORDINAL_MAPS["income"][ind_raw["ses_income"]] != int(ind_enco["income"]):
        return False, "Income mismatch"
    
    if ORDINAL_MAPS["age"][ind_raw["age"]] != int(ind_enco["age"]):
        return False,"Age mismatch"

    # 3. Activity consistency (relative)
    if not np.isclose(
        ind_raw["act_Volunteering"] * 4,
        ind_enco["act_Volunteering"],
        atol=tol
    ):
        return False, "Volunteering mismatch"
    
    # Galerie
    if not np.isclose(
        ind_raw["act_VisitsMuseumsGaleries"] * 4,
        ind_enco["act_VisitsMuseumsGaleries"],
        atol=tol
    ):
        return False, "Galere mismatch"
    
    # Fishing
    if not np.isclose(
        ind_raw["act_Fishing"] * 4,
        ind_enco["act_Fishing"],
        atol=tol
    ):
        return False, "Fishing mismatch"

    # Hunting
    if not np.isclose(
        ind_raw["act_Hunting"] * 4,
        ind_enco["act_Hunting"],
        atol=tol
    ):
        return False, "Fishing mismatch"
    
    # Motorized
    if not np.isclose(
        ind_raw["act_MotorizedOutdoorActivities"] * 4,
        ind_enco["act_MotorizedOutdoorActivities"],
        atol=tol
    ):
        return False, "Fishing mismatch"
             

    return True, "Same individual"

#%% functions to get comparisons
from copy import deepcopy

#modele = mod
#Xtest = X_test
#ytest = y_test
def get_YCOMP(modele,Xtest,ytest):
    
    to_be_aligned = False
    # Not aligned.
    if not (Xtest.index.values == Xtest["absolute_idx"].values).all():
        print("Xtest Not aligned or absolute_idx not the index")
    
    if not(ytest.index.values == ytest["absolute_idx2"].values).all():
        print("y_test not aligned or absolute_idx not the index")
        
    if not(ytest.index.values == Xtest.index.values).all():
        to_be_aligned = True;
        print("REAL PROBLEM : ytest and Xtest are NOT aligned")
        
    #-- Choose to align it 
    if to_be_aligned:
        Xtest.set_index("absolute_idx",inplace= True,drop = False)
        ytest.set_index("absolute_idx2",inplace= True,drop = False)
    
    # Get the probability of the predictions
    y_pred_prob = deepcopy(pd.DataFrame(modele.predict(Xtest.drop(columns = "absolute_idx")), index = Xtest.index))
    
    # Get the predicted class
    y_pred_class = y_pred_prob.idxmax(axis = 1)
    
    #---- Get an indicator of correctly classified points
    
    # 1. get a dictionary to map class name to number
    map_class = { name:idx for idx,name in enumerate(ytest.iloc[:,2:].columns)}
    
    #2.  Rename the column so that the labels matchs
    ytest.rename(columns = map_class,inplace = True)
    
    # 3. create the variable to compare to
    y_true = pd.Series(ytest.iloc[:,2:].idxmax(axis=1))
    
    # 5. Correct classification indicator
    correct = (y_pred_class.astype(int) == y_true.astype(int)).astype(int)
    
    #6. Create a column to see the misclassified row (binary index 0 misclassfied, 1 correctly classified)
    y_pred_prob["correct"] = correct
    y_pred_prob["pred"] = y_pred_class
    y_pred_prob["true class"] = y_true.astype(int)
    
    # add the index
    y_pred_prob["absolute_idx"] = Xtest["absolute_idx"]
    # CHeck if there is some NA value => 0 meaning no NA
    
    if (y_pred_prob.isna().any(axis = 0).sum() != 0):
        raise Exception("There are some NA values")
    
    print("No NaN values and absolute_idx is present")
    
    # Rename the first 5 columns 
    y_pred_prob.rename(columns = {0:"proba C0",1:"proba C1",2:"proba C2",3:"proba C3",4:"proba C4"}, inplace=True)
    
    # Alignement check
    assert (y_pred_prob.index.values == Xtest.index.values).all()
    
    # Na check
    assert y_pred_prob.isna().any(axis = 1).sum() == 0,"NaN introduced somehow"
    print("\nEverything's okay. Let's continue !")

    return y_pred_prob    
    
#%% SHAP

# This function exist because shap will only give one argument, data
# and I want to be able to give it model and encoded features too.
def make_predict_proba_from_raw(model, encoded_features,RAW_FEATURES_NAMES):
    def predict_proba_from_raw(x_raw):
        x_enc = encode_for_nn(x_raw, encoded_features)
        return model.predict(x_enc.to_numpy())
    return predict_proba_from_raw

#%% Gower distance inspired from Michael Yan "Gower" package.
# Build with the help of chatGPT


def gower_distance_matrix_fast(df, categorical_cols, ordinal_cols, continuous_cols,
                               ordinal_ranges, continuous_ranges):
    n = df.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    feature_count = 0

    X = df.reset_index(drop=True)

    # --- CATEGORICAL
    for col in categorical_cols:
        x = X[col].to_numpy()
        D += (x[:, None] != x[None, :]).astype(np.float32)
        feature_count += 1

    # --- ORDINAL
    for col in ordinal_cols:
        x = X[col].to_numpy(dtype=np.float32)
        rng = ordinal_ranges[col]
        D += np.abs(x[:, None] - x[None, :]) / rng
        feature_count += 1

    # --- CONTINUOUS
    for col in continuous_cols:
        x = X[col].to_numpy(dtype=np.float32)
        rng = continuous_ranges[col]
        D += np.abs(x[:, None] - x[None, :]) / rng
        feature_count += 1

    return D / feature_count

#%% F9

#neighborhoods[neighbors] = fc.get_close_prediction(k, pred_prob[proper_neighbors], chosen_class, proper_neighbors_idx)
#predicted_probabilities = pred_prob[proper_neighbors]
#classes = chosen_class
#vrai_idx = proper_neighbors_idx
#k = len()
def get_close_prediction(k,predicted_probabilities,classes,vrai_idx):
    
    close_predictions = []
    for i in range(0,k):
        appended = False
        for j in range(i+1,k-1):
            
            
            if (np.abs(predicted_probabilities[i][classes] - predicted_probabilities[j][classes] ) < 0.1):
                appended = True;
                
                if vrai_idx[j] not in close_predictions:
                    close_predictions.append(vrai_idx[j])
        
        # After looping on j, we add i if it some close relatives
        if (appended & (vrai_idx[i] not in close_predictions)):
                close_predictions.append(vrai_idx[i])
                
                
    return close_predictions


#%% F3 selectivity get the s of dice


#dice_explainer = dice
def get_dice_s(correct_points : dict ,X_dice_ds,dice_explainer):
    total_changed1 = []
    total_changed2 = []
    
    
    for classe in correct_points.keys():
        
        idx1 = correct_points[classe][0]
        idx2 = correct_points[classe][1]
        
        # get the point
        point1 = X_dice_ds.loc[[idx1]] 
        point2 = X_dice_ds.loc[[idx2]]
        
        #randomly generate another class
        classes = [0,1,2,3,4]
        classes.remove(classe)
        seed(4243 + classe)
        selected_classes = sample(classes,1)[0]
        
        # generate counterfactuals
        cf1 = dice_explainer.generate_counterfactuals(
            query_instances=point1,
            total_CFs=3,
            desired_class=selected_classes,
            random_seed = 4243 + classe)
        
        cf2 = dice_explainer.generate_counterfactuals(
            query_instances=point2,
            total_CFs=3,
            desired_class=selected_classes,
            random_seed = 4243 + + classe)
        
        # Select cf as dataframe
        cf1_df = cf1.cf_examples_list[0].final_cfs_df.drop(columns = ["op_intent"])
        cf2_df = cf2.cf_examples_list[0].final_cfs_df.drop(columns = ["op_intent"])
        assert (list(cf1_df.columns.values) == list(point1.columns.values))
        assert (list(cf2_df.columns.values) == list(point2.columns.values))
        
        
        instance_nchanged1 = []
        instance_nchanged2 = []
        for i in range(len(cf1_df)):
            
            # compute the number of changement
            n_changed1 = sum(
                1 for feature in  list(point1.columns.values) 
                if point1[feature].values[0] != cf1_df.loc[i][feature])
            
            n_changed2 = sum(1 for feature in list(point2.columns.values) 
                             if point2[feature].values[0] != cf2_df.loc[i][feature])
            
    
            instance_nchanged1.append(n_changed1)
            instance_nchanged2.append(n_changed2)
        
        #Append to the summary results
        total_changed1.extend(instance_nchanged1)
        total_changed2.extend(instance_nchanged2)
    
    get_avg = total_changed1 + total_changed2
    
    # bootstrap the results 
    
    print(get_avg)
    np.mean(get_avg)
    np.std(get_avg)
    
    s_dice = round( np.mean(get_avg)) # 8
    return s_dice


'''
import random


#### mini test to bootstrap cuz few points
get_avg = [2, 1, 2, 16, 17, 15, 1, 2, 1, 17, 16, 18, 21, 19, 20, 2, 2, 2, 6, 7, 8, 1, 2, 2, 12, 7, 8, 8, 9, 10]
np.mean(get_avg) # 8.4...
np.std(get_avg) # 6.77

boot_avg_list = []
boot_std_list = []
n = len(get_avg)
n_boot = 1000

for i in range(n_boot):
    
    random.seed(4243 + i)
    sample = [get_avg[random.randrange(n)] for _ in range(n)]
    boot_mean = np.mean(sample)
    boot_std = np.std(sample)
    
    boot_avg_list.append(boot_mean)
    boot_std_list.append(boot_std)
    
# then compute the mean of boot_avg and boot_std
np.mean(boot_avg_list) # 8.4944
np.mean(boot_std_list) #6.6439
np.percentile(boot_avg_list, [2.5,97.5]) # array([ 6.19916667, 10.86666667])
np.percentile(np.array(boot_std_list),[2.5, 97.5]) # array([5.50837742, 7.59972404])
np.median(boot_std_list) # 6.6779
np.median(boot_avg_list)

        
'''

#%% OSDT F3 White box 


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


def true_valid_feature_sets(x):
    if x["x2"] > 0.3:
        return [ {"x2"} ]
    else:
        if x["x1"] <= 0.5:
            return [ {"x1"} ]
        else:
            return [ {"x1","x2"} ]



#true_valid_feature_sets(X_test_simple.iloc[0])

other_map = {0:"x2",1:"x1"}
def predicates_to_raw_features(path_feature_indices, map_vars = other_map):
    """
    Maps predicate indices → raw feature names.
    predicate space => input feature
    """
    raw_feats = set()
    for idx in path_feature_indices:
        predicate = map_vars[idx]
        raw_feat = predicate.split("_")[0]  # or custom parser
        raw_feats.add(raw_feat)
    return raw_feats



def jaccard(a, b):
    return len(a & b) / len(a | b)

tree_map_vars_simple=  {0: 'x2 <= 0.2992497831583023', 1: 'x1 <= 0.4993823915719986'}
map_classes_simple  = {0: 0, 1: 1}

def osdt_f7_3_instance(x_raw, x_predicate, root):
    _, path_idx = get_prediction_and_path_indices(x_predicate, root,
                                                  tree_map_vars_simple,map_classes_simple)

    expl_feats = predicates_to_raw_features(path_idx, other_map)
    valid_sets = true_valid_feature_sets(x_raw)

    return max(jaccard(expl_feats, s) for s in valid_sets)

#osdt_f7_3_instance(X_test_simple.iloc[0]  ,X_test_guessed_simple.iloc[[0]], root_simple)

#%% OSDT F9.1 SIMILARITY
def compute_sign_dist(s1,s2):
    '''
    Parameters
    ----------
    s1 : TYPE
        DESCRIPTION.
    s2 : TYPE
        DESCRIPTION.

    Raises
    ------
    

    Returns
    -------
    TYPE : INT 
    DESCRIPTION : The function returns the distance between two signatures.
    Identical paths yields a distance of  0.
    Diverge at the root yields a distance (close) to 1

    '''
    
    #---- compute the Longuest Common Prefix (LCP) length
    L = 0
    for a,b in zip(s1,s2):
        if a == b:
            L +=1
        else:
            break
    #--- Here we have the LCP
    
    #--- Normalize by the max path length between the two
    m = np.maximum(len(s1),len(s2))
    
    #---- Sanity check
    if m == 0:
        return 0
    
    dist = 1 - (L/m)
    
    return dist


from typing import Tuple, List

tree_map_vars = {0: 'cons_coffee_place_ind <= 0.5',
 1: 'ses_income_i201toInf <= 0.5',
 2: 'ses_dwelling_App <= 0.5',
 3: 'act_transport_Car <= 0.5',
 4: 'age_34m <= 0.5',
 5: 'age_55p <= 0.5',
 6: 'people_predict_CAQ <= 0.5',
 7: 'act_VisitsMuseumsGaleries <= 0.5',
 8: 'act_VisitsMuseumsGaleries <= 1.5',
 9: 'act_MotorizedOutdoorActivities <= 0.5',
 10: 'vehicule_VUS <= 0.5'}

tree_map_classes = {0: 'CAQ', 1: 'PCQ', 2: 'PLQ', 3: 'PQ', 4: 'QS'}

# =============================================================================
# # function to predict and get unique, compact, easily readable signature
# =============================================================================

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
#%% Presentation Latex
def get_latex_format(metric_table : pd.DataFrame,caption: str):
    
    label_cap = [caption.split(" ")[word -1].lower()+"_" + caption.split(" ")[word].lower() for word in range(len(caption.split(" "))) if word == 1]

    label = "tab:" + label_cap[0]

    latex = metric_table.to_latex(
        index=True,
        caption= caption,
        label=label,
        bold_rows=True,
        column_format="|l" +"c"*len(metric_table.columns.tolist())+  "|",
        escape=False
    )

    # Insert placement + centering
    latex = latex.replace(
        "\\begin{table}\n",
        "\\begin{table}[htbp]\n\\centering\n"
    )

    # Add horizontal lines after each row
    latex = latex.replace("\\\\\n", "\\\\ \\hline\n")

    # Remove booktabs rules if present
    latex = latex.replace("\\toprule\n", "\\hline\n")
    latex = latex.replace("\\midrule\n", "")
    latex = latex.replace("\\bottomrule\n", "")

    print(latex)