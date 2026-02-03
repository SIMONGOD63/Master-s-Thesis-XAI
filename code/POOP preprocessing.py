#%% Imports

#--- Packages
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer



#--- load the datas
#path = r"C:\Users\simeo\A. Data Science\Master thesis\01. Data" #
path = r"C:/Users/User/Documents/Data Science/Master thesis/01. Data/"
download_path = r"C:\Users\User\Documents\Data Science\Master thesis\07. Correction\Improve code"
# Spatial datasets.
data_spatial_zip = pd.read_csv(path + "/lat long/full_dataset_csv.csv")
data_spatial_zip = data_spatial_zip[data_spatial_zip["country"] == "Canada"]
data_spatial_city = pd.read_csv(path + "/lat long/canadacities.csv")
data_spatial_zip["zipCode"]

donnee = pd.read_csv(path + "/quebec_prov_2022_Simeon/hub/data_clean.csv")

# remove the columns. We start with a fresh index from 0 to n
donnee.drop(columns = ['Unnamed: 0',"id"], inplace = True) # old indexes from 1 to...
# Keep an original index marker
donnee["absolute_idx"] = donnee.index.astype(int)
#%% POOP preprocessing

class Preprocessing:
    
    
    def __init__(self,initial_data,data_path):
        self.data = deepcopy(initial_data)
        self.path = data_path
        self.y = None
        
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    def concentrate_geographical(self,DS_zip, DS_city):
        
        #Zip code based
        zip_to_coords = {zip_code.lower(): [lat, lon] for zip_code, lat, lon in zip(DS_zip["zipCode"].str[:3],DS_zip['latitude'],
                                                                                    DS_zip['longitude'])}
        
        # City based
        city_to_coords = {city : [lat,lon] for city,lat,lon in zip(DS_city['city'],DS_city['lat'],DS_city['lng'])}
        
        # Transform the data
        self.data['geo_posPC'] = self.data['postal_code'].map(zip_to_coords)
        self.data['geo_posCN'] = self.data['city_name'].map(city_to_coords)
        
        # fill in nan value from PC to CN
        self.data['geo_posCN'] = self.data['geo_posCN'].combine_first(self.data['geo_posPC'])
        self.data.drop(columns = ['geo_posPC'], inplace = True)
        
        # Suppress useless columns (before it was done in the Prepare Y section)
        to_del = ['postal_code',"riding","city_name","mrc_name","region_name"]
        self.data.drop(columns = to_del,inplace = True)
        
        # put it into simpler form 
        self.data['lat'] = self.data['geo_posCN'].apply(lambda x : x[0] if isinstance(x, list) else np.nan)
        self.data['long'] = self.data['geo_posCN'].apply(lambda x : x[1] if isinstance(x, list) else np.nan)
        self.data.drop(columns = ['geo_posCN'], inplace = True)
        
        # Have a standardized version
        self.data['lat_scaled'] = (self.data['lat'] - np.mean(self.data['lat']))/ np.sqrt(np.var(self.data['lat']))
        self.data['long_scaled'] = (self.data['long'] - np.mean(self.data['long']))/ np.sqrt(np.var(self.data['long']))
        
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #---- Handle NaN values
    def handle_NaN(self):
        
        ##########################################################################
        #--------- 1. High prop variables
        ##########################################################################
        
        high_p = ['health_relation_entourage',"health_stay_home","health_physical_health","health_time_nature","health_mental_health"]
        self.data.drop(columns = high_p,inplace = True)
        
        # Further delete nan values
        self.data = self.data[self.data['op_intent'].notna()]
        
        # print the proportion of NaN values
        print("4. The new proportion of rows not having any nan value : ", round(1 - self.data.isna().any(axis = 1).mean(),4)*100,"%")
        
        ##########################################################################
        #--------- 2. Film columns
        ##########################################################################
        film_col = [col for col in self.data.columns if col.startswith("film_")]

        # Initialize the col
        self.data["film_no"] = 0

        # mask to map the condition 
        mask = self.data['film_title'].isna() & self.data['film_unknown'].isna()

        # Change value of film col depending on the condition (not interested in movie)
        self.data.loc[mask,"film_no"] = 1 # translate a non interest/not willing to answer to

        #test_self.data = self.data[film_col] self.dataset to inspect 

        # remove from the list otherwise would set it to 0
        film_col.remove("film_title")

        self.data.loc[mask,film_col] = 0 # change NaN to 0 
        
        del(mask) #??? Idk if necessary
        
        ##########################################################################
        #--------- 3. Music columns
        ##########################################################################
        #initialize the column
        self.data['music_no'] = 0
        music_col = [col for col in self.data.columns if col.startswith("music_")]


        ## Create the condition
        mask2 = pd.isna(self.data['artist_name'])

        # change the value
        self.data.loc[mask2,music_col] = 0
        self.data.loc[mask2,"music_no"] = 1

        self.data.drop(columns = ['artist_name',"film_title"], inplace = True)

        print("3. The new proportion of rows NOT HAVING any nan value : ", round(1 - self.data.isna().any(axis = 1).mean(),4)*100,"%")

        del(music_col, film_col,mask2)
        
        ##########################################################################
        #--------- 4. Delete NaN Except those in voting proba
        ##########################################################################
        
        # build a correct mask
        mask_no_nan = self.data.notna().all(axis =1)

        mask_nan_only_vp = (self.data["voting_probability"].isna() & self.data.drop(columns=["voting_probability"]).notna().all(axis =1) )

        self.data = self.data[mask_no_nan | mask_nan_only_vp].copy()

        print(f"Proportion of initial data is {round( (len(self.data)/len(self.data))*100,4) }") # 71.1731 => It changes a lot of things.


        # Check if it did the job 
        assert self.data.isna().any(axis = 0).sum() == 1
        assert self.data["voting_probability"].isna().any(axis = 0) # true

        del(mask_nan_only_vp, mask_no_nan)
        
        ##########################################################################
        #--------- 5. Split the time variable
        ##########################################################################
        self.data['year'] = self.data['time'].apply(lambda x : x[:4])
        self.data['month'] = self.data['time'].apply(lambda x: x[5:7])
        self.data['day'] = self.data['time'].apply(lambda x: x[8:10])
        self.data.drop(columns = ['year'],inplace = True)
        self.data.drop(columns = ['time'], inplace = True)
        

        

    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------      
    
    def initialize_y(self):
        y_variables = ['op_intent',"vote_pred","people_predict"]
        self.data[y_variables] = self.data[y_variables].astype("category")
        
        # Add original index to self.data_y and self.data for alignment
        self.y = pd.DataFrame(self.data[y_variables + ["absolute_idx"]].copy()) # ???
        
        # drop what's unecessary
        self.data.drop(columns = y_variables, inplace = True)
    
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    # Imputation
    def impute(self):
        
        # Create imputed indicator
        self.data["imp_ind"] = self.data["voting_probability"].isna().astype(int)
        
        # Keep the idx
        abs_idx = self.data["absolute_idx"].copy()
        X = self.data.drop(columns=["absolute_idx"]).copy()
        

        
        # Initialize the imputator object
        imp = IterativeImputer(max_iter = 10, random_state = 4243,
                               n_nearest_features= None, # all feature used
                               initial_strategy='median', # less affected by outliers
                               skip_complete = True,
                               verbose = 2,
                               min_value= 0,
                               max_value= 1,
                               #imputation_order=
                               #add_indicator= True
                               )
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=abs_idx)
        self.data = X_imp
        # Keep it as a columns too
        self.data["absolute_idx"] = abs_idx.values  
        
        # Align y and X
        self.y = self.y.set_index("absolute_idx", drop=False).loc[abs_idx]

        # Check if no more NaN
        assert sum(self.data.isna().any(axis = 0)) == 0,"In impute : Still some NaN values"
        
        ## Put the predicted voting proba value into the right format
        self.data["voting_probability"] = self.data["voting_probability"].apply( lambda x : max(0,round(x,1)))
        self.data["voting_probability"] = self.data["voting_probability"].apply( lambda x : 1 if x > 1 else x )
        
        # Shuffle cuz all imputed value were concentrated

        from sklearn.utils import shuffle
        self.data,self.y = shuffle(self.data,self.y,random_state = 4243)
        
        #-- Discretize day variable 
        self.data["day"] = self.data["day"].astype("int")
        day_dis = pd.qcut(self.data["day"], 3, labels = ["<11","11-24",">24"])
        self.data["day"] = day_dis
        self.data["day"] = self.data["day"].astype("category")
    
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    # Download intermediate results
    def dl_inter_res(self):
        self.data.to_csv(self.path + "/00. FINALS_POOP/X_intermediate.csv")
        self.y.to_csv(self.path + "/00. FINALS_POOP/y_intermediate.csv")
        
    
    
    def ordinal_encoding(self):
        #--- Load what's necessary
        from sklearn.preprocessing import OrdinalEncoder
        ordenco = OrdinalEncoder()
        raw_cat = pd.read_parquet(self.path + "/00. FINALS_POOP/raw_aligned.parquet")
        
        # To ease the memory problem
        self.data = self.data.copy()
        
        # Check if self.data and raw_cat can be alinged 
        assert set(raw_cat.index) == set(self.data["absolute_idx"]),"ordinal_encoding : raw_cat and data cannot be aligned."
        
        # Set absolute idx as the index
        self.data.set_index("absolute_idx",inplace = True, drop = False)
        
        # check if they are aligned
        assert (raw_cat.index.values == self.data.index.values).all(),"ordinal_encoding : raw_cat and data are not aligned."

        
        
        #-- Get the ordinal categorial features
        to_ord = ["ses_income","ses_educ","day", "act_VisitsMuseumsGaleries","act_Fishing","act_Hunting","act_MotorizedOutdoorActivities","act_Volunteering"]
        ord_data = raw_cat[to_ord].copy()
        
        # Create a map for income
        map_inc = {"None":0,
                   "i1to30": 1,
                   "i31to60": 2,
                   "i61to90": 3,
                   "i91to110": 4,
                  "i111to150" : 5,
                   "i151to200": 6,
                   "i201toInf" : 7}
        
        # Create a map for Education
        map_educ = {
            "None":   0,  # no formal education completed
            "Prim":   1,  # primary school
            "Sec":    2,  # secondary school (high school)
            "Coll":   3,  # college / CEGEP / technical diploma
            "Bacc":   4,  # bachelor's degree
            "Master": 5,  # master's degree
            "PhD":    6   # doctoral degree
        }
        
        # Create a map for day
        map_d ={'<11' :0,
               '11-24':1,
               '>24':2}
        
        # Create a map for age
        map_a = {"34m":0,
                 "3554":1,
                 "55p":2}
        
        # Encode the variables
        ord_data["ses_income"] = ord_data["ses_income"].map(map_inc)
        ord_data["ses_educ"] = ord_data["ses_educ"].map(map_educ)
        ord_data["day"] = ord_data["day"].map(map_d)
        
        #-- Create ordinal val
        ordinals = pd.DataFrame(ordenco.fit_transform(ord_data),columns = ordenco.get_feature_names_out(), index = ord_data.index  )
        
        # Chech alignment
        assert (ordinals.index.values == self.data.index.values).all(),"ordinal encoding : 1. alignment check"
        
        # Assign it to my data
        self.data[to_ord] = ordinals[to_ord]
        self.data["day"] = ordinals["day"]
        
        #--- Check for silent NaN
        assert self.data.isna().any(axis = 1).sum() == 0,"ordinal encoding : 2. silent NaN check"
        
        #---- Drop the encoded version of this variable
        # ses_educ
        educ_cols = [col for col in self.data.columns if col.startswith("ses_educ")]
        self.data["educ"] = ordinals["ses_educ"]
        self.data.drop(columns = educ_cols, inplace = True)
        
        # ALways check silent NaN
        assert self.data.isna().any(axis = 1).sum() == 0,"ordinal encoding : 3. silent NaN check"
        
        # income
        inc_cols = [col for col in self.data.columns if col.startswith("ses_income")]
        self.data["income"] = ordinals["ses_income"] 
        self.data.drop(columns = inc_cols, inplace = True)

        assert self.data.isna().any(axis = 1).sum() == 0,"ordinal encoding : 3. silent NaN check"
        
        # Age (different method cuz I forgot about it)
        self.data["age"] = raw_cat["age"].apply(lambda x: map_a[x])
        age_col = [col for col in self.data.columns if col.startswith("age")]
        age_col.remove("age")
        self.data.drop(columns = age_col, inplace= True)
        
        #---- Final check for NaN
        assert self.data.isna().any(axis = 0).sum(),"ordinal encoding : 4. Final NaN check"
    
    
    #--- Get all of the categories into k-1 one hot encoded format
    def all_cat_encoding(self):
        L = ["cons_coffee","ses_dwelling","app_swag_","music_","film_","ses_ethn","act_transport","vehicule_","cons_Smoke","cons_meat","cons_brand","animal_"]
        cat_to_remove =[]
        col_names = self.data.columns
        for cat in L:
            cat_col = [col for col in self.data.columns if col.startswith(cat)]
            self.data_cat = self.data[cat_col]
            test_k = sum(self.data_cat.sum(axis = 1) != 1)
            
            if test_k != 0:
                print(f'There is a problem with the variable categories {cat} \n')
            else :
                print(f"with the category {cat}, we have k categories. So we remove one \n")
                cat_to_remove.append(cat_col[-1])
            
        self.data.drop(columns = cat_to_remove, inplace = True)
        
        #--------------------------------------------------------------------------------------
        #---- Removed by hand 
        # educ related 
        remove_by_hand = ["educCollege","educUniv","educBHS"] 
        
        #----  Sport
        sport = [col for col in self.data.columns if col.startswith("act_")]
        sport = [col for col in sport if not col.startswith("act_transport")]
        rem = ['act_VisitsMuseumsGaleries','act_Fishing','act_Hunting','act_MotorizedOutdoorActivities','act_Volunteering']
        sport = [col for col in sport if col not in rem]
        assert sum(self.data[sport].sum(axis=1) !=1) == 0,"all cat encoding, sport"
        remove_by_hand.append(sport[-1])
        
        
        #---- alcohol
        alc = [col for col in self.data.columns if col.startswith("cons_")]
        alc = [col for col in alc if not col.startswith("cons_Smoke")]
        alc = [col for col in alc if not col.startswith("cons_meat")]
        alc = [col for col in alc if not col.startswith("cons_brand")]
        alc = [col for col in alc if not col.startswith("cons_coffee")]
        
        assert sum(self.data[alc].sum(axis = 1) !=1) == 0,"all cat encoding, alcohol" # 0 => mean k categories
        remove_by_hand.append(alc[-1]) # cons_noDrink
        
        #---- Gender
        col_g2 = ["male","female","genderOther"]
        gender_del = [col for col in self.data.columns if col.startswith("gender")]
        gender_del.remove("genderOther")
        remove_by_hand.extend(gender_del)
        remove_by_hand.append("genderOther")
        
        #---- Orientation sexuelles
        col_sex = ['ses_Asexual','ses_Pansexual',"ses_Queer","ses_bisex","ses_gai","ses_hetero","ses_sexOri_other","ses_Questionning"] # start with ses_ foesnt work
        col_sex2 = [ col for col in self.data.columns.values if col.startswith("ses_")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_educ")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_dwelling")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_income")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_ethn")]
        col_sex2.remove("ses_languageOther")

        #--- Solve the issue of several categories per row
        self.data_sex = self.data[col_sex].copy()
        mask = (self.data_sex.sum(axis=1) !=1) 
        sum(self.data_sex.loc[mask,"ses_sexOri_other"])
        self.data_sex.loc[mask,"ses_sexOri_other"] = 0
        
        # Put into k-1 categories
        remove_by_hand.append("ses_sexOri_other")
        
        #-- Modify it in the OG ds
        sum(self.data.loc[mask,"ses_sexOri_other"])
        self.data.loc[mask,"ses_sexOri_other"] = 0
        
        ###----- langue
        col_lang = ["langFr","langEn","ses_languageOther"]
        self.data_lang = self.data[col_lang]
        sum(self.data[col_lang].sum(axis= 1) !=1) # 0 => k categories
        remove_by_hand.append(col_lang[-1])
        
        #---- Remove the columns to have k-1 type of one hot encoding
        self.data.drop(columns = remove_by_hand,inplace = True)
        
        #--- final NaN check 
        assert self.data.isna().any(axis = 1).sum() == 0
        
        #--- Check for same individuals
        print("idx data : ", list(set(self.data.index))[:5])
        print("idx y ",list(set(self.y["absolute_idx"]))[:5])
        assert set(self.data.index) == set(self.y["absolute_idx"]),"Check same individuals" # Failed. Not aligned anymore.
        print("Check same individuals : passed")
        
        #--- check for duplicates

        assert self.data.index.is_unique,"Check duplicates" # passed
        assert self.y["absolute_idx"].is_unique, "Check duplicates 2" # passed
        print("Check for duplicates : passed")
        
        #--- check alignment

        assert (self.data.index.to_numpy() == self.y["absolute_idx"].to_numpy()).all(),"Check alignement" # passed
        print("Check for alignement : passed")
        self.y.set_index("absolute_idx",inplace = True, drop = False)
        
        #--- Check for any NaN left

        assert self.data.isna().sum().sum() == 0
        print("Check for NaN : passed")
        
        print("Those test proves that the logic is sound, congratz !")

    def download(self):
        self.data.to_csv(self.path + "/00. FINALS_POOP/X_post_preprocess.csv")
        self.y.to_csv(self.path + "/00. FINALS_POOP/y_post_preprocess.csv")
        
    def do_preprocess(self, ini_data,download_path,DS_city_out, DS_zip_out):
        preprocessed_data = Preprocessing(ini_data, download_path)
        preprocessed_data.concentrate_geographical(DS_city = DS_city_out, DS_zip = DS_zip_out)
        preprocessed_data.handle_NaN()
        preprocessed_data.initialize_y()
        preprocessed_data.impute()
        preprocessed_data.all_cat_encoding()
        return preprocessed_data
        
#%% Test and pipeline

# =============================================================================
# preprocessed_data = Preprocessing(donnee,download_path)
# preprocessed_data.concentrate_geographical(DS_city = data_spatial_city, DS_zip = data_spatial_zip)
# preprocessed_data.handle_NaN()
# preprocessed_data.initialize_y()
# preprocessed_data.impute()
# #preprocessed_data.dl_inter_res()
# 
# preprocessed_data.all_cat_encoding()
# =============================================================================
