'''
The goal of this file is to create a raw datasets, i.e. not encoded.
'''

#%% Imports

import numpy as np
import pandas as pd
#--- The datasets
#path = r"C:\Users\simeo\A. Data Science\Master thesis\01. Data"
path = r"C:/Users/User/Documents/Data Science/Master thesis/01. Data/"
path2 = r"C:\Users\User\Documents\Data Science\Master thesis\07. Correction\Improve code\00. FINALS"
data_raw = pd.read_csv(path +"/quebec_prov_2022_Simeon/hub/_raw/data-hub-2022-10-27_raw.csv")
catds = pd.read_csv(path2 +"/X_intermediate.csv")
y_ds = pd.read_csv(path2 +"/y_intermediate.csv", index_col=0)
#wds = pd.read_csv(path + "/00. FINALS/X_post_preprocess.csv", index_col=0)

# Set the index absolute_idx to avoid silent issues
#if (catds["absolute_idx"].values != catds.index.values).all() is False:
catds["aboslute_idx"] = catds["absolute_idx"].astype(int)
catds.set_index("absolute_idx",inplace = True,drop = False)

#wds["absolute_idx"] = wds["absolute_idx"].astype(int)
#wds.set_index("absolute_idx",inplace = True)

#%%

class Raw_dataset:
    
    def __init__(self,X_inter,y_inter):
        self.X_intermediate = X_inter.copy()
        self.y_intermediate = y_inter.copy()
        self.raw_out = None
        
        
    def automated_reconstruction(self):
        L = ["day","cons_coffee","ses_income","ses_dwelling","ses_educ","app_swag","music","film","ses_ethn","act_transport","vehicule","cons_Smoke","cons_meat","cons_brand","animal"]
        self.raw_out = pd.DataFrame(index=self.X_intermediate.copy().index) # absolute_idx
        
        # Add forgotten variables
        self.raw_out["absolute_idx"] = self.X_intermediate["absolute_idx"].copy()
        self.raw_out["month"] = self.X_intermediate["month"].copy()
        self.raw_out["imp_ind"] = self.X_intermediate["imp_ind"].copy()
        self.raw_out["app_noTattoo"] = self.X_intermediate["app_noTattoo"].copy()
        
        
        for cat in L:
            # Get all the columns from a category
            cat_col = [col for col in self.X_intermediate.columns if col.startswith(cat)]
            sub_cat = self.X_intermediate[cat_col].copy()
            test_k = sum(sub_cat.sum(axis = 1) != 1)
            
            if test_k != 0:
                print(f'There is a problem with the variable categories {cat} \n')
            else :
                print(f"with the category {cat}, we have k categories. So it is a full category \n")
                col_idx = np.argmax(sub_cat, axis = 1)
                dic_map = {col:name.split(cat+"_")[1] for col, name in enumerate(cat_col) }
                self.raw_out[cat] = list(map( lambda x: dic_map[x],col_idx ))
                
        # Check index again
        assert (self.raw_out.index == self.X_intermediate.index).all(),"Automated R : aligned index"
        
        #-- Add day as it stands.
        self.raw_out["day"] = self.X_intermediate["day"]

        #--- check for any nan values

        assert self.raw_out.isna().any(axis = 1).sum() == 0
        print("everything works perfectly")
    
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------

    def non_automated_reconstruction(self):
        
        #---- Function required for non automated reconstruction 
        def raw_col(col_names, general_name, sep = "_"):
            # col_names is the name of all the columns in the one hot encoding format
            # dataFrame is the dataFrame to which we will add this columns
            # general_name is the general name of the category, i.e. sport 
            sub = self.X_intermediate[col_names]
            test_k = sum(sub.sum(axis = 1) != 1)
            
            if test_k != 0:
                print(f'There is a problem with the variable categories {general_name} \n')
            else :
                print(f"with the category {general_name}, we have k categories. So it is a full category \n")
                col_idx = np.argmax(sub, axis = 1)
                dic_map = {col:name.split(general_name + sep)[1] for col, name in enumerate(col_names) }
                return list(map( lambda x: dic_map[x],col_idx ))
                #dataFrame[general_name] = list(map( lambda x: dic_map[x],col_idx ))        
                
        #----------                
        #--- Sport
        # filter the columns
        sport = [col for col in self.X_intermediate.columns if col.startswith("act_")]
        sport = [col for col in sport if not col.startswith("act_transport")]
        rem = ['act_VisitsMuseumsGaleries','act_Fishing','act_Hunting','act_MotorizedOutdoorActivities','act_Volunteering']
        sport = [col for col in sport if col not in rem]

        self.raw_out["sport"] = raw_col(sport,"act")
        
        #--- Alcohol

        alc = [col for col in self.X_intermediate.columns if col.startswith("cons_")]
        alc = [col for col in alc if not col.startswith("cons_Smoke")]
        alc = [col for col in alc if not col.startswith("cons_meat")]
        alc = [col for col in alc if not col.startswith("cons_brand")]
        alc = [col for col in alc if not col.startswith("cons_coffee")]

        # Creating the columns
        self.raw_out["alcohol"] = raw_col(alc, "cons")
        
        #---- Age
        age = [col for col in self.X_intermediate.columns if col.startswith("age")]
        age.remove("age")
        self.raw_out["age"] = raw_col(age,"age",sep="")
        
        #---- Langue
        col_lang = ["langFr","langEn","ses_languageOther"]
        dic_m = {0: "Fr",1:"En",2:"Other"}
        col = np.argmax(self.X_intermediate[col_lang],axis =1)
        self.raw_out["lang"] = list(map(lambda x: dic_m[x],col))
        
        #---- people_predict 
        self.y_intermediate.set_index("absolute_idx", inplace = True, drop = False)
        
        # Check alignment first
        assert (self.y_intermediate.index.values == self.raw_out.index.values).all(),"N-A recons. : 1st check alignement y and X"
        
        self.raw_out["people_predict"] = self.y_intermediate["people_predict"]
        self.raw_out["op_intent"] = self.y_intermediate["op_intent"]
        
        #--- Checks

        assert self.raw_out.index.is_unique,"N-A recons. : non-unique idx"
        assert self.y_intermediate.index.is_unique,"N-A recons. : non unique y idx "
        assert self.raw_out.isna().any(axis=0).sum() == 0,"N-A recons. : NaN in raw_out"
        print("\nEverything is okay. Let's continue.")
        
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    
    def add_special_cases(self):
        #---- Gender 
        col_g2 = ["male","female","genderOther"]
        sub_g2 = self.X_intermediate[col_g2].copy()

        (sub_g2.sum(axis = 1) == 1).all()
        sum(sub_g2.sum(axis = 1) != 1)
        dic_m = { k:name  for k,name in enumerate(col_g2)}
        col = np.argmax(self.X_intermediate[col_g2],axis = 1)
        self.raw_out["gender"] = list(map(lambda x: dic_m[x],col))
        
        #---- Orientation sexuelles
        #Get the col name
        col_sex = ['ses_Asexual','ses_Pansexual',"ses_Queer","ses_bisex","ses_gai","ses_hetero","ses_sexOri_other","ses_Questionning"] # start with ses_ foesnt work
        col_sex2 = [ col for col in self.X_intermediate.columns.values if col.startswith("ses_")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_educ")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_dwelling")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_income")]
        col_sex2 = [ col for col in col_sex2 if not col.startswith("ses_ethn")]
        col_sex2.remove("ses_languageOther")
        len(col_sex2)== len(col_sex) == len(data_raw.iloc[:,38].value_counts().tolist())
        set(col_sex2) == set(col_sex)
        sub_sex = self.X_intermediate[col_sex].copy()

        sub_sex.sum(axis = 0)

        # Is there a problem ? Several categories per rows
        sum(sub_sex.sum(axis=1) !=1) # 2309

        #-- Identify and correct the problem =>
        mask = (sub_sex.sum(axis=1) !=1) 
        sum(sub_sex.loc[mask,"ses_sexOri_other"])
        sub_sex.loc[mask,"ses_sexOri_other"] = 0
        assert sum( sub_sex.sum(axis = 1) != 1) == 0

        #-- Add this category
        dic_m = {k:name for k,name in enumerate(col_sex)}
        col = np.argmax(sub_sex, axis =1)
        np.unique(col)
        self.raw_out["sex_ori"] = list(map(lambda x: dic_m[x],col))
        
        #--- Checks

        assert self.raw_out.index.is_unique
        assert y_ds.index.is_unique
        assert self.raw_out.isna().any(axis=0).sum() == 0
        print("\nEverything is okay. Let's continue.")

        #del(sub_g2,sub_sex, col_sex,col_sex2,col, dic_m,col_g2,data_raw,age, alc,cat,cat_col,col_idx, col_lang, col_names, dic_map, L, mask,rem, sport, sub_cat)
        
        #------- Remaining binary variables/ordinals
        col_catds = self.X_intermediate.columns.values
        self.raw_out[col_catds[:6]] = self.X_intermediate[col_catds[:6]]


        #--- pays qc, a binary variable
        self.X_intermediate["pays_qc"].value_counts() # binary
        self.raw_out["pays_qc"] = self.X_intermediate["pays_qc"]


        #--- immigrant 
        self.X_intermediate["immigrant"].value_counts() # binary
        self.raw_out["immigrant"] = self.X_intermediate["immigrant"]

        #--- lat and long 
        self.raw_out["lat"] = self.X_intermediate["lat"]
        self.raw_out["long"] = self.X_intermediate["long"]
        self.raw_out["lat_scaled"] = self.X_intermediate["lat_scaled"]
        self.raw_out["long_scaled"] = self.X_intermediate["long_scaled"]

        #----- voting_probability
        self.raw_out["voting_probability"] = self.X_intermediate["voting_probability"]

        # Remove this weird col
        self.raw_out.drop(columns = ["Unnamed: 0"],inplace = True)
        
        #---- Final check

        # No nan values
        assert self.raw_out.isna().any(axis = 1).sum() == 0
        print("Tout est en ordre !")
        
    def create_Raw_dataset(self):
        out = Raw_dataset(catds, y_ds)
        out.automated_reconstruction()
        out.non_automated_reconstruction()
        out.add_special_cases()
        return out
        
    def download_Raw_dataset(self, dload_path):
        self.raw_out.to_parquet(dload_path + "/raw_aligned.parquet")
        
#%% Test

#dl_path = r"C:\Users\User\Documents\Data Science\Master thesis\07. Correction\Improve code\00. FINALS_POOP"
#test.download_Raw_dataset(dl_path)
