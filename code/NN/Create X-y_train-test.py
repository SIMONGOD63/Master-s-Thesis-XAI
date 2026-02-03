#%% Import part
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import accuracy_score



#os.chdir(r"C:\Users\simeo\A. Data Science\Master thesis\02. Code")
path = r"C:\Users\simeo\A. Data Science\Master thesis\01. Data"

# from working_DS_1707 => X_post_preprocess
data = pd.read_csv(path + "/00. FINALS/X_post_preprocess.csv", index_col = 0)
y_og = pd.read_csv(path +"/00. FINALS/y_post_preprocess.csv",index_col = 0)

# Alignement check
assert (data.index.values == y_og.index.values).all()
print("Check for alignement : passed")

# Remove the unscaled useless variables
data.drop(columns = ["lat","long"], inplace = True)



#--- Function to get performances metrics

def get_met(test, pred):
    
    # get the clean y_test
    c_test =np.array(test).argmax(axis = 1)
    
    # get the clean y_pred
    c_pred = np.array(pred).argmax(axis = 1)
    
    acc = accuracy_score(c_test, c_pred)
    #f1 = f1_score(c_test, c_pred)
    f1 = 1
    return acc


conv_let = {
    0 : "CAQ" ,
    1: "PQ",
    2: "PLQ",
    3: "QS",
    4: "PCQ",
    5: "Other"
    }
    
conv_numb = {
    "CAQ":0 ,
    "PQ":1,
    "PLQ": 2,
    "QS":3,
    "PCQ":4,
    "Other":5
    }

del(conv_numb,conv_let)


#%% First transformations
# Merge Other and Did not vote
data.loc[data["op_intent_no_vote"] == 1, 'op_intent_Other'] = 1
data.drop(columns = ['op_intent_no_vote'],inplace = True)

# Delete the two low occuring classes Other and Did not vote
idx_other = (data.index[data.loc[:,'op_intent_Other'] == 1])
data.drop(index = idx_other, inplace = True)
data.drop(columns = ['op_intent_Other'], inplace = True)

# Change it in Y matrix
y_og["op_intent"] = y_og['op_intent'].apply(lambda x : "Other" if x == "Did not vote" else x)
idx_del = y_og.index[y_og["op_intent"] == "Other"]
y_og.drop(index = idx_del, inplace = True)


## Keep the impute indicator to see how my model reacts to it
#data.drop(columns = ['impt_ind'],inplace = True)

#----- Create a class dic
op_macro = [col for col in data.columns if col.startswith("op_intent_") ]

# Drop what's unnecessary now
data.drop(columns = op_macro,inplace = True)

c_dic = dict(len(y_og)/(y_og['op_intent'].nunique() * y_og['op_intent'].value_counts()))
c_dic = { "op_intent_" + key:val for key,val in c_dic.items() }

del(idx_other)

print("Data shape (first pipeline):", data.shape) # 44 426, 168 => 162 ? why The genders.
print("Y shape (first pipeline):", y_og.shape)
#%% Gotta remove the variables vote_pred and op_intent from the dataset

# Delete the response variable of the previous study
vote_macro = [col for col in data.columns if col.startswith("vote_pred") and col != "vote_pred"]

# Keep it as accuracy benchmark.
y_benchmark = data[vote_macro]

# Drop what's not necessary
data.drop(columns = vote_macro,inplace = True)

#----- Put the variable people_predict into k-1 encoded form ! 

pp_macro = [col for col in data.columns if col.startswith("people_pred")]

data.drop(columns = pp_macro[-1], inplace = True)
del(vote_macro,op_macro,pp_macro)

#%% Prepare the data

assert (data.index.to_numpy() == y_og.index.to_numpy()).all()

## Train and test set
X_train, X_test, y_train, y_test = train_test_split(data,y_og,test_size = 0.2, random_state = 4243, stratify=y_og["op_intent"])


assert (X_train.index.values == y_train.index.values).all()
assert (X_test.index.values == y_test.index.values).all()
#%%------------- RandomOversampling
from imblearn.over_sampling import RandomOverSampler
untouched_samples = y_train["op_intent"].value_counts()

# Check the types
X_train.dtypes
X_test.dtypes


#-- Get the new desired proportion for each class (I tried other but this one works better)
# Gives the best performance is SS
desired_SS = { key : round(1.33*untouched_samples[key]) for key in untouched_samples.index.values.tolist()}
desired_SS['QS'] = untouched_samples['QS']


#-- Initialize the random oversampler
OS = RandomOverSampler(
    sampling_strategy = desired_SS,    
    random_state= 4243)

# Keep the col names
col_names = X_train.columns.tolist()
y_names = y_train.columns.tolist()


# Oversample
X_train_OS, y_train_OS = OS.fit_resample(np.array(X_train),np.array(y_train["op_intent"]))

# Put back into dataFrame format
X_train_OS = pd.DataFrame(data=X_train_OS, columns=col_names)
y_train_OS = pd.DataFrame(data= y_train_OS, columns = ["op_intent"])
# Rename with the correct name
X_train_OS.rename(columns = {"absolute_idx.1" : "absolute_idx"},inplace= True)
X_test.rename(columns = {"absolute_idx.1" : "absolute_idx"},inplace= True)

#%% Re-order my dataset post oversampling 

X_train_OS["absolute_idx"].value_counts().head()
#############"#############"#############"#############"#############"
# HERE I NEED TO HANLDE INDEXES KEEPING IT SOMEHOW => then handle doublon => or not just keep the absolute_idx index
#############"#############"#############"#############"#############"
# Check to be sure no errors nor NaN
assert X_train_OS.isna().any(axis = 1).sum() == 0

# I can do that since I know they are still aligned
y_train_OS["absolute_idx2"] = X_train_OS["absolute_idx"]


# FInal shape check
assert y_train_OS.shape[0] == X_train_OS.shape[0]

# Check if NA values introduced
print("Expected 0 :",y_train_OS["absolute_idx2"].isna().any().sum())

# Shuffle mandatory after Oversampling
from sklearn.utils import shuffle
X_train_OS, y_train_OS = shuffle(X_train_OS,y_train_OS, random_state = 4243)


#%% Check still aligned
# X_train_OS has the index but they are aligned. 

assert (X_train_OS["absolute_idx"].to_numpy() == y_train_OS["absolute_idx2"].to_numpy()).all()
# I don't have absolute_idx2 in y, why ?
print("Still aligned")

#%% one hot encode y_train

y_train_OS["op_intent"] = y_train_OS["op_intent"].astype("category")

y_one = pd.get_dummies(y_train_OS["op_intent"],prefix ="op_intent",prefix_sep="_",dtype=int)

# final y dataset
y_train_final = pd.concat(
    [
        y_train_OS[["absolute_idx2", "op_intent"]],  # raw + identity
        y_one                                   # encoded response
    ],
    axis=1)

assert (X_train_OS["absolute_idx"].to_numpy() == y_train_final["absolute_idx2"].to_numpy()).all()

#%% Download the training sets
X_train_OS.to_csv(path + "/00. FINALS/X_train_aligned.csv")
y_train_final.to_csv(path +"/00. FINALS/y_train_aligned.csv")

#%% Prepare and download the testing sets

y_test_clean = y_test.reset_index(drop=True).copy()
X_test_clean = X_test.reset_index(drop=True).copy()

y_test_clean["absolute_idx2"] = X_test_clean["absolute_idx"].values

assert len(y_test_clean) == len(X_test_clean)

# One hot encode it
y_test_onehot = pd.get_dummies(
    y_test_clean["op_intent"],
    prefix="op_intent", dtype= int
)

y_test_final = pd.concat(
    [
        y_test_clean[["absolute_idx2", "op_intent"]],
        y_test_onehot
    ],
    axis=1
)

assert (X_test_clean["absolute_idx"].to_numpy() == y_test_final["absolute_idx2"].to_numpy()).all()

X_test_clean.to_csv(path + "/00. FINALS/X_test_aligned.csv")
y_test_final.to_csv(path +"/00. FINALS/y_test_aligned.csv")

#%% Test if it works with the neural network model => 

X_train_OS.drop(columns = ["orig_index"],inplace= True);
X_test.drop(columns = ["orig_index"],inplace= True);


#%% Test if the NN model works

# It works.
