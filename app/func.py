import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import streamlit as st


# <============================================== Data Preparation functions ==============================================>
# >>>>>>>>>> Data Cleanning functions <<<<<<<<<<
# 1. find car characteristics in collected earlier data
def fillna_by_col(data, col):
    df = data.copy()
    unique_vals = df.loc[df.isnull().any(axis=1), col].unique()
    for val in unique_vals:
        df[df[col]==val] = df[df[col]==val].ffill().bfill()
    
    return df

# 2. create brand column from name
def brand_col(data):
    df = data.copy()
    df['brand'] = df['name'].str.split().str[0]
    
    replace_dict = {'Land' : 'Land Rover', 'Mini' : 'Mini Cooper', 'Isuzu' : 'ISUZU'}
    df['brand'] = df['brand'].replace(replace_dict)
    
    return df


# 3. check if all values seem to be correct
def values_check(data, drop_df):
    df = data.copy()
    
    allowed_vals = {
        'fuel': [np.nan, 'Petrol', 'Diesel', 'CNG', 'LPG'],
        'seller_type': [np.nan, 'Individual', 'Dealer'],
        'transmission': [np.nan, 'Manual', 'Automatic'],
        'owner': [np.nan, 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'],
        'seats': [np.nan, '2', '4', '5', '6', '7', '8', '9', '10'],
        'brand': [np.nan, 'Hyundai', 'Mahindra', 'Chevrolet', 'Honda', 'Ford', 'Tata', 'Toyota', 'Maruti', 'BMW', 'Volkswagen', 'Audi', 'Nissan', 'Skoda', 'Mercedes-Benz', 'Datsun', 'Renault', 'Fiat', 'MG', 'Jeep', 'Volvo', 'Kia', 'Land Rover', 'Mitsubishi', 'Jaguar', 'Porsche', 'Mini Cooper', 'ISUZU']
    }
    
    for col, allowed_vals in allowed_vals.items():
        tmp = df[~df[col].isin(allowed_vals)]
        if not tmp.empty:
            tmp['drop_reason'] = f'Wrong value in the "{col}" column'
            drop_df = drop_df.append(tmp)
            df = df[df[col].isin(allowed_vals)].reset_index(drop=True)
        
    return df, drop_df


# 4.1. update of func for unseen data (known_df added as source of known data)
def fillna_stat_by_col(data, known_df, g_col, f_colls, fill_method):
    df = data.copy()
    unique_vals = df.loc[df.isnull().any(axis=1), g_col].unique()
    for val in unique_vals:
        if fill_method=='mean':
            df.loc[df[g_col]==val, f_colls] = df.loc[df[g_col]==val, f_colls].fillna(known_df.loc[known_df[g_col]==val, f_colls].mean(numeric_only=True))
        elif fill_method=='mode':
            df.loc[df[g_col]==val, f_colls] = df.loc[df[g_col]==val, f_colls].fillna(known_df.loc[known_df[g_col]==val, f_colls].mode().iloc[0,:])
        else:
            print(f"Error: method '{fill_method}' is incorrect.")
    
    return df

# 4.2. for filling all nulls with the help of fillna_stat_by_col() func
def all_fillna_stat(data, known_df):
    df = data.copy()
    
    # name (brand), year and owner will be mandatory input fields
    mode_all_col = 'seller_type'
    mode_brand_cols = ['fuel', 'transmission', 'seats']
    mean_brand_cols = ['engine_cc', 'max_power_bhp']
    mean_owner_col = 'km_driven'
        
    na_cols = df.columns[df.isna().any()].tolist()
    
    if mode_all_col in na_cols:
        df[mode_all_col] = known_df[mode_all_col].mode()
    
    if mean_owner_col in na_cols:
        df = fillna_stat_by_col(df, known_df, 'owner', mean_owner_col, 'mean')
    
    inter = lambda lst1, lst2: [value for value in lst1 if value in lst2]
    mode_brand_cols = inter(mode_brand_cols, na_cols)
    df = fillna_stat_by_col(df, known_df, 'brand', mode_brand_cols, 'mode')
    
    mean_brand_cols = inter(mean_brand_cols, na_cols)
    df = fillna_stat_by_col(df, known_df, 'brand', mean_brand_cols, 'mean')
    
    return df


# 5. check and drop nulls
def na_check(data, drop_data):
    df, drop_df = data.copy(), drop_data.copy()
    if df.isna().any().any():
        tmp = df[df.isna().any(axis=1)]
        tmp['drop_reason'] = 'Unable to fill NA'
        drop_df = drop_df.append(tmp)
        df.dropna(inplace=True).reset_index(drop=True)
        
    return df, drop_df



# >>>>>>>>>> Feature Engineering functions <<<<<<<<<<
# 1.1. create '..._flg' column
def col_to_col_flg(data, col, vals_1):
    df = data.copy()
    df[col+'_flg'] = df.apply(lambda row: int(row[col] in vals_1), axis=1)
    return df

# 1.2. create '..._flg' columns with col_to_col_flg() func
def all_col_to_col_flg(data):
    df = data.copy()
    cols_vals_1_dict = {
        'fuel': ['Diesel', 'Petrol'],
        'seats': ['2', '4', '5', '7'],
        'owner': ['First Owner'],
        'seller_type': ['Individual'],
        'transmission': ['Manual']
    }

    for col, vals_1 in cols_vals_1_dict.items():
        df = col_to_col_flg(df, col, vals_1)
        
    return df


# 2. create columns from brand
def brand_by_mean_price(data, known_df):
    df = data.copy()
    mean_price_per_brand = known_df.groupby(['brand']).mean()['selling_price_inr'].sort_values(ascending=False)
    n = len(mean_price_per_brand)

    df['brand_top_half'] = df.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[:n//2].index), axis=1)
    df['brand_top_third'] = df.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[:n//3].index), axis=1)
    df['brand_bottom_third'] = df.apply(lambda row: int(row['brand'] in mean_price_per_brand.iloc[-n//3:].index), axis=1)

    return df


# >>>>>>>>>> Features/target split <<<<<<<<<<
def xy_split(data):
    return data.select_dtypes(include='number').drop(columns='selling_price_inr'), data['selling_price_inr']


# >>>>>>>>>> Main function (unify all functions above) <<<<<<<<<<
def data_prep(data, train_df, y_true_flg=False, skip_dc=False):
    df = data.copy()
    drop_df = pd.DataFrame(columns = ['drop_reason']+list(df.columns))
    
    # Data Cleaning
    if not skip_dc:
        cols = ['name', 'fuel', 'transmission', 'seats', 'engine_cc', 'max_power_bhp']
        df.loc[:, cols] = fillna_by_col(df.loc[:, cols], 'name')
        df = brand_col(df)
        df, drop_df = values_check(df, drop_df)
        df = all_fillna_stat(df, train_df)
        df, drop_df = na_check(df, drop_df)
    
    # Feature Engineering
    df = all_col_to_col_flg(df)
    df = brand_by_mean_price(df, train_df)
    
    # Features/target split
    if y_true_flg:
        X, y = xy_split(df)
        return (X, y), drop_df
    else:
        df = df.select_dtypes(include='number')
        return df, drop_df




# <============================================== Prediction function ==============================================>
# predict with evaluation of model accuracy
def pred_with_scores(model, X, y_true=None):
    y_pred = model.predict(X)
    y_pred = np.round(y_pred, 2)
    if y_true is not None:
        r2 = r2_score(y_true, y_pred)
        n, p = X.shape
        score_dict={
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'adj_R2': 1 - (1 - r2)*(n - 1)/(n - p -1)
        }
        scores = pd.Series(score_dict)
        #st.dataframe(scores)
        return y_pred, scores
        
    return y_pred




# <======================================== Data Preparation + Prediction function ========================================>
def data_prep_and_predict(data, train_df, model, y_true_flg=False, return_drop=True, skip_dc=False):
    df = data.copy()
    df, drop_df = data_prep(df, train_df, y_true_flg=y_true_flg, skip_dc=skip_dc)
    if y_true_flg:
        y_pred, scores = pred_with_scores(model, df[0], df[1])
        return (y_pred, drop_df, scores) if return_drop else (y_pred, scores)
    else:
        y_pred = pred_with_scores(model, df)
        return (y_pred, drop_df) if return_drop else y_pred




# <============================================== Other functions ==============================================>
# Read csv file
def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    df = df.astype({'seats':str})
    df['seats'] = df['seats'].str.replace('.0','', regex=False).replace({'nan':np.nan})

    if 'selling_price_inr' in df.columns:
        cols = list(df.columns)
        cols.remove('selling_price_inr')
        df = df.loc[:, ['selling_price_inr']+cols]
    
    return df


# For displaying prediced prices on the web page
def predict_df(df, y_pred):
    y_pred = pd.Series(y_pred, name='predicted_price_inr')
    return pd.concat([y_pred, df], axis=1)

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False).encode('utf-8')