import pandas as pd
import numpy as np
import datetime
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
import sys

class Fill_NaN_Method(Enum):
    AVERAGE = 1
    MOST_COMMON = 2

# combines different categories that contains a given word into one unique category
def group_values_in_column(column_name:str, value:str):
    categories_to_group = []
    unique_categories = df[column_name].unique()
    for category in unique_categories:
        if value.lower() in str(category).lower():
            categories_to_group.append(category)
    df[column_name] = df[column_name].replace(categories_to_group, value)
    
def fill_nan_most_common_groupby(column_name:str, groupby:list[str]):
    # fill nan in the column with most common value of given group
    most_common_value = df.groupby(groupby)[column_name].transform(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    df[column_name].fillna(most_common_value, inplace=True)
    # rest of the values can be removed because overall data is missing in the group
    df.dropna(subset=[column_name], inplace=True)

def fill_nan_average_groupby(column_name:str, groupby:list[str]):
    # fill nan in the column with average value of given group
    average_value = df.groupby(groupby)[column_name].transform('mean')
    df[column_name].fillna(average_value, inplace=True)
    # rest of the values can be removed because overall data is missing in the group
    df.dropna(subset=[column_name], inplace=True)
    
def remove_outliers_in_column(column_name:str, lower_threshold:float, upper_threshold:float):
    return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]

def print_nan_in_columns():
    nan_counts = df.isna().sum()
    for column, count in nan_counts.items():
        print("No of NaN in", column, ":", count)

#old_df = pd.read_csv("/Users/salihozdemir/Downloads/cars.csv")
df = pd.read_csv("/Users/salihozdemir/Downloads/car_cleared.csv")
df_org = pd.read_csv("/Users/salihozdemir/Downloads/car_cleared.csv")

# place 'price' column to first index
cols = df.columns.tolist()
cols.insert(0, cols.pop(14))
df = df[cols]

# change unknown entries to NaN for dataframe
df = df.replace('-', np.nan)

# drop rows which has a NaN value for some attributes since these are important identifiers for cars
df.dropna(subset=['price'], inplace=True)
df.dropna(subset=['marka'], inplace=True)
df.dropna(subset=['model'], inplace=True)
df.dropna(subset=['seri'], inplace=True)
df.dropna(subset=['arac_turu'], inplace=True)

# drop columns
df.drop('motor_hacmi', inplace=True, axis=1) # useless column because of fetch error
df.drop('tramer_kaydi', inplace=True, axis=1) # has the same information with 'tramer_tutari'
df.drop('sanziman', inplace=True, axis=1) # almost half of it is nan

# add the car age as a column and drop "yil" column
currentYear = datetime.datetime.now().year
df['araba_yasi'] = currentYear - df['yil']
df.drop('yil', inplace=True, axis=1)

# change a value with a similar one for a column because it almost has the same meaning
df['vites_tipi'] = df['vites_tipi'].replace('Yarı Otomatik', 'Otomatik')
df['yakit_tipi'] = df['yakit_tipi'].replace('LPG & Benzin', 'Benzin')

# group relative categories in a column
group_values_in_column(column_name='kasa_tipi', value='Hatchback')
group_values_in_column(column_name='kasa_tipi', value='Station wagon')
group_values_in_column(column_name='kasa_tipi', value='Coupe')
group_values_in_column(column_name='kasa_tipi', value='Roadster')
group_values_in_column(column_name='kasa_tipi', value='Sedan')
group_values_in_column(column_name='kasa_tipi', value='MPV')
group_values_in_column(column_name='renk', value='Gri')
group_values_in_column(column_name='renk', value='Yeşil')
group_values_in_column(column_name='renk', value='Mavi')

# remove false value entries from column
df = df[~df['kasa_tipi'].isin([', 4 Koltuk', ', 2 Koltuk'])]

# group categories as 'other' in a column
cekis_categories_to_keep = ['Önden Çekiş', 'Arkadan İtiş']
df['cekis'] = np.where(df['cekis'].isin(cekis_categories_to_keep), df['cekis'], 'Diğer')

# convert NaN values to most logical value
df['garanti_durumu'] = df['garanti_durumu'].fillna('Garantisi Yok')
df['takasa_uygun'] = df['takasa_uygun'].fillna('Takasa Uygun Değil')
df['aracin_ilk_sahibi'] = df['aracin_ilk_sahibi'].fillna('İlk Sahibi Değilim')

# convert NaN values in tramer_tutari to '0'  ## experimental
df['tramer_tutari'] = df['tramer_tutari'].fillna(0)


# convert column type 
df['motor_gucu'] = df['motor_gucu'].astype(float)

### fill in Nan values for categorical columns:
fill_nan_most_common_groupby(column_name='kasa_tipi', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='renk', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='on_lastik', groupby=['marka', 'seri', 'model'])

### fill in NaN values for numerical columns:
    # if there are too many unique values in column then 'average' method is used, if there are not many
    # unique values in column 'most_common' method is used because it is shared between too many rows so it's easy to find similar data.
    # the nature of the category is also important (for example cylinder count, average method can not be used on such data).
fill_nan_most_common_groupby(column_name='motor_gucu', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='yakit_deposu', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='silindir_sayisi', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='maksimum_guc', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='minimum_guc', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='hizlanma', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='maksimum_hiz', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='sehir_ici_yakit_tuketimi', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='sehir_disi_yakit_tuketimi', groupby=['marka', 'seri', 'model'])
fill_nan_most_common_groupby(column_name='koltuk_sayisi', groupby=['marka', 'seri', 'model'])

fill_nan_average_groupby(column_name='ortalama_yakit_tuketimi', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='yillik_mtv', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='tork', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='uzunluk', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='genislik', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='yukseklik', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='agirlik', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='bos_agirlik', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='bagaj_hacmi', groupby=['marka', 'seri', 'model'])
fill_nan_average_groupby(column_name='aks_araligi', groupby=['marka', 'seri', 'model'])

# these thresholds can be updated based on the models performance
df = remove_outliers_in_column(column_name='price', lower_threshold=60000, upper_threshold=5000000)
df = remove_outliers_in_column(column_name='kilometre', lower_threshold=0, upper_threshold=500000)
df = remove_outliers_in_column(column_name='motor_gucu', lower_threshold=50, upper_threshold=500)
df = remove_outliers_in_column(column_name='yillik_mtv', lower_threshold=219, upper_threshold=50000)
df = remove_outliers_in_column(column_name='hizlanma', lower_threshold=3.2, upper_threshold=20)
df = remove_outliers_in_column(column_name='bagaj_hacmi', lower_threshold=100, upper_threshold=850)
df = remove_outliers_in_column(column_name='tramer_tutari', lower_threshold=0, upper_threshold=200000)
df = remove_outliers_in_column(column_name='araba_yasi', lower_threshold=0, upper_threshold=33)



#import seaborn as sns
#import matplotlib.pyplot as plt

#numerical_columns = ['kilometre', 'motor_gucu', 'ortalama_yakit_tuketimi', 
#                     'yakit_deposu', 'yillik_mtv', 'silindir_sayisi', 'tork', 
#                     'maksimum_guc', 'minimum_guc', 'hizlanma', 'maksimum_hiz', 
#                     'sehir_ici_yakit_tuketimi', 'sehir_disi_yakit_tuketimi', 'uzunluk', 
#                     'genislik', 'yukseklik', 'agirlik', 'bos_agirlik', 'koltuk_sayisi', 
#                     'bagaj_hacmi', 'aks_araligi', 'tramer_tutari', 'araba_yasi']
#for column in numerical_columns:
#    plt.figure()  # Create a new figure for each column
#    sns.distplot(df[column].astype(float))
#    plt.title(column)
#    plt.show()



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# gaussian (standardization, z-score): kilometre, hizlanma, uzunluk, genislik, yukseklik, agirlik, bos_agirlik, araba_yasi
# not gaussian (normalization, min-max): motor_gucu, ortalama_yakit_tuketimi, yakit_deposu, yillik_mtv, silindir_sayisi,
#               tork, maksimum_guc, minimum_guc, maksimum_hiz, sehir_ici_yakit_tuketimi,
#               sehir_disi_yakit_tuketimi, koltuk_sayisi, bagaj_hacmi, aks_araligi, tramer_tutari

standardization_columns = ['kilometre', 'hizlanma', 'uzunluk', 'genislik', 'yukseklik', 
                           'agirlik', 'bos_agirlik', 'araba_yasi']
standardization_scaler = StandardScaler()
df[standardization_columns] = standardization_scaler.fit_transform(df[standardization_columns])

normalization_columns = ['price', 'motor_gucu', 'ortalama_yakit_tuketimi', 'yakit_deposu', 
                         'yillik_mtv', 'silindir_sayisi', 'tork', 'maksimum_guc', 
                         'minimum_guc', 'maksimum_hiz', 'sehir_ici_yakit_tuketimi', 
                         'sehir_disi_yakit_tuketimi', 'koltuk_sayisi', 'bagaj_hacmi', 
                         'aks_araligi', 'tramer_tutari']
normalization_scaler = MinMaxScaler()
df[normalization_columns] = normalization_scaler.fit_transform(df[normalization_columns])



import category_encoders as ce
categorical_columns = ['marka', 'seri', 'model', 'vites_tipi', 'yakit_tipi', 
                       'kasa_tipi', 'cekis', 'takasa_uygun', 'arac_turu', 'renk', 
                       'garanti_durumu', 'aracin_ilk_sahibi', 'on_lastik']

# one-hot encoding (nominal and low cardinality): marka, vites_tipi, yakit_tipi, kasa_tipi, cekis, takasa_uygun
# ordinal encoding (ordinal): garanti_durumu, aracin_ilk_sahibi, arac_turu
# target encoding (nominal, high cardinality): seri, model, on_lastik, renk
# frequency encoding (nominal, high cardinality): 

# vites_tipi : ['Düz' 'Otomatik']
# takasa_uygun : ['Takasa Uygun' 'Takasa Uygun Değil']
# garanti_durumu : ['Garantisi Yok' 'Garantisi Var']            # ordinal?
# aracin_ilk_sahibi : ['İlk Sahibi Değilim' 'İlk Sahibiyim']    # ordinal?
# arac_turu : ['Bireysel' 'Şirket' 'Taksi']                     # ordinal?
# kasa_tipi : ['Hatchback' 'Station wagon' 'Sedan' 'Coupe' 'Roadster' 'MPV' 'SUV' 'Cabrio']
# yakit_tipi : ['Benzin' 'Dizel' 'Hibrit' 'Elektrik']
# cekis : ['Önden Çekiş' 'Diğer' 'Arkadan İtiş']

#for column in categorical_columns:
#    print(f'{column} : {len(df[column].unique())}')

columns_to_one_hot_encode = ['marka', 'vites_tipi', 'yakit_tipi', 'kasa_tipi', 'cekis', 'takasa_uygun']
one_hot_encoder = ce.OneHotEncoder(cols=columns_to_one_hot_encode, use_cat_names=True)
df = one_hot_encoder.fit_transform(df)

ordinal_mapping = {'Garantisi Var': 1, 'Garantisi Yok': 0}
ordinal_encoder = ce.OrdinalEncoder(mapping=[{'col': 'garanti_durumu', 'mapping': ordinal_mapping}])
df = ordinal_encoder.fit_transform(df)
ordinal_mapping = {'İlk Sahibiyim': 1, 'İlk Sahibi Değilim': 0}
ordinal_encoder = ce.OrdinalEncoder(mapping=[{'col': 'aracin_ilk_sahibi', 'mapping': ordinal_mapping}])
df = ordinal_encoder.fit_transform(df)
ordinal_mapping = {'Bireysel': 2, 'Şirket': 1, 'Taksi': 0}
ordinal_encoder = ce.OrdinalEncoder(mapping=[{'col': 'arac_turu', 'mapping': ordinal_mapping}])
df = ordinal_encoder.fit_transform(df)

target_encoder = ce.TargetEncoder(cols=['model', 'seri'])
df = target_encoder.fit_transform(df, df['price'])

frequency_encoder = ce.CountEncoder(cols=['on_lastik', 'renk'])
df = frequency_encoder.fit_transform(df)

"""from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor

X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TPOTRegressor with desired configuration
tpot = TPOTRegressor(generations=5, population_size=10, scoring='neg_mean_squared_error', cv=5, random_state=42, verbosity=2)

# Fit the TPOTRegressor on the training data
tpot.fit(X_train, y_train)

# Access the performance of different algorithms
algorithm_scores = tpot.evaluated_individuals_

# Print the scores for each algorithm
for algorithm, score in algorithm_scores.items():
    print(f"{algorithm}: {score}")

# Evaluate the best model on the test set
y_pred = tpot.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")"""



#df = df.filter(lambda column: 'marka' not in column, axis=1)
columns_to_keep = ['price', 'seri', 'model', 'araba_yasi', 'kilometre', 'yillik_mtv', 
                   'motor_gucu', 'vites_tipi_Otomatik', 'vites_tipi_Düz', 
                   'garanti_durumu']
df = df[columns_to_keep]



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at each split
}

model = RandomForestRegressor()
# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Print the best parameters and mean squared error
print("Best Parameters:", best_params)
print("Mean Squared Error:", mse)"""

import matplotlib.pyplot as plt
import seaborn as sns

# Create and train model
rf = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 7, random_state = 18)
rf.fit(X_train, y_train)

# Predict on test data
prediction = rf.predict(X_test)

# Compute mean squared error
mse = mean_squared_error(y_test, prediction)

# Compute R-squared score
r2_score = rf.score(X_test, y_test)

# Plot predicted vs. actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=prediction, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

# Plot residuals
residuals = y_test - prediction
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Actual Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.show()

# Plot feature importances
importances = rf.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 12))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.xticks(rotation=90)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.show()

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", mse**0.5)
print("R-squared Score:", r2_score)

from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

dump(rf, 'random_forest_regressor.joblib')

# selected features:    model, seri, yillik_mtv, araba_yasi, motor_gucu, kilometre, 
#                       vites_tipi_duz, vites_tipi_otomatik, garanti_durumu
# target encoding: model, seri
# ordinal encoding: garanti_durumu
# standardization: kilometre, araba_yasi
# normalization: motor_gucu, yillik_mtv
columns_to_transform = ['kilometre', 'araba_yasi']

target_encoder.fit(df, df['price'])
standardization_scaler.fit(df['kilometre'])

dump(standardization_scaler, 'std_scaler.joblib')
dump(normalization_scaler, 'norm_scaler.joblib')
dump(target_encoder, 'target_encoder.joblib')

# seri, model, araba_yasi, kilometre, yillik_mtv, motor_gucu, vites_tipi_otomatik, vites_tipi_duz, garanti_durumu
new_data = pd.read_csv("/Users/salihozdemir/Downloads/new_data.csv")
new_data = target_encoder.transform(new_data)

new_data = standardization_scaler.transform(new_data)

