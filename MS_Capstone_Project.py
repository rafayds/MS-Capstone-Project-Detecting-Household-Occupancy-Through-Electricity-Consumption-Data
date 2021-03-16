#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import os
import warnings

from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from matplotlib import pyplot as plt
from matplotlib import dates, ticker


# In[ ]:


target = 'occupied'
seasons = ['summer', 'winter']


# In[ ]:


# converts indices of dataframes to datetime (since when loaded from CSV they become string-typed)

def fix_index(df):
    return df.set_index(pd.to_datetime(df.index))

# extracts the rows from dataframes whose index values are between 6 AM and 10 PM

def filter_waking_hours(df):
    dt_index = pd.to_datetime(df.index)
    hours = dt_index.to_series().dt.hour
    return df.set_index(dt_index)[(hours >= 6) & (hours <= 21)]

def engineer_meter_features(df, groups, aggregations):
    return pd.concat([df[[f'{group}{i}' for i in range(1, 4)]]
                      .agg(aggregation, axis=1)
                      .rename(f'{group}_{aggregation}') 
                      for group in groups 
                      for aggregation in aggregations], axis=1)

# adds features to each row that are equal to the value of corresponding features from earlier timesteps

def lag(df, timesteps):
    return pd.concat([df.shift(i).add_suffix(f'_lag_{i}') for i in range(timesteps)], axis=1).fillna(method='bfill')

#converts df from wide format to long format and resamples into 15 min intervals

def process_occupancy_df(df):
    long_df = df.reset_index().melt(id_vars='index')
    return (pd.concat([long_df['value'].rename('occupied'), 
               pd.to_datetime(long_df['index'] + long_df['variable'], format="%d-%b-%Y'%H:%M:%S'").rename('datetime')], axis=1)
            .set_index('datetime')                             
            .resample('15min')   
            .first()                               
            .dropna())

# divides arrays into a tuple of consecutive values (used for plotting occupancy periods)

def consecutive(data):
    return np.split(data, np.where(np.diff(data) != 1)[0] + 1)

# plots the "day in the life of" plots

def plot_dil(meter, occupancy, ax, when, min_value, max_value):
    ax.plot(meter.index, meter.values, color='black')
    ax.tick_params(labelsize=12)
    ax.axis(xmin=meter.index.min(), xmax=meter.index.max())

    ax.set_ylabel(f'Min-max Scaled Power Consumption', fontsize=13)
    ax.set_title(f'A Day in the Life of Household 1 ({when})', fontsize=14)
    ax.set_xlabel('Hour', fontsize=13)

# create a separate Axes sharing the x-axis
    
    ax2 = ax.twinx()
    
    groups = consecutive(np.where(1 - occupancy.values)[0])
    
# color each group, which corresponds to an unoccupied time
    
    for i, group in enumerate(groups):
        start_edge = group[0]
        end_edge = group[-1]
        ax2.axvspan(occupancy.index[start_edge], occupancy.index[end_edge], color='goldenrod', alpha=0.6, **({'label': 'Unoccupied'} if i == 0 else {}))

    ax2.tick_params(right=False, labelright=False)
    ax2.xaxis.set_major_locator(dates.HourLocator())
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x % 1 / (1 / 24):.0f}'))
    ax2.axis(ymin=0, ymax=1)
    
    ax2.set_ylabel(f'min = {min_value:.1f}, max = {max_value:.1f}', fontsize=13)

    ax2.legend(loc='upper left')
    
# split a dataframe at a given point temporally
    
def time_train_validation_split(df, test_size):
    split_index = int(len(df) * (1 - test_size))
    split_point = df.index[split_index]
    return df[df.index < split_point], df[df.index >= split_point]

def prepare_X_y(df):
    X = df.drop(columns=['occupied', 'household'])
    y = df['occupied']
    household = df['household']
    return X, y, household

def evaluate_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {'accuracy': accuracy_score(y_true, y_pred), 
            'precision': precision_score(y_true, y_pred), 
            'recall': recall_score(y_true, y_pred), 
            'f1_score': f1_score(y_true, y_pred), 
            'ROC-AUC score': roc_auc_score(y_true, y_pred_proba)}

# given a set of splits (train, validation) and a test dataset, train and evaluate models

def evaluate_splits_models(summer_splits, winter, models, thresholds):
    train = pd.concat(summer_splits[0], axis=0)
    valid = pd.concat(summer_splits[1], axis=0)
    X_train, y_train, household_train = prepare_X_y(train)
    X_valid, y_valid, household_valid = prepare_X_y(valid)
    X_test, y_test, household_test = prepare_X_y(winter)
    fit_models = [model.fit(X_train, y_train) for model in models]
    summer_predictions = [model.predict_proba(X_valid)[:, 1] for model in models]
    winter_predictions = [model.predict_proba(X_test)[:, 1] for model in models]
    return {model.__class__.__name__: {'summer': (evaluate_metrics(y_valid, summer_prediction, threshold), 
                                                  np.stack([summer_prediction, household_valid], axis=1)),
                                       'winter': (evaluate_metrics(y_test, winter_prediction, threshold), 
                                                  np.stack([winter_prediction, household_test], axis=1))}
            for model, summer_prediction, winter_prediction, threshold in zip(models, summer_predictions, winter_predictions, thresholds)}


# In[ ]:


#create smart meter dataframe csv files for all households

meter_headers = ['power_all_phases', 'power_l1', 'power_l2', 'power_l3', 'current_neutral', 'current_l1', 'current_l2', 'current_l3', 'voltage_l1', 'voltage_l2', 'voltage_l3', 'phase_voltage_l1_l2', 'phase_voltage_l1_l3', 'phase_current_voltage_l1', 'phase_current_voltage_l2', 'phase_current_voltage_l3']
groups = ['power_l', 'current_l', 'voltage_l', 'phase_current_voltage_l']
aggregations = ['std', 'min', 'max']

for i in range(1, 6):
    household = f'0{i}'
    if not os.path.isfile(f'meter_{household}.csv'):
        # get all qualifying filenames
        filenames = [filename for filename in os.listdir(household) if filename.endswith('.csv') and not filename.startswith('._')]
        # for each dataframe, create a datetime range index, resample to 15 minute intervals and combine
        meter_df = pd.concat([pd.read_csv(os.path.join(household, filename), header=None, names=meter_headers)                      
                              .set_index(pd.date_range(os.path.splitext(filename)[0], periods=86400, freq='s').rename('datetime'))  
                              .resample('15min')                                                                                    
                              .mean()
                              
                              for filename in filenames])
        # perform the simple aggregations detailed above
        engineered_features = engineer_meter_features(meter_df, groups, aggregations)
        combined_df = pd.concat([meter_df, engineered_features], axis=1)
        combined_df.to_csv(f'meter_{household}.csv')
        
meter_dfs = [fix_index(pd.read_csv(f'meter_0{i}.csv', index_col=0)) for i in range(1, 6)]
filtered_meter_dfs = [filter_waking_hours(meter_df) for meter_df in meter_dfs]


# In[ ]:


for i in range(1, 6):
    household = f'0{i}'
    # process each occupancy file separately (5 households x 2 seasons)
    summer_filename = f'occupancy_summer_{household}.csv'
    winter_filename = f'occupancy_winter_{household}.csv'
    if not os.path.isfile(summer_filename):
        summer_df = process_occupancy_df(pd.read_csv(os.path.join(f'{household}_occupancy', f'{household}_summer.csv'), index_col=0))
        summer_df.to_csv(summer_filename)
        
    if not os.path.isfile(winter_filename):
        winter_df = process_occupancy_df(pd.read_csv(os.path.join(f'{household}_occupancy', f'{household}_winter.csv'), index_col=0))
        winter_df.to_csv(winter_filename)

occupancy_dfs = [{'summer': pd.read_csv(f'occupancy_summer_0{i + 1}.csv', index_col=0), 'winter': pd.read_csv(f'occupancy_winter_0{i + 1}.csv', index_col=0)} for i in range(5)]


# In[ ]:


appliances = ['fridge', 'dryer', 'coffee machine', 'kettle', 'washing machine', 'pc', 'freezer']

#fill list with one df per appliance 
for i in range(1, 6):
    if not os.path.isfile(f'appliance_0{i}.csv'):
        folder = f'0{i}_plugs'
        sub_dfs = []
        for appliance_id in range(1, 8):
            directory = os.path.join(folder, f'0{appliance_id}')

            # Create one DataFrame per appliance

            sub_dfs.append(pd.concat([pd.read_csv(os.path.join(directory, filename), header=None, names=[appliance_id])
                                      .set_index(pd.date_range(os.path.splitext(filename)[0], periods=86400, freq='s').rename('datetime'))
                                      .resample('15min')
                                      .mean() 
                                      for filename in os.listdir(directory) if filename.endswith('.csv') and not filename.startswith('._')], axis=0))

        # This DataFrame contains all the appliance data for a single household

        df = pd.concat(sub_dfs, axis=1).rename(columns=dict(enumerate(appliances, 1))) #df for whole household
        df.to_csv(f'appliance_0{i}.csv')
    
appliance_dfs = [fix_index(pd.read_csv(f'appliance_0{i + 1}.csv', index_col=0)) for i in range(5)]


# # Mean Daily Power Consumption by Household

# In[ ]:


fig, ax = plt.subplots(figsize=(14, 6))

for household, meter_df in enumerate(meter_dfs, 1):
    meter_df.set_index(pd.to_datetime(meter_df.index, format='%Y-%m-%d %H:%M:%S')).resample('1d')['power_all_phases'].mean().plot(ax=ax, label=f'Household {household}')

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f} kWh'))
    
ax.xaxis.set_major_formatter(dates.DateFormatter('%B'))
ax.xaxis.set_major_locator(dates.MonthLocator())

ax.set_title('Mean Daily All-phase Power Consumption by Household')
    
ax.legend()

fig.savefig('mean_daily_consumption.png')


# # Mean Power Seasonal Comparison

# In[ ]:


winter_start = datetime(2012, 12, 1)
seasonal_means = [(meter_df.loc[meter_df.index < winter_start, 'power_all_phases'].mean(), meter_df.loc[meter_df.index >= winter_start, 'power_all_phases'].mean()) for meter_df in meter_dfs]

fig, ax = plt.subplots(figsize=(8, 6))

for i, means in enumerate(seasonal_means, 1):
    summer_mean, winter_mean = means
    ax.bar([i - .1], [summer_mean], width=0.2, fc='orange', ec='black', **({'label': 'summer'} if i == 1 else {}))
    ax.bar([i + .1], [winter_mean], width=0.2, fc='steelblue', ec='black', **({'label': 'winter'} if i == 1 else {}))
    
ax.tick_params(labelsize=13)

ax.set_xlabel('Household', fontsize=13)
ax.set_ylabel('Mean Power Consumption', fontsize=13)

ax.legend(fontsize=13)

fig.savefig('mean_seasonal_consumption.png')


# # A Day In The Life of Household 1 (Total Power)

# In[ ]:


dil_h1_meter = meter_dfs[0]
dil_h1_occupancy_summer = fix_index(occupancy_dfs[0]['summer'])
dil_h1_occupancy_winter = fix_index(occupancy_dfs[0]['winter'])

dil_meter_dt = pd.to_datetime(dil_h1_meter.index).to_series().dt
dil_occupancy_dt_summer = pd.to_datetime(dil_h1_occupancy_summer.index).to_series().dt
dil_occupancy_dt_winter = pd.to_datetime(dil_h1_occupancy_winter.index).to_series().dt

dil_day_summer = 20
dil_month_summer = 7
dil_day_winter = 20
dil_month_winter = 12

dil_day_meter_summer = dil_h1_meter.loc[(dil_meter_dt.day == dil_day_summer) & (dil_meter_dt.month == dil_month_summer), 'power_all_phases']
dil_day_meter_summer_min = dil_day_meter_summer.min()
dil_day_meter_summer_max = dil_day_meter_summer.max()
dil_day_meter_summer_scaled = (dil_day_meter_summer - dil_day_meter_summer_min) / (dil_day_meter_summer_max - dil_day_meter_summer_min)
dil_day_occupancy_summer = dil_h1_occupancy_summer.loc[(dil_occupancy_dt_summer.day == dil_day_summer) & (dil_occupancy_dt_summer.month == dil_month_summer), 'occupied']

dil_day_meter_winter = dil_h1_meter.loc[(dil_meter_dt.day == dil_day_winter) & (dil_meter_dt.month == dil_month_winter), 'power_all_phases']
dil_day_meter_winter_min = dil_day_meter_winter.min()
dil_day_meter_winter_max = dil_day_meter_winter.max()
dil_day_meter_winter_scaled = (dil_day_meter_winter - dil_day_meter_winter_min) / (dil_day_meter_winter_max - dil_day_meter_winter_min)
dil_day_occupancy_winter = dil_h1_occupancy_winter.loc[(dil_occupancy_dt_winter.day == dil_day_winter) & (dil_occupancy_dt_winter.month == dil_month_winter), 'occupied']


# In[ ]:


fig, axes = plt.subplots(figsize=(10, 12), nrows=2)

ax1, ax2 = axes

plot_dil(dil_day_meter_summer_scaled, dil_day_occupancy_summer, ax1, f'Summer - {dil_day_summer}/{dil_month_summer}', dil_day_meter_summer_min, dil_day_meter_summer_max)
plot_dil(dil_day_meter_winter_scaled, dil_day_occupancy_winter, ax2, f'Winter - {dil_day_winter}/{dil_month_winter}', dil_day_meter_winter_min, dil_day_meter_winter_max)

fig.savefig('day_in_life_seasonal.png')


# # A Day in the Life of Household 4 (Appliance Plug Data)

# In[ ]:


adv_h4_appliance = appliance_dfs[3]
adv_h4_occupancy = fix_index(occupancy_dfs[3]['summer'])

adv_appliance_dt = pd.to_datetime(adv_h4_appliance.index).to_series().dt
adv_occupancy_dt = pd.to_datetime(adv_h4_occupancy.index).to_series().dt

adv_day = 19
adv_month = 8

# get the specific day that meets the above day/month requirements

adv_day_appliance = adv_h4_appliance[(adv_appliance_dt.day == adv_day) & (adv_appliance_dt.month == adv_month)]

# do the same thing for occupancy data

adv_day_occupancy = adv_h4_occupancy.loc[(adv_occupancy_dt.day == adv_day) & (adv_occupancy_dt.month == adv_month), 'occupied']

# min-max scale so we get a range between 0 and 1

adv_day_appliance_scaled = (adv_day_appliance - adv_day_appliance.min()) / (adv_day_appliance.max() - adv_day_appliance.min())

edges = np.where(1 - adv_day_occupancy.values)[0]
start_edge = edges[0]
end_edge = edges[-1]


# In[ ]:


fig, axes = plt.subplots(figsize=(16, 20), ncols=2, nrows=4, gridspec_kw={'hspace': 0.3})

for ax1, col in zip(axes.flat, adv_day_appliance_scaled.columns):
    data = adv_day_appliance_scaled[col]
    ax1.plot(data.index, data.values, color='black')
    
    ax1.tick_params(labelsize=12)
    
    ax1.set_xlabel('Hour', fontsize=13)
    ax1.set_title(col.capitalize(), fontsize=14)
    
    ax1.axis(xmin=data.index.min(), xmax=data.index.max())
    
    ax1.set_ylabel(f'Min-max Scaled Power Consumption', fontsize=13)
    
    ax2 = ax1.twinx()

    ax2.axvspan(data.index[0], data.index[start_edge], color='skyblue', alpha=0.5, label='Occupied')
    ax2.axvspan(data.index[start_edge], data.index[end_edge], color='r', alpha=0.5, label='Unoccupied')
    ax2.axvspan(data.index[end_edge], data.index[-1], color='skyblue', alpha=0.5)

    ax2.tick_params(right=False, labelright=False)

    ax2.xaxis.set_major_locator(dates.HourLocator())
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x % 1 / (1 / 24):.0f}'))

    ax2.axis(ymin=0, ymax=1)

    ax2.set_xlabel('Hour', fontsize=13)

    ax2.legend()

axes[-1, -1].set_visible(False)
    
fig.suptitle('A Day in the Life of Household 4', fontsize=14, y=.91)

fig.savefig('day_in_life_plug.png')


# In[ ]:


models = [LogisticRegressionCV(cv=5, max_iter=500, class_weight='balanced', random_state=0),
          RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, class_weight='balanced_subsample'),
          GradientBoostingClassifier(random_state=0)]


# In[ ]:


vanilla_df_splits = list(zip(*[time_train_validation_split(meter_df
                                                           .join(occupancy_df['summer'], how='left')
                                                           .dropna(subset=['occupied'])
                                                           .assign(household=i), 0.2)
                               for i, (meter_df, occupancy_df)
                               in enumerate(zip(filtered_meter_dfs, occupancy_dfs), 1)]))

lagged_df_splits = list(zip(*[time_train_validation_split(lag(meter_df, 4)
                                                              .join(occupancy_df['summer'], how='left')
                                                              .dropna(subset=['occupied'])
                                                              .assign(household=i), 0.2)
                               for i, (meter_df, occupancy_df)
                               in enumerate(zip(filtered_meter_dfs, occupancy_dfs), 1)]))

appliance_df_splits = list(zip(*[time_train_validation_split(lag(meter_df, 4)
                                                             .join(appliance_df, how='left')
                                                             .fillna(-1)                           
                                                             .join(occupancy_df['summer'], how='left')                          
                                                             .dropna(subset=['occupied'])                         
                                                             .assign(household=i), 0.2)
                                 for i, (meter_df, appliance_df, occupancy_df)
                                 in enumerate(zip(filtered_meter_dfs, appliance_dfs, occupancy_dfs), 1)]))

vanilla_df_winter = pd.concat([meter_df              
                               .join(occupancy_df['winter'], how='left')
                               .dropna(subset=['occupied'])
                               .assign(household=i)
                               for i, (meter_df, occupancy_df)
                               in enumerate(zip(filtered_meter_dfs, occupancy_dfs), 1)], axis=0)

lagged_df_winter = pd.concat([lag(meter_df, 4)
                              .join(occupancy_df['winter'], how='left')
                              .dropna(subset=['occupied'])
                              .assign(household=i)
                              for i, (meter_df, occupancy_df)
                              in enumerate(zip(filtered_meter_dfs, occupancy_dfs), 1)], axis=0)

appliance_df_winter = pd.concat([lag(meter_df, 4)
                                 .join(appliance_df, how='left')
                                 .fillna(-1)        
                                 .join(occupancy_df['winter'], how='left')   
                                 .dropna(subset=['occupied'])      
                                 .assign(household=i)
                                 for i, (meter_df, appliance_df, occupancy_df)
                                 in enumerate(zip(filtered_meter_dfs, appliance_dfs, occupancy_dfs), 1)], axis=0)


# In[ ]:


# getting the baseline accuracy for summer

print(pd.concat([pd.concat(pair) for pair in zip(*vanilla_df_splits)])['occupied'].mean())

# same for winter

print(vanilla_df_winter['occupied'].mean())


# In[ ]:


if not os.path.isfile('results.pkl'):
    results = {}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        results['vanilla'] = evaluate_splits_models(vanilla_df_splits, vanilla_df_winter, models, [0.25, 0.5, 0.5])
        results['lagged'] = evaluate_splits_models(lagged_df_splits, lagged_df_winter, models, [0.25, 0.5, 0.5])
        results['appliance_lagged'] = evaluate_splits_models(appliance_df_splits, appliance_df_winter, models, [0.25, 0.5, 0.5])
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
else:
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)

evaluation_df = pd.DataFrame({(season, dataset, model.__class__.__name__): result[model.__class__.__name__][season][0]
                              for dataset, result in results.items()
                              for model in models
                              for season in seasons})


# In[ ]:


evaluation_df
evaluation_df.to_csv('evaluation.csv')


# In[ ]:


model_abbreviations = {
    'LogisticRegressionCV': 'LR',
    'RandomForestClassifier': 'RF',
    'GradientBoostingClassifier': 'GBT'
}

fig, axes = plt.subplots(figsize=(12, 20), nrows=3)

colours = {'summer': 'orange', 'winter': 'skyblue'}

ax3 = axes[-1]

for ax, season in zip(axes, seasons):
    # create one bar for the score of each model-dataset-season combination
    for i, ((dataset, model_name), score) in enumerate(evaluation_df[season].loc['ROC-AUC score'].sort_values(ascending=False).iteritems()):
        ax.barh(f'{model_abbreviations[model_name]} ({dataset}, {season})', score, fc='red' if i == 0 else colours[season], ec='black')
    
    ax.axis(xmin=0, xmax=1)
    ax.set_xlabel('ROC-AUC score', fontsize=13)

    ax.tick_params(labelsize=13)

    ax.set_title(f'Model Evaluation by Dataset ({season.capitalize()})', fontsize=14)

for i, ((dataset, model_name), score) in enumerate(pd.concat([evaluation_df['summer'].loc['ROC-AUC score'], evaluation_df['winter'].loc['ROC-AUC score']], axis=1).mean(axis=1).sort_values(ascending=False).iteritems()):
    ax3.barh(f'{model_abbreviations[model_name]} ({dataset})', score, fc='red' if i == 0 else 'darkgreen', ec='black')
    
    ax3.axis(xmin=0, xmax=1)
    ax3.set_xlabel('ROC-AUC score', fontsize=13)

    ax3.tick_params(labelsize=13)

    ax3.set_title(f'Model Evaluation by Dataset (Average)', fontsize=14)

fig.tight_layout()
fig.savefig('model_evaluation_by_dataset.png')


# In[ ]:


threshold = 0.5

# generate predicted probabilities for summer

y_pred_proba_summer = pd.DataFrame(results['appliance_lagged']['GradientBoostingClassifier']['summer'][1], columns=['y_pred_proba', 'household'])

# convert predicted probabilities to predictions based on threshold

y_pred_summer = (y_pred_proba_summer['y_pred_proba'] > threshold).astype(int).rename('y_pred')

# get ground truth values

y_true_summer = pd.concat([df['occupied'] for df in lagged_df_splits[1]], axis=0).reset_index(drop=True).rename('y_true')

diagnostics_summer = pd.concat([y_true_summer, y_pred_proba_summer, y_pred_summer, (y_true_summer == y_pred_summer).rename('accurate')], axis=1)

# repeat for winter

y_pred_proba_winter = pd.DataFrame(results['appliance_lagged']['GradientBoostingClassifier']['winter'][1], columns=['y_pred_proba', 'household'])
y_pred_winter = (y_pred_proba_winter.iloc[:, 0] > threshold).astype(int).rename('y_pred')
y_true_winter = lagged_df_winter['occupied'].reset_index(drop=True).rename('y_true')

diagnostics_winter = pd.concat([y_true_winter, y_pred_proba_winter, y_pred_winter, (y_true_winter == y_pred_winter).rename('accurate')], axis=1)


# In[ ]:


fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for ax, diagnostics, season in zip(axes, [diagnostics_summer, diagnostics_winter], seasons):
    ax = diagnostics.groupby(diagnostics['household'].astype(int))['accurate'].mean().plot.bar(fc=colours[season], ec='black',ax=ax)
    score = roc_auc_score(diagnostics['y_true'], diagnostics['y_pred_proba'])
    ax.text(-0.3, 0.95, f'ROC-AUC score: {score:.3f}', fontsize=12, va='center', ha='left', bbox={'boxstyle': 'round', 'fc': 'wheat', 'alpha': 0.5})
    ax.tick_params(rotation=0, labelsize=13)
    ax.set_title(f'Performance of GBT by Household ({season.capitalize()}) on appliance_lagged data')
    ax.set_xlabel('Household', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.axis(ymin=0.0, ymax=1.0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
fig.savefig('model_performance.png')

