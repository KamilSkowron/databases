
#%%
from os import stat
from random import random
from unittest import result
import matplotlib
from matplotlib.pyplot import title


#%% create variable which takes file "climate.txt", commas are seperators, skips header (how many lines have it)
import numpy as np
climate_data = np.genfromtxt('climate.txt', delimiter=',', skip_header=1)
print(climate_data)

#%% create weights of every category
weights = np.array([0.3, 0.2, 0.5])

#%% multiply them  A[0] * B[0] + A[1] * B[1] + A[2] * A[2]
yields = climate_data @ weights

#%% connect two "tables" and changes yields orientations reshape(rows,columns)
climate_results = np.concatenate((climate_data, yields.reshape(10000, 1)), axis=1)

np.savetxt('climate_results.txt', 
           climate_results, 
           fmt='%.0f', 
           delimiter=',',
           header='temperature,rainfall,humidity,yeild_apples', 
           comments='')


#%% arrays operations
Z = np.empty((0,3), int)
X = np.empty((0,3), int)

A = np.array([1,2,3])
B = np.array([4,5,6])
C = np.array([7,8,9])

Z = np.append(Z, np.array([A]), axis=0)            #[[1,2,3],
Z = np.append(Z, np.array([B]), axis=0)            # [4,5,6],
Z = np.append(Z, np.array([C]), axis=0)            # [7,8,8]]
print(Z)

X = np.append(X, np.array([A]), axis=0)            #[1,2,3,4,5,6,7,8,9]
X = np.append(X, np.array([B]), axis=1)
X = np.append(X, np.array([C]), axis=1)
print(X)

#%% concatenate function

a = np.array([[1,2,3]])                           #[[1 2 3]
b = np.array([[4,5,6]])                            #[4 5 6]]

c = np.concatenate((a,b), axis=0)

print(c)

a = np.array([[1,2,3]])                           #[[1 2 3 4 5 6]]
b = np.array([[4,5,6]])                            

c = np.concatenate((a,b), axis=1)                   

print(c)


#%% respahing array

G = np.array([[1,2,3,4,5,6],
             [7,8,9,0,1,2]])
print(G)

print(np.shape(G))          # (2,6)

H = np.reshape(G,(6,2))
print(H)

K = np.arange(200).reshape((4,50))
print(K)

#%% others numpy functions


L = np.arange(2,200,3)
print(L)

M = np.linspace(0,10,11)            #(start,stop,number_of_samples)
print(M)

P = np.repeat(20,5)                 #(what,how many)
print(P)

O = np.random.randint(20)
print(O)

####################################################################
                # FILES STAFF

#%% open a file
file1 = open("loans1.txt",'r')

###################################################################

#%% the simplest way to read file (all at ones)

fite1_contents = file1.read()
print(fite1_contents)

for lines in file1:
    print(lines)

###################################################################

#%% open and close file

with open("loans1.txt") as file1:
    file1_contents = file1.read()
    print(file1_contents)

###################################################################

#%% the best way to handle files

with open("loans3.txt") as file3:
    file3_lines = file3.readlines()
    print(file3_lines)

###################################################################

#%% seperates headers from file
def parse_headers(header_line):                     
    return header_line.strip().split(',')

headers = parse_headers(file3_lines[0])
print(headers)

###################################################################

#%% converts all values on float, if value doesn't exist, replace it with "0.0"
def parse_values(data_line):                        
    values = []
    for item in data_line.strip().split(','):
        if item == '':
            values.append(0.0)
        else:
            values.append(float(item))
    return values

for line in file3_lines[1:]:
   print(parse_values(line))

###################################################################

#%% creates a dictionary {header:value}

def create_item_dict(values, headers):              
    result = {}
    for value, header in zip(values,headers):
        result[header] = value
    return result

values_1 = parse_values(file3_lines[1])
print(create_item_dict(values_1,headers))

###################################################################

#%% reading csv functions

def read_csv(path):
    result = []
    with open(path, 'r') as file:
        lines = file.readlines()
        headers = parse_headers(lines[0])
        for data_line in lines[1:]:
            values = parse_values(data_line)
            item_dict = create_item_dict(values, headers)
            result.append(item_dict)
    return result

text_csv = read_csv("loans3.txt")
print(text_csv)

###################################################################

#%% reading csv functions

import math

def loan_emi (amount, duration, rate, down_payment=0):
    """Calculates the equal montly installment (EMI) for a loan.
    Arguments:
        amount - TOtal amount to be spent (loan + down payment)
        duration - Duration of the loan (in months)
        rate - Rate of interest (monthly)
        down_payment(optional) - Optional intial payment
    """
    loan_amount = amount - down_payment
    try:
        emi = loan_amount * rate * ((1+rate)**duration) / (((1+rate)**duration)-1)
    except ZeroDivisionError:
        emi = loan_amount / duration
    emi = math.ceil(emi)
    return emi

loans3 = read_csv('loans3.txt')
print(loans3)


###################################################################

#%% function adding new column
def compute_emis(file):
    loans = read_csv(file)
    for loan in loans:
        loan['emi'] = loan_emi(loan['amount'],
                            loan['duration'],
                            loan['rate']/12,
                            loan['down_payment'])
    return loans

###################################################################

# %% example use of function

compute_emis("loans2.txt")

# %% writing data to a file
def write_to_file(old, new):
    bal_file = compute_emis(old)
    
    with open(new, 'w') as f_new:
        f_new.write(str(",".join(bal_file[0].keys()) + "\n"))
        for loan in bal_file:
            f_new.write('{},{},{},{},{}\n'.format(
                loan['amount'],
                loan['duration'],
                loan['rate'],
                loan['down_payment'],
                loan['emi'],
            ))
    

# %% test

write_to_file("loans1.txt","analysis_loans1.txt")

# %% PANDAS
#########################################################   PANDAS  ##########################################################

import pandas as pd

covid_df = pd.read_csv('italy-covid-daywise.csv')
covid_df

# %%
dark_df = pd.read_csv('06_14_2017-ad-dowgin-gdata-3e30f2644a2e9f1b81f7f5a810e5f6ce.pcap_ISCX.csv')
dark_df

# %% show all data
covid_df
# %% show column, count of non-null values, dtype
covid_df.info()
# %% show statistics (counts, mean, std, min , max itp)
covid_df.describe()
# %%
covid_df.shape
# %%
covid_df.columns

# %% example of pandas file
# covid_data_dict = {
#     'date':       ['2020-08-30', '2020-08-31', '2020-09-01', '2020-09-02', '2020-09-03'],
#     'new_cases':  [1444, 1365, 996, 975, 1326],
#     'new_deaths': [1, 4, 6, 8, 6],
#     'new_tests': [53541, 42583, 54395, None, None]
# }

# %% Show value of record 246 and column 'new_cases'
covid_df['new_cases'][246]

# %% create new table with selected data (but it is still the same table)
cases_df = covid_df[['date','new_cases']]
cases_df

# %% copy table
covid_df_copy = covid_df.copy()
covid_df_copy

# %% show all record 243
covid_df.loc[243]

# %% show first 5 records
covid_df.head(5)

# %% show last 5 records
covid_df.tail(5)

# %% show value of 200 record of 'new_tests' column
covid_df.at[200,'new_tests']

# %% show number of first record with non NaN value
covid_df.new_tests.first_valid_index()

# %% show 10 random records
covid_df.sample(10)

#########################################################################################

# %% 
total_cases = covid_df.new_cases.sum()
total_deaths = covid_df.new_deaths.sum()
total_new_tests = covid_df.new_tests.sum()

print('The number of reported cases is {} and the number of reported deaths is {}'.format(int(total_cases), int(total_deaths)))

# %%
death_rate = total_deaths/total_cases
print("The overall reported death rate in Italy is {:.2f} %.".format(death_rate*100))

# %%
initial_tests = 935310
total_tests = initial_tests + total_new_tests
total_tests

# %%
positive_rate = total_cases / total_tests
print('{:.2f}% of tests in Italy led to a positive diagnosis.'.format(positive_rate*100))

# %%
high_new_cases = covid_df.new_cases > 1000
high_cases_df = covid_df[high_new_cases]
high_cases_df

# %% display options
from IPython.display import display
with pd.option_context('display.max_rows',100):
    display(covid_df[covid_df.new_cases > 1000])

# %%
high_ratio_df = covid_df[covid_df.new_cases / covid_df.new_tests > positive_rate]
high_ratio_df

##########################################################################################

# %% adds new columns
covid_df['positive_rate'] = covid_df.new_cases / covid_df.new_tests
covid_df

# %% deletes columns
covid_df.drop(columns=['positive_rate'], inplace=True)

# %% sorting records        ascending=True inverse sorting
#dataframe.sort_values('by_what', ascending=True/False)
covid_df.sort_values('new_cases', ascending=False).head(10)
covid_df.sort_values('new_deaths', ascending=False).head(10)

# %% we found a suspect value in data so let's look around it
covid_df.loc[169:175]
"""depends of the case, if we found error value in record we can:
    - replace it with 0
    - replace it with the averange of the entire column
    - replace it with the averange of the values on the previous & next date
    - discard the row entirely
    """

# %% replace a value at certain record
covid_df.at[172, 'new_cases'] = (covid_df.at[171, 'new_cases'] + covid_df.at[173, 'new_cases']) / 2

# %%
covid_df.loc[169:175]
##########################################################################################

# %% working with time data

covid_df.date
# %% convert dtype of column 'date' from 'object' into 'datetime'
covid_df['date'] = pd.to_datetime(covid_df.date)
covid_df['date']
# %% adds new columns to the data
covid_df['year'] = pd.DatetimeIndex(covid_df.date).year
covid_df['month'] = pd.DatetimeIndex(covid_df.date).month
covid_df['day'] = pd.DatetimeIndex(covid_df.date).day
covid_df['weekday'] = pd.DatetimeIndex(covid_df.date).weekday

# %% creates table with only may record
covid_df_may = covid_df[covid_df.month == 5]
covid_df_may

# %% demanding specific information from new data
covid_df_may_metrics = covid_df_may[['new_cases','new_deaths','new_tests']]
covid_df_may_metrics

# %%sum colums
covid_df_may_totals = covid_df_may_metrics.sum()
covid_df_may_totals

# %% short version
covid_df[covid_df.month == 5][['new_cases','new_deaths' ,'new_tests']].sum()

# %%
covid_df[covid_df.weekday == 6].new_cases.mean()
# %%
stats = []
for days in range (0,7,1):
    stats.append(covid_df[covid_df.weekday == days].new_cases.mean())
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

stats_data = list(zip(days, stats))
sorted_data = sorted(stats_data, key = lambda x: int(x[1]), reverse= True)
sorted_data

##########################################################################################
# %% grouping and aggregation
covid_month_df = covid_df.groupby('month')[['new_cases','new_deaths','new_tests']].sum()
covid_month_df

# %% mean for weekdays
covid_weekday_mean_df = covid_df.groupby('weekday')[['new_cases','new_deaths','new_tests']].mean()
covid_weekday_mean_df

# %% sum record by record
covid_df['total_cases'] = covid_df.new_cases.cumsum()
covid_df['total_deaths'] = covid_df.new_deaths.cumsum()
covid_df['total_tests'] = covid_df.new_tests.cumsum() + initial_tests

# %%
covid_df[140:150]

# %% MERGING DATA
##########################################################################################

from urllib.request import urlretrieve

urlretrieve('https://gist.githubusercontent.com/aakashns/8684589ef4f266116cdce023377fc9c8/raw/99ce3826b2a9d1e6d0bde7e9e559fc8b6e9ac88b/locations.csv', 
            'locations.csv')

locations_df = pd.read_csv('locations.csv')

# %% checking if "Italy is in database"
locations_df[locations_df.location == "Italy"]

# %% adds another column "location" with value "Italy"
covid_df['location'] = "Italy"
covid_df

# %% merging two database
merged_df = covid_df.merge(locations_df, on="location")
merged_df

# %%addinng a few new columns
merged_df['cases_per_million'] = merged_df.total_cases * 1e6 / merged_df.population
merged_df['deaths_per_milion'] = merged_df.total_deaths * 1e6 / merged_df.population
merged_df['tests_per_milion'] = merged_df.total_tests * 1e6 / merged_df.population
merged_df

# %% SAVE new database
result_df = merged_df[['date',
                       'new_cases',
                       'total_cases',
                       'new_deaths',
                       'total_deaths',
                       'new_tests',
                       'total_tests',
                       'cases_per_million',
                       'deaths_per_milion',
                       'tests_per_milion']]

result_df.to_csv('results.csv', index=None)

##########################################################################################
# %% VISUALISATION
result_df.new_cases.plot();

# %% Erasing index from datebase and replace it with 'date'
result_df.set_index('date', inplace=True)
result_df
# %%
result_df.loc['2020-09-01']

# %%
result_df.new_cases
# %%
result_df.new_cases.plot();
result_df.new_deaths.plot();
# %%
result_df.total_cases.plot()
result_df.total_deaths.plot()
# %%
death_rate = result_df.total_deaths / result_df.total_cases
death_rate.plot(title='Death Rate')
# %%
positive_rates = result_df.total_cases / result_df.total_tests
positive_rates.plot(title='Positive Rate')
# %%
covid_month_df.new_cases.plot(kind='bar')
# %%
covid_month_df.new_tests.plot(kind='bar')

# %% importing libraries
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline

# %% #################### Line Chart #################### (data below)
yield_apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931]
years = [2010, 2011, 2012, 2013, 2014, 2015]
# %%
plt.plot(yield_apples);

# %% Customizing the X-axis
plt.plot(years, yield_apples);
plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)');

# %% Plotting Multiple Lines
years = range(2000, 2012)
apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0.939]
oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.901, 0.898, 0.9, 0.896, ]

# %% Markers, colors etc
plt.plot(years, apples, marker='s', c='b', ls='-', lw=2, ms=8, mew=2, mec='navy')
plt.plot(years, oranges, marker='o', c='r', ls='--', lw=3, ms=10, alpha=.5)


plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.legend(['Apples','Oranges'])

"""
    color or c: Set the color of the line (supported colors)
    linestyle or ls: Choose between a solid or dashed line
    linewidth or lw: Set the width of a line
    markersize or ms: Set the size of markers
    markeredgecolor or mec: Set the edge color for markers
    markeredgewidth or mew: Set the edge width for markers
    markerfacecolor or mfc: Set the fill color for markers
    alpha: Opacity of the plot
"""

# %% shortcut of specified marker (+ title)

# fmt = '[marker][line][color]'

plt.plot(years, apples, 's-b')
plt.plot(years, oranges, 'o--r')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.legend(['Apples','Oranges'])
plt.title("Crop Yields in Kanto")
# %% no lines
plt.plot(years, oranges, 'or')
plt.title("Yield od Oranges (tons per hectare")

# %% Changing the size of figure
plt.figure(figsize=(8,4))

plt.plot(years, oranges, 'or')
plt.title("Yield od Oranges (tons per hectare")

# %% ############## SEABORN ################
sns.set_style("whitegrid")
sns.set_style("darkgrid")
"https://seaborn.pydata.org/generated/seaborn.set_style.html ."

#global apply
# %% test graf
plt.plot(years, apples, 's-b')
plt.plot(years, oranges, 'o--r')

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.legend(['Apples','Oranges'])
plt.title("Crop Yields in Kanto")
# %% manual edit styles
import matplotlib
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9,5)
matplotlib.rcParams['figure.facecolor'] = '#A4B5C6D7'

# %% ####################### Scatter Plot #####################


# %% load default dataset with comes with Seaborn
flowers_df = sns.load_dataset("iris")
flowers_df

# %%
flowers_df['species'].unique()

# %%
sns.scatterplot(flowers_df.sepal_length, #X - axis
                flowers_df.sepal_width,  #Y - axis
                hue=flowers_df.species,  #group_by species
                s=100)                   #makes dots bigger
# %%plt + sns
plt.figure(figsize=(12, 6))
plt.title('Sepal Dimensions')

sns.scatterplot(x=flowers_df.sepal_length, 
                y=flowers_df.sepal_width, 
                hue=flowers_df.species,
                s=100);
# %%
plt.figure(figsize=(12, 6))
plt.title('Sepal Dimensions')

sns.scatterplot('sepal_length', 
                'sepal_width', 
                hue='species',
                s=100,
                data = flowers_df);

# %% ####################### Histogram #####################
flowers_df = sns.load_dataset("iris")
flowers_df.sepal_width

# %% specifying the boundaries of each bin
plt.title("Distribution of Sepal Width")
plt.hist(flowers_df.sepal_width,
         bins=np.arange(2, 5, 0.25)) #arange(start, end, step) [2, 2.25, 2.50...etc]

# %% bins of unequal sizes
plt.hist(flowers_df.sepal_width,
         bins=[1,3,4,4.5])

# %% separate histograms for each species of flowers
setosa_df = flowers_df[flowers_df.species == 'setosa']
versicolor_df = flowers_df[flowers_df.species == 'versicolor']
virginica_df = flowers_df[flowers_df.species == 'virginica']

plt.hist(setosa_df.sepal_width,
         alpha=0.4,
         bins=np.arange(2,5,0.25))

plt.hist(versicolor_df.sepal_width,
         alpha=0.4,
         bins=np.arange(2,5,0.25))
plt.legend(['Setosa','Versicolor'])

# %% stacked histogram
plt.hist([setosa_df.sepal_width, virginica_df.sepal_width, versicolor_df.sepal_width],
         bins=np.arange(2,5,0.25),
         stacked=True);
# %% ################## Bar chart
years = range(2000, 2006)
apples = [0.35, 0.6, 0.9, 0.8, 0.65, 0.8]
oranges = [0.4, 0.8, 0.9, 0.7, 0.6, 0.8]

# %%
plt.bar(years, oranges);
plt.plot(years,apples, 'o--r')
# %%
plt.bar(years, apples)
plt.bar(years, oranges, bottom=apples);


# %% another dataset from sns config
tips_df = sns.load_dataset("tips");
tips_df
# %%
bill_avg_df = tips_df.groupby('day')[['total_bill']].mean()
# %% pandas df + plt stuff
plt.bar(bill_avg_df.index,bill_avg_df['total_bill'])

# %%
sns.barplot('day','total_bill', data=tips_df);
# %%
sns.barplot('day','total_bill', hue='sex', data=tips_df);

# %%
sns.barplot('total_bill', 'day', hue='sex', data=tips_df)       #('x','y',hue, use data_base)

# %% ################ HEATMAP #####################
flights_df = sns.load_dataset('flights').pivot('month','year','passengers')     #pivot('y','x','inside')
flights_df
# %%
plt.plot(flights_df.passengers);
# %% heatmap
plt.title("No. of Passengers (1000s)")
sns.heatmap(flights_df);
# %% heatmap with data
plt.title("No. of Passengers (1000s)")
sns.heatmap(flights_df, fmt="d", annot=True, cmap="Blues");

# %% # %% ################ Plotting multiple charts in a grid #####################

fig, axes = plt.subplots(2, 3, figsize=(16, 8))         #(2 rows, 3 columns)

# Use the axes for plotting
axes[0,0].plot(years, apples, 's-b')
axes[0,0].plot(years, oranges, 'o--r')
axes[0,0].set_xlabel('Year')
axes[0,0].set_ylabel('Yield (tons per hectare)')
axes[0,0].legend(['Apples', 'Oranges']);
axes[0,0].set_title('Crop Yields in Kanto')


# Pass the axes into seaborn
axes[0,1].set_title('Sepal Length vs. Sepal Width')
sns.scatterplot(x=flowers_df.sepal_length, 
                y=flowers_df.sepal_width, 
                hue=flowers_df.species, 
                s=100, 
                ax=axes[0,1]);

# Use the axes for plotting
axes[0,2].set_title('Distribution of Sepal Width')
axes[0,2].hist([setosa_df.sepal_width, versicolor_df.sepal_width, virginica_df.sepal_width], 
         bins=np.arange(2, 5, 0.25), 
         stacked=True);

axes[0,2].legend(['Setosa', 'Versicolor', 'Virginica']);

# Pass the axes into seaborn
axes[1,0].set_title('Restaurant bills')
sns.barplot(x='day', y='total_bill', hue='sex', data=tips_df, ax=axes[1,0]);

# Pass the axes into seaborn
axes[1,1].set_title('Flight traffic')
sns.heatmap(flights_df, cmap='Blues', ax=axes[1,1]);

# Plot an image using the axes
axes[1,2].set_title('Data Science Meme')
#axes[1,2].imshow(img)
axes[1,2].grid(False)
axes[1,2].set_xticks([])
axes[1,2].set_yticks([])

plt.tight_layout(pad=2);
# %%

# %%
