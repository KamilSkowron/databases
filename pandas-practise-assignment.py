#%%
from numpy import isin
import pandas as pd
# %%
countries_df = pd.read_csv("countries.csv")
countries_df
# %% Q1: How many countries does the dataframe contain?
countries_df.info()
countries_df.describe()
countries_df.shape

# %%
num_countries = 210
print('There are {} countries in the dataset'.format(num_countries))
# %% Q2: Retrieve a list of continents from the dataframe?
countries_df['continent'].unique()

# %% Q3: What is the total population of all the countries listed in this dataset?
countries_df['population'].sum()

total_population = countries_df['population'].sum()
# %% Q: (Optional) What is the overall life expectancy across in the world?
countries_df['life_expectancy'].mean()

# %% Q4: Create a dataframe containing 10 countries with the highest population.
most_populous_df = countries_df.sort_values('population',ascending=False).head(10)
most_populous_df

# %% Q5: Add a new column in countries_df to record the overall GDP per country (product of population & per capita GDP).
countries_df['gdp'] = countries_df['gdp_per_capita'] * countries_df['population']
countries_df

# %% Q: (Optional) Create a dataframe containing 10 countries with the lowest GDP per capita, among the counties with population greater than 100 million.
counties_high_pop_df = countries_df[countries_df.population > 100000000]
lowest_gdp = counties_high_pop_df.sort_values('gdp_per_capita',ascending=True).head(10)
lowest_gdp

# %% Q6: Create a data frame that counts the number countries in each continent?
country_counts_df = countries_df.groupby('continent')['location'].count()
country_counts_df

# %% Q7: Create a data frame showing the total population of each continent.
continent_populations_df = countries_df.groupby('continent')['population'].sum()
continent_populations_df

# %% Q8: Count the number of countries for which the total_tests data is missing.
covid_data_df = pd.read_csv('covid-countries-data.csv')
total_tests_missing = covid_data_df[covid_data_df['total_tests'].isna()].count().location
total_tests_missing

print("The data for total tests is missing for {} countries.".format(int(total_tests_missing)))

# %% Q9: Merge countries_df with covid_data_df on the location column.
combined_df = countries_df.merge(covid_data_df, on='location')
combined_df

# %% Q10: Add columns tests_per_million, cases_per_million and deaths_per_million into combined_df.
combined_df['tests_per_million'] = combined_df['total_tests'] * 1e6 / combined_df['population']
combined_df['cases_per_million'] = combined_df['total_cases'] * 1e6 / combined_df['population']
combined_df['deaths_per_million'] = combined_df['total_deaths'] * 1e6 / combined_df['population']
combined_df

# %% Q11: Create a dataframe with 10 countires that have highest number of tests per million people.
highest_tests_df = combined_df.sort_values('tests_per_million', ascending=False).head(10)
highest_tests_df

# %% Q12: Create a dataframe with 10 countires that have highest number of positive cases per million people.
combined_df['positive_cases_per_million'] = combined_df['total_cases'] * 1e6 / combined_df['total_tests']
highest_number_of_cases_per_million = combined_df.sort_values('positive_cases_per_million',ascending=False).head(10)
highest_number_of_cases_per_million

# %% Q13: Create a dataframe with 10 countires that have highest number of deaths cases per million people?
highest_deaths_df = combined_df.sort_values('deaths_per_million',ascending=False).head(10)
highest_deaths_df

# %% (Optional) Q: Count number of countries that feature in both the lists of "highest number of tests per million" and "highest number of cases per million".
both_df = highest_deaths_df['location'].isin(highest_number_of_cases_per_million['location'])
both_df.sum()

# %% (Optional) Q: Count number of countries that feature in both the lists "20 countries with lowest GDP per capita" and "20 countries with the lowest number of hospital beds per thousand population". Only consider countries with a population higher than 10 million while creating the list.
lowest_GDP_per_df = combined_df[combined_df.population > 1e8].sort_values('gdp_per_capita', ascending=True).head(20)
lowest_bed_df = combined_df[combined_df.population > 1e8].sort_values('hospital_beds_per_thousand', ascending=True).head(20)

answer = lowest_GDP_per_df['location'].isin(lowest_bed_df['location']).sum()
answer
# %%
