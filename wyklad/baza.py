# %% importy

import pandas as pd
import matplotlib.pyplot as plt

# %%

earning_df = pd.read_csv('table1.csv')
earning_df
# %% Średnia roczna

earning_year_df = earning_df.groupby('Year').mean()
earning_year_df
# %% Płaca tygodniowa męźczyzn [$]
earn_women_df = earning_year_df['Median weekly earnings (in current dollars) - Men']
earn_women_df
# %% Płaca tygodniowa kobiet [$]
earn_men_df = earning_year_df['Median weekly earnings (in current dollars) - Women']
earn_men_df
# %% Średnia płaca tygodniowa dla obu płci [$]
earn_df = earning_year_df['Median weekly earnings (in current dollars) - Total']
earn_df
# %% Zarobki poszczególnych płci na tle średniej z nich obu

earn_women_df.plot()
earn_df.plot()
earn_men_df.plot()
plt.legend()

# %%

gap_men = earn_men_df.mean() - earn_df.mean()
gap_men
# %%
gap_women = earn_women_df.mean() - earn_df.mean()
gap_women
# %%
gap_men = earn_df - earn_men_df 
gap_women = earn_df - earn_women_df

gap_men.plot(label='Gap men')
gap_women.plot(label='Gap women')
plt.legend()
# %%

# %%

# %%
