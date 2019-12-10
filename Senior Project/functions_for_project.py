import tensorflow as tf
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pylab import rcParams


def format_both_df(df1,df2,event_time,given_start = -1,given_end = -1):
	df1 = df1.drop_duplicates()
	df2 = df2.drop_duplicates()
	
	timecolsdf1 = [col for col in df1.columns if 'time' in col]
	df1 = df1.rename(columns={timecolsdf1[0] : 'timestamp'})
	df1['timestamp'] = pd.to_datetime(df1['timestamp'],infer_datetime_format=True).dt.tz_localize(None)
	
	timecolsdf2 = [col for col in df2.columns if 'time' in col]
	df2 = df2.rename(columns={timecolsdf2[0] : 'timestamp'})
	df2['timestamp'] = pd.to_datetime(df2['timestamp'],infer_datetime_format=True).dt.tz_localize(None)
	
	df1_slim = df1[['timestamp','open','high','low','close','volume']]
	df2_slim = df2[['timestamp','open','high','low','close','volume']]
	
	start_time = max(df1_slim['timestamp'].iloc[0],df2_slim['timestamp'].iloc[0])
	end_time = min(df1_slim['timestamp'].iloc[-1],df2_slim['timestamp'].iloc[-1])
	
	df1_slim = df1_slim.loc[(df1_slim['timestamp'] >= start_time) & (df1_slim['timestamp'] <= end_time)]
	df2_slim = df2_slim.loc[(df2_slim['timestamp'] >= start_time) & (df2_slim['timestamp'] <= end_time)]
	df1_slim['after_event'] = df1_slim['timestamp'] > event_time
	df2_slim['after_event'] = df2_slim['timestamp'] > event_time
	
	if given_start != -1:
		df1_slim = df1_slim.loc[(df1_slim['timestamp'] >= given_start)]
		df2_slim = df2_slim.loc[(df2_slim['timestamp'] >= given_start)]
	if given_end != -1:
		df1_slim = df1_slim.loc[(df1_slim['timestamp'] <= given_end)]
		df2_slim = df2_slim.loc[(df2_slim['timestamp'] <= given_end)]
	df1_slim['min_elapsed'] = df1_slim['timestamp'].values.astype(float)/(60*10**9)
	df1_slim['min_elapsed'] = df1_slim['min_elapsed'] - df1_slim['min_elapsed'].iloc[0]
	df1_slim['min_elapsed'] = df1_slim['min_elapsed'].values.astype(int)
	
	
	df2_slim['min_elapsed'] = df2_slim['timestamp'].values.astype(float)/(60*10**9)
	df2_slim['min_elapsed'] = df2_slim['min_elapsed'] - df2_slim['min_elapsed'].iloc[0]
	df2_slim['min_elapsed'] = df2_slim['min_elapsed'].values.astype(int)
	
	df1_slim = df1_slim[df1_slim['min_elapsed'].isin(df2_slim['min_elapsed'])]
	df2_slim = df2_slim[df2_slim['min_elapsed'].isin(df1_slim['min_elapsed'])]
	
	df1_slim['minutes'] = df1_slim['min_elapsed']
	df1_slim = df1_slim.set_index('min_elapsed')
	df2_slim['minutes'] = df2_slim['min_elapsed']
	df2_slim = df2_slim.set_index('min_elapsed')
	
	df1_slim['swing'] = df1_slim['high'] - df1_slim['low']
	df2_slim['swing'] = df2_slim['high'] - df2_slim['low']
	df1_slim['swingdiff'] = df1_slim['swing'] - df2_slim['swing']
	
	df1_slim['swing_rolling_daily'] = df1_slim['swingdiff'].rolling(window=1440).mean()

	df2_vol_adj_coef = np.sum(df1_slim.volume[df1_slim.after_event == 0][-1*(1440*30):]) / np.sum(df2_slim.volume[df2_slim.after_event == 0][-1*(1440*30):])

	df2_slim['vol_adjusted'] = df2_slim['volume']*df2_vol_adj_coef
	df1_slim['vol_over_adjvol_rolling_1day'] = df1_slim['volume'].rolling(window=1440).mean() / df2_slim['vol_adjusted'].rolling(window=1440).mean()


	return([df1_slim,df2_slim])

def graph_both(df1,df2,event_time,title1,title2,graph_using_volume = False):
	if graph_using_volume == False:
		mod_df1 = smf.ols(formula='swingdiff ~ minutes + after_event + after_event * minutes', data=df1)
		res_df1 = mod_df1.fit()
		df1['regression_results'] = res_df1.params[0] + df1['after_event'] * res_df1.params[1] + df1['minutes'] * res_df1.params[2] + df1['after_event'] * df1['minutes'] * res_df1.params[3]


		mod_df2 = smf.ols(formula='swingdiff ~ minutes + after_event + after_event * minutes', data=df2)
		res_df2 = mod_df2.fit()
		df2['regression_results'] = res_df2.params[0] + df2['after_event'] * res_df2.params[1] + df2['minutes'] * res_df2.params[2] + df2['after_event'] * df2['minutes'] * res_df2.params[3]
	else:
		mod_df1 = smf.ols(formula='vol_over_adjvol_rolling_1day ~ minutes + after_event + after_event * minutes', data=df1)
		res_df1 = mod_df1.fit()
		df1['regression_results'] = res_df1.params[0] + df1['after_event'] * res_df1.params[1] + df1['minutes'] * res_df1.params[2] + df1['after_event'] * df1['minutes'] * res_df1.params[3]


		mod_df2 = smf.ols(formula='vol_over_adjvol_rolling_1day ~ minutes + after_event + after_event * minutes', data=df2)
		res_df2 = mod_df2.fit()
		df2['regression_results'] = res_df2.params[0] + df2['after_event'] * res_df2.params[1] + df2['minutes'] * res_df2.params[2] + df2['after_event'] * df2['minutes'] * res_df2.params[3]


	rcParams['figure.figsize'] = 11, 6

	plt.subplot(1, 2, 1)
	ax = plt.gca()
	if graph_using_volume == False:
		df1.plot(kind='line',x='timestamp',y='swing_rolling_daily', ax=ax)
	else:
		df1.plot(kind='line',x='timestamp',y='vol_over_adjvol_rolling_1day', ax=ax)
	df1.plot(kind='line',x='timestamp',y='regression_results', color='blue', ax=ax)
	ax.axvline(x= event_time, c = 'red')
	ax.title.set_text(title1)

	plt.subplot(1, 2, 2)
	ax = plt.gca()
	if graph_using_volume == False:
		df2.plot(kind='line',x='timestamp',y='swing_rolling_daily', ax=ax)
	else:
		df2.plot(kind='line',x='timestamp',y='vol_over_adjvol_rolling_1day', ax=ax)
	df2.plot(kind='line',x='timestamp',y='regression_results', color='blue', ax=ax)
	ax.axvline(x= event_time, c = 'red')
	ax.title.set_text(title2)

	plt.savefig('both_diffs.png')
	plt.show()
	

def variance_testing(df1,exchanges_compared):
	rcParams['figure.figsize'] = 13, 7

	fig, (ax1, ax2) = plt.subplots(ncols=2) # create two subplots, one in each row


	x = df1['swingdiff'][0:-1]
	y1 = df1['swingdiff'][1:]
	ax1.scatter(x,y1)
	ax1.title.set_text(exchanges_compared + ' 1 minute autocorrelation')

	sm.graphics.tsa.plot_pacf(sm.graphics.tsa.pacf(df1['swingdiff'], nlags=40),ax = ax2)
	plt.show()

	F = np.var(df1.swingdiff[df1.after_event == 1]) / np.var(df1.swingdiff[df1.after_event == 0])
	print(exchanges_compared + ' F-test value: ' + str(F))
	bart = scipy.stats.bartlett(df1.swingdiff[df1.after_event == 1],df1.swingdiff[df1.after_event == 0])
	lev = scipy.stats.levene(df1.swingdiff[df1.after_event == 1],df1.swingdiff[df1.after_event == 0])
	print(exchanges_compared + ' bartlett test for equal variances, p value: ' + str(bart[1]))
	print(exchanges_compared + ' levene test for equal variances, p value: ' + str(lev[1]))