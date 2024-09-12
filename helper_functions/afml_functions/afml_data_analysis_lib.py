import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Chpt 2: Financial Data Structures
'''


def get_dollar_sample_indices(dollar_interval, dollar_traded):
    cumsum = 0
    indices = []
    for i in range(len(dollar_traded)):
        cumsum += dollar_traded[i]
        if cumsum >= dollar_interval:
            cumsum = 0
            indices.append(i)
    return indices


def get_daily_vol_close(close, span0=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0], index=close.index[close.shape[0] - df0.shape[
        0]:]).to_frame()  # all the above does is creates 2 columns with adjacent dates in each column
    df0['close'] = close.loc[df0.index]  # returns
    df0['vol'] = df0['close'].ewm(span=span0).std()  # compute std of ewm of price
    return df0


def get_daily_vol_ret(close, span0=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0], index=close.index[close.shape[0] - df0.shape[
        0]:])  # all the above does is creates 2 columns with adjacent dates in each column
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # returns
    df1 = df0.ewm(span=span0).mean()
    df2 = df0.ewm(span=span0).std()
    df = pd.DataFrame(columns=['return', 'ret vol'])
    df['return'] = df0
    df['ret vol'] = df2
    return df


def sym_cusum(raw_data,
              h):  # raw data here is the returns if we are using a boundary which is of the unit of returns (std of returns is unit of returns) - remember we are checking if returns are accumulating in some way
    filtered_events_indices = []
    s_pos = 0
    s_neg = 0
    diff = raw_data.diff()
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        if s_neg < -h:
            s_neg = 0
            filtered_events_indices.append(i)
        elif s_pos > h:
            s_pos = 0
            filtered_events_indices.append(i)
    return pd.DatetimeIndex(filtered_events_indices)


'''
Chpt 3: Labelling
'''


def get_vertical_barrier(close, tEvents, numDays):
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])


def apply_pt_sl_on_t1(close, events, ptSl, molecule):  # (PtSl means profit taking and stop loss)
    '''
    close: close price data - horizontal barriers will be calculated directly from close return vol
    tEvents: vertical barrier TIMES
    ptSl: list of 1 or 0 indicating if upper or lower barriers should be activated
    '''
    events = events.loc[molecule]  # to do with multiprocessing
    out = events[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events['trgt']
    else:
        pt = np.infty * events['trgt']
    if ptSl[1] > 0:
        sl = -ptSl[1] * events['trgt']
    else:
        sl = - np.infty * events['trgt']
    for loc, t1 in events['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[
              loc:t1]  # takes all the close prices between the time in events['t1'] (the index) and the corresponding barrier - remember the indexed times here are equivalent to the cusum filtered times, and the end barriers are approx a day later
        df0 = (df0 / close[loc] - 1) * events.at[
            loc, 'side']  # checks all the returns at once - comparing to the starting price at the beginning of this triple barrier scan
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[
            loc]].index.min()  # not sl and pt are constant across a barrier search ('box') - taken as the sl/pt at the time of box creation
    return out


def get_events(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # If side is None, it means we are trying to learn the side (position) we should take, if we already know the side and are getting the events for the labelling of a secondary model, we will have an argument for side

    trgt = trgt.loc[tEvents]  # ensures we only care about the trgt values for which our cusum filter has identified
    trgt = trgt[trgt > minRet]  # not particularly important, just some lower limit of return required

    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    if side is None:
        side = pd.Series(1, index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side}, axis=1).dropna(subset=['trgt'])
    df0 = apply_pt_sl_on_t1(close, events, ptSl, events.index)
    #  df0 = mpPandsObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close, events=events, ptSl=ptSl) # this function will be implemented in final chapter
    events['t1'] = df0.dropna(how='all').min(
        axis=1)  # finds the first barrier touch out of either it being a vertical, sl, or pt
    if side is None:
        events = events.drop('side', axis=1)
    return events


def get_bins(events, close):
    '''
    events.index: each triple barrier events start time
    events['t1']: events end time (the barrier hit)
    events['trgt']: horizontal barriers
    events['side'] (optional): pre-found side learnt from (separate) primary model
    '''
    events = events.dropna(subset=['t1'])
    px = events.index.union(events[
                                't1'].values).drop_duplicates()  # taking all the index times and event times into one set and dropping any duplicates (as we will only need prices at these times)
    px = close.reindex(px,
                       method='bfill')  # just take the close prices at the relevant times - as defined above - then just deal with the odd nan by backfilling

    out = pd.DataFrame(
        index=events.index)  # dataframe will have all the barrier hits from the events dataframe (as returned by the getEvents function)
    out['ret'] = px.loc[events['t1'].values].values / px.loc[
        events.index] - 1  # takes return from the initial indexed time to the barrier hit
    out['hit'] = np.where((out['ret'] > events['trgt']) | (out['ret'] < -1 * events['trgt']), 1, 0)

    if 'side' in events:
        out['ret'] *= events[
            'side']  # we are not double multiplying here - remember these returns have been calculated fresh from the close in this function
    out['bin'] = np.sign(out['ret'])  # remember this is the return specifically for barrier hits

    if 'side' in events:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    else:
        out['bin'] = np.where(out['hit'] == 1, out['bin'], 0)

    return out


'''
Chpt 4: Bagging Classifiers and Uniqueness
'''


def mp_num_co_events(closeIdx, t1, molecule):
    t1 = t1.fillna(closeIdx[-1])
    t1 = t1.loc[t1 >= molecule[0]]
    t1 = t1.loc[t1 <= molecule[1]]
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[
                                                    1] + 1])  # just ensures we use the time indices that occur AFTER the closest time indices to t1 start and finish (t_i,0 to the very last time entry in t1)
    for tIn, tOut in t1.iteritems():
        count.loc[
        tIn:tOut] += 1  # any times from the close data indices between the triple barrier times get a +1 (so we form a count for each close index time as to how many times they appear in a triple barrier interval)
    return count


def mp_sample_tw(t1, numCoEvents, molecule):
    wght = pd.Series(0, index=t1.index)
    wght = wght.loc[molecule[0]:molecule[1]]
    for tIn, tOut in t1.loc[wght.index].iteritems():
        # wght.index gives the full range of times the function will provide
        # here we go through each triple barrier interval and find the mean 1/count for each interval (where as we are checking each specific triple barrier interval, the indicator is of course always 1 for that interval)
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()
    return wght


def get_ind_matrix(barIx, t1):
    '''
    barIx: The index of the raw data bars
    t1: The CUSUM triple barrier times (contains 2 columns - the start time and barrier hit time)
    '''
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[
                                                          0]))  # we construct a matrix with rows indexed by the raw data, and columns equal to the number of triple barriers
    # Now we want to set the element to 1 if the time on the row index is contained in the triple barrier interval t0:t1
    for i, (t0, t1) in enumerate(
            t1.iteritems()):  # here we go through each triple barrier interval, remember the columns are indexed by numerate indices, hence the i here to check which column we're 'filling in'
        indM.loc[t0:t1,
        i] = 1  # here we are looking at column i (which corresponds to the particular triple barrier interval) and assigning 1 to any of the raw data indices between the particular triple barrier intervals 2 times, [t0,t1]
    return indM


def get_avg_uniqueness(indM):
    '''
    indM: the indicator matrix returned by the above function
    '''
    c = indM.sum(
        axis=1)  # calculates number of concurrent labels at t (see  notes) so for each row we sum across that row, so count the 1 entries in the matrix (so returns a table with rows equal to the raw data number of rows)
    u = indM.div(c,
                 axis=0)  # calculates the uniqueness for each triple barrier interval, giving yet another matrix with the same dimensions as indM
    avgU = u[
        u > 0].mean()  # we only calculate the mean across t when for t intervals overlapping with the particular triple barrier interval (as we want average uniqueness across just that interval)
    return avgU


def seq_bootstrap(indM, sLength=None):  # sLength allows us to have bootstrap samples of any size if desired
    if sLength is None:
        sLength = indM.shape[1]  # default is original sample size (no. of triple barrier labels)
    phi = []  # same phi as in notes
    while len(phi) < sLength:  # outer loop is taking the samples until we reach our desired size
        avgU = pd.Series()  # this new series is indexed by the label indices; same structure as the average uniqueness returned by the function above
        for i in indM:  # inner loop is going through all the columns of indM, so we are looking at each observation (triple barrier interval) and calculating the average uniqueness across t
            # we look at a smaller matrix as we want to only consider the average of the samples taken so far
            indM_ = indM[phi + [
                i]]  # reduce indM to only the columns which include the target vatriables sampled and the current target variable for which the average uniqueness is being calculated
            avgU.loc[i] = get_avg_uniqueness(indM_).iloc[
                -1]  # we take the last index as we see above that the variable i under consideration has been added on as the last column (and will therefore show up as the last index)
        prob = avgU / avgU.sum()  # gives the updated probabilities of selection based on the average uniqueness of each observation i with respect to the current sample so far
        phi += [np.random.choice(indM.columns,
                                 p=prob)]  # adds to the list the next sample taken at random according to the probabilities
    return phi


def mp_sample_w(t1, numCoEvents, close, molecule):
    '''
    t1: The triple barrier times
    numCoEvents: The concurrency table indexed by raw data times (t, not i)
    close: The actual raw data for prices (using whatever bars the data is supplied in)
    molecule: The list of triple barrier (i) indices that we will use in this particular parallel processing iteration
    '''
    ret = np.log(close).diff()  # log returns to ensure additivity
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[
                                             tIn:tOut]).sum()  # calculates the weight given in notes for each i
    return wght.abs()  # returns weights pre normalization


def get_time_decay(tW, clfLastW=1):
    '''
    tW: The table of uniqueness foreach barrier
    clfLastW: The user chosen c constant
    '''
    clfW = tW.sort_index().cumsum()  # sorts the index for the labels and takes the cumulative sum of the uniqueness (see notes)
    # Find a and b as calculated in notes according to BCs
    if clfLastW >= 0:
        b = (1 - clfLastW) / clfW.iloc[-1]
    else:
        b = 1 / ((clfLastW + 1) * clfW.iloc[-1])
    a = 1 - b * clfW.iloc[-1]
    clfW = a + b * clfW  # actually calculates our d function for all the values of cumulative uniqueness
    clfW[clfW < 0] = 0  # applies our max function
    print(a, b)
    return clfW


'''
Chpt 5: Fractionally Differentiated Features
'''


def get_weights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] * (d - k + 1) / k
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff(series, d, thres=0.01):
    # 1 - Compute weights for longest series, ie weights up until T
    w = get_weights(d, series.shape[
        0])  # note that the 1, indicating the 'first' binomial weight; the one also attached to the most recent data point, is the final element in the array.
    # 2 - Weight loss threshold calculations
    w_ = np.cumsum(abs(w))
    lamda = w_ / w_[-1]
    skip = lamda[lamda > thres].shape[
        0]  # the higher the threshold, the fewer points are greater than the tthreshold, therefore we 'skip' fewer points, so ou skip index (used below) starts from a lower value, as we skip fewer initial points
    # 3 - apply weights to find each fractionally differentiated value, excluding the skipped values
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()  # going through supplied dataframe column by column
        X_tilda = pd.Series(index=seriesF.iloc[skip:seriesF.shape[0]].index)
        for iloc in range(skip, seriesF.shape[0]):  # ranging from after skipped values to most recent values
            loc = seriesF.index[iloc]  # time of current int index
            if not np.isfinite(series.loc[loc, name]):
                continue
            # Now we look at the current time for which we want to calculate X_tilda
            X_tilda[loc] = np.dot(w[-(iloc + 1):, :].T, seriesF.loc[:loc])[
                0, 0]  # dotting the latter weights in the transposed array (so the last weight here will still be 1) with all the past times up until the current time
        df[name] = X_tilda.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def get_weights_ffd(d, thres):
    w = [1.]
    k = 1
    while w[-1] >= thres:
        w_ = -w[-1] * (d - k + 1) / k
        w.append(w_)
        k += 1
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(series, d, thres=1e-5):
    # 1 - Compute weights for longest series, ie weights up until T
    print('getting weights with d=', d)
    w = get_weights_ffd(d, thres)
    print('weights obtained')
    width = len(w) - 1  # this is how many memory terms we will have for X tilda at all t
    # 2 - apply weights to find each fractionally differentiated value, excluding the skipped (equal to the width) values
    df = {}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()  # going through supplied dataframe column by column
        X_tilda = pd.Series(index=seriesF.iloc[width:seriesF.shape[0]].index)
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[
                iloc1]  # times at beginning of window and end of window
            if not np.isfinite(series.loc[loc1, name]):
                continue
            X_tilda[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = X_tilda.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def plot_min_ffd(series, thres=1e-2):
    from statsmodels.tsa.stattools import adfuller
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, 1, 11):
        df1 = np.log(series[['close']]).resample(
            '1D').last()  # resamples series to daily instead of whatever it was before and uses log prices
        df1 = np.log(series[['close']])
        df1 = series[['close']]
        df2 = frac_diff_ffd(df1, d, thres)  # our new fractionally differentiated series using fixed window method
        corr = np.corrcoef(df1.loc[df2.index, 'close'], df2['close'])[
            0, 1]  # correlation between differenced series and original series - we want this to be high as this indicates the series are similar with regards to information, therefore we have not lost a lot of memory for high correlation coefficients
        df2 = adfuller(df2['close'], maxlag=1, regression='c', autolag=None)
        print(df2)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [
            corr]  # just takes returns from adfuller function and puts into nice list
    out[['adfStat', 'corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    plt.xlabel('d')
    return
