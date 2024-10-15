import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import rv_continuous
import string

'''
Chpt 7: Cross Validation
'''


def get_train_times(t1, testTimes):
    '''
    t1: the start and end time of each observation (i.e. triple barrier times), this includes all the labels (test and training)
    testTimes: the test observation times
    '''
    trn = t1.copy(deep=True)
    for i, j in testTimes:
        df0 = trn[(trn.index >= i) & (
                    trn.index <= j)].index  # the train label indices which START inside a testTime interval [i,j]
        df1 = trn[(trn >= i) & (trn <= j)].index  # the train label indices which END inside a testTime interval [i,j]
        df2 = trn[(trn.index <= i) & (trn >= j)].index  # the train labels which ENVELOP a testTime interval [i,j]
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


class PurgedKFold(KFold):
    def __init__(self, n_splits=3, t1=None, pct_embargo=0., shuffle=False, random_state=42):
        if not isinstance(t1, type(pd.Series())):
            raise ValueError('Label through dates must be a pd.Series')
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if not self.t1:
            return super().split(X, y)
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]  # represent the label indices THESE ARE TIME INDICES
            test_indices = indices[i:j]
            # the searchsorted function will return the integer indices, not the time indices
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:  # check if the test set isn't at the end of the data
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices
            '''
            yield is used in generator functions to 'save' the state of a function for future use, the k-fold cross validator will essentially 'save' this state of the test and train sets
            and do whatever with these sets in its validation (done separately), where these yields can be accessed using the built in 'next' function which is acted on the function.
            Everytime next is called on this function, we carry on from where we left off, so in this case we will continue in the loop above and then produce and return a new test and train
            set without having to 'start the function again'.
            '''


class PurgedKFoldSimplified(KFold):
    def __init__(self, time_index, n_splits=10, purging_period=60000):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.purging_period = purging_period
        self.time_index = time_index

    def split(self, X, y=None, groups=None):
        times = pd.DataFrame(columns=['startTime', 'endTime'])
        times['startTime'] = self.time_index
        times['endTime'] = self.time_index + self.purging_period
        for train_index, test_index in super().split(X):
            tr = times.iloc[train_index]
            te = times.iloc[test_index]

            tr_purged_pt1 = tr[tr['endTime'] < te['startTime'].iloc[0]]
            tr_purged_pt2 = tr[tr['startTime'] > te['endTime'].iloc[-1]]
            tr_ind_purged = pd.concat([tr_purged_pt1, tr_purged_pt2]).index
            yield tr_ind_purged, test_index


def cv_score(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cvGen=None,
             pctEmbargo=None):  # this is a fix to some bugs in the sklearn KFold class
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    if cvGen is None:
        cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged class created
    score = []
    for train, test in cvGen.split(
            X=X):  # the split function will return a new train and test set each time it cycles through the function (remember it saves its progress in the loop each time it loops)
        fit = clf.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, sample_weight=sample_weight.iloc[test].values, labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


'''
Chpt 8: Feature Importance
'''


def feat_imp_mdi(fit, featNames):
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')  # ensures each tree in the dict above comes out as a row
    df0.columns = featNames
    df0 = df0.replace(0,
                      np.nan)  # we ensure that its highlighted that a 0 importance just means we didn't select it at all
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() / (df0.shape[0] ** 0.5)},
                    axis=1)  # we obtain the standard error on the mean here
    imp /= imp['mean'].sum()  # ensures means add to 1
    return imp


def feat_imp_mda(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged class created
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)  # score for k fold, score for feat imp
    for i, (train, test) in enumerate(cvGen.split(
            X=X)):  # the split function will return a new train and test set each time it cycles through the function (remember it saves its progress in the loop each time it loops)
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train].values
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test].values
        fit = clf.fit(X0, y=y0, sample_weight=w0)
        # get the score for without any features missing (scrambled) for this fold
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            np.random.shuffle(X1_[j].values)  # permutation (scrambling) of a single column
            # get the scores for a particular k fold i and particular column removed j
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1, labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1)
    imp = (-scr1).add(scr0, axis=0)  # essentially subtracting scr1 from scr0 (adding each row to all j column removals)
    # we wish to find the relative difference between the scr0,scr1 difference and the max_score,scr1 difference
    if scoring == 'neg_log_loss':
        imp = imp / (0. - scr1)  # max score for log loss is 0
    else:
        imp = imp / (1. - scr1)  # max score for accuracy is 1
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() / (imp.shape[0] ** 0.5)},
                    axis=1)  # gives mean importance and std error on each feature
    return imp, scr0.mean()  # return feature importance and cross validation score


def aux_feat_imp_sfi(featNames, clf, trnsX, cont, scoring, cvGen):
    imp = pd.DataFrame(columns=['mean', 'std'])
    for featName in featNames:
        df0 = cv_score(clf, X=trnsX[[featName]], y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen)
        imp.loc[featName, 'mean'] = df0.mean()
        imp.loc[featName, 'std'] = df0.std() / (df0.shape[0] ** 0.5)
    return imp


# PCA analysis
def get_e_vec(dot, varThres):
    # dot is the covariance matrix
    eVal, eVec = np.linalg.eigh(dot)  # library for computing eigenvalues and eigenvectors of symmetric matrix
    idx = eVal.argsort()[::-1]  # sorts eigenvalues and eigenvectors into descending order
    eVal, eVec = eVal[idx], eVec[:, idx]
    # we want only positive eigenvalues (those components that actually contribute to variance)
    eVal = pd.Series(eVal, index=['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index,
                        columns=eVal.index)  # vector now written with principle component names in columns
    # reduce dimensions according to varThres
    cumVar = eVal.cumsum() / eVal.sum()  # cumulative % of variance
    dim = cumVar.values.searchsorted(varThres)  # gets the index up until the cumsum of variance % meets the threshold
    eVal, eVec = eVal.iloc[:dim + 1], eVec.iloc[:, :dim + 1]
    return eVal, eVec


def ortho_feats(dfX, varThres=0.95):  # dfX is our usual dataframe of features
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)  # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)  # cov matrix of original features
    eVal, eVec = get_e_vec(dot, varThres)
    dfP = dfZ.dot(eVec)
    return dfP  # gives the new (reduced dimensions) features


# create synthetic data for the purpose of evasluating the feature importance algos
def get_test_data(n_features=40, n_informative=10, n_redundant=10, n_samples=1000):
    # generate random dataset for a classification problem
    from sklearn.datasets import make_classification
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                                      n_redundant=n_redundant, random_state=0, shuffle=False)
    df0 = pd.date_range(periods=n_samples, freq=pd.tseries.offsets.BDay(),
                        end=pd.datetime.today())  # business day frequency, finishing today
    trnsX, cont = pd.DataFrame(trnsX, index=df0), pd.Series(cont, index=df0).to_frame('bin')
    df0 = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
    df0 += ['N_' + str(i) for i in range(n_features - len(df0))]
    trnsX.columns = df0
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)  # labels won't overlap
    return trnsX, cont


def feat_importance(trnsX, cont, n_estimators=1000, cv=10, max_samples=1., numThreads=24, pctEmbargo=0,
                    scoring='accuracy', method='MDI', fitter='BaggedTree', minWLeaf=0.,
                    **kwargs):  # min leaf is where below this weight, the node becomes a leaf (so 0 means must have only 1 sample left to be a leaf)
    # from mpEngine import mpPandasObj
    n_jobs = (-1 if numThreads > 1 else 1)  # -1 means use all processors
    clf = None
    if fitter == 'BaggedTree':
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import BaggingClassifier
        clf = DecisionTreeClassifier(criterion='entropy', max_features=int(1), class_weight='balanced',
                                     min_weight_fraction_leaf=minWLeaf)
        clf = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, max_features=1., max_samples=max_samples,
                                oob_score=True, n_jobs=n_jobs)
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=int(1), max_samples=max_samples,
                                     class_weight='balanced', min_weight_fraction_leaf=minWLeaf, n_jobs=n_jobs)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    imp = None
    oob = fit.oob_score_
    oos = None
    if method == 'MDI':
        imp = feat_imp_mdi(fit, featNames=trnsX.columns)
        oos = cv_score(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=pctEmbargo,
                       scoring=scoring).mean()
    elif method == 'MDA':
        imp, oos = feat_imp_mda(clf, X=trnsX, y=cont['bin'], cv=cv, sample_weight=cont['w'], t1=cont['t1'],
                                pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == 'SFI':
        cvGen = PurgedKFold(n_splits=cv, t1=cont['t1'], pctEmbargo=pctEmbargo)
        oos = cv_score(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'], scoring=scoring, cvGen=cvGen).mean()
        clf.n_jobs = 1
        # bring back below when this is implemented in later chapters
        # imp=mpPandasObj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen)
        imp = aux_feat_imp_sfi(trnsX.columns, clf=clf, trnsX=trnsX, cont=cont, scoring=scoring, cvGen=cvGen)
    return imp, oob, oos


def test_func(n_features=40, n_informative=10, n_redundant=10, n_estimators=1000, n_samples=10000, cv=10):
    trnsX, cont = get_test_data(n_features, n_informative, n_redundant, n_samples)
    dict0 = {'minWLeaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI', 'MDA', 'SFI'], 'max_samples': [1.]}
    jobs, out = (dict(zip(dict0, i)) for i in product(
        *dict0.values())), []  # here we look at all the combinations of arguments (product(*dict0.values())), then zip each of these with each argument label in the dictionary, then convert each entry in the list to a dictionary
    kargs = {'pathOut': './testFunc/', 'n_estimators': n_estimators, 'tag': 'testFunc', 'cv': cv}
    featImps = []

    for job in jobs:
        job['simNum'] = job['method'] + '_' + job['scoring'] + '_' + '%.2f' % job['minWLeaf'] + '_' + str(
            job['max_samples'])
        print(job['simNum'])
        kargs.update(job)
        imp, oob, oos = feat_importance(trnsX=trnsX, cont=cont, **kargs)
        featImps.append(imp)
        df0 = imp[['mean']] / imp['mean'].abs().sum()  # 'relative' importance
        df0['type'] = [i[0] for i in
                       df0.index]  # first letter of feature gives whether it is important, redundant, noise
        df0 = df0.groupby('type')[
            'mean'].sum().to_dict()  # total importance of each type of feature - we should now have one value for each of these three types
        df0.update({'oob': oob, 'oos': oos})  # adds the score for the particular test done
        df0.update(job)  # add to dataframe the additional details about the job
        out.append(df0)  # each row for each method

    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples'])
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
    return out, featImps


def plot_feat_importance(imp, oob, oos, method, tag=0, simNum=0, **kargs):
    plt.figure(figsize=(10, imp.shape[0] / 5.))
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    if method == 'MDI':
        plt.xlim([0, imp.sum(axis=1).max()])
        plt.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    ax.get_yaxis().set_visible(False)
    for i, j in zip(ax.patches, imp.index):
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha='center', va='center', color='black')
    plt.title('tag=' + str(tag) + ' | simNum=' + str(simNum) + ' | oob=' + str(round(oob, 4)) + ' | oos=' + str(
        round(oos, 4)))
    plt.show()
    return


def parallelized_feat_imp(datasets, n_estimators=1000, cv=10, max_samples=1., numThreads=24, pctEmbargo=0,
                          scoring='accuracy', method='MDI', fitter='BaggedTree', minWLeaf=0.):
    mean_imp = pd.DataFrame(index=list(range(len(datasets))), columns=datasets[0][0].columns)
    stdsqd_imp = pd.DataFrame(index=list(range(len(datasets))), columns=datasets[0][0].columns)
    for i in range(len(datasets)):
        imp, oob, oos = feat_importance(datasets[i][0], datasets[i][1], n_estimators=n_estimators, cv=cv,
                                        max_samples=max_samples, numThreads=numThreads, pctEmbargo=pctEmbargo,
                                        scoring=scoring, method=method, minWLeaf=minWLeaf, fitter=fitter)
        mean_imp.iloc[i] = imp['mean']
        stdsqd_imp.iloc[i] = imp['std'] ** 2
    df = pd.DataFrame(index=mean_imp.columns, columns=['mean', 'std'])
    df['mean'] = mean_imp.mean()
    df['std'] = (stdsqd_imp.sum() ** 0.5) / len(datasets)
    return df


'''
Chapter 9: Hyper Parameter Tuning
'''


class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][
                           0] + '__sample_weight'] = sample_weight  # so if we try to use the fit() function with a sample_weight parameter, it will just assign the sample weight to the final model's sample weight param name in fit_params and will then use it in the pipeline's fit function as usual. (see below for explanation)
        return super(MyPipeline, self).fit(X, y, **fit_params)


def clf_hyper_fit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0, None, 1.], rndSearchIter=0, n_jobs=-1,
                  pctEmbargo=0, **fit_params):
    '''
    feat: features (X)
    lbl: classification labels (y)
    t1: usual barrier times
    pipe_clf: the base fitter we are using
    param_grid: dict of lists of parameters to test / dict of distributions of each parameter to be sampled from - can either just be a list (which will be sampled uniformly) or a scipy distribution with a rvs method to allow for sampling
    cv: no. of cross validation splits
    bagging: list giving values for [n_estimators (int), max_samples (float), max_features (float)]
    rndSearchIter: number of random parameter combinations we will test, 0 if not doing random search
    n_jobs: no. of processors used
    fit_params: other parameters for the .fit() method, such as sample_weight; should be written as a dictionary in the format 'pipeline_step__fit_param':fit_param - so 'model__sample_weight' for this example
    '''
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'
    else:
        scoring = 'neg_log_loss'
    # hyperparameter search on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    if rndSearchIter == 0:
        # ordinary grid search no randomness
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, cv=inner_cv,
                                n_jobs=n_jobs, n_iter=rndSearchIter)
    # ***
    gs = gs.fit(feat, lbl,
                **fit_params).best_estimator_  # gives the model with the best hyperparameters selected (remember usually we input the hyperparameters when we initialize a fitter, so this is the same principle) - note: this gs will still be a pipeline
    # fit final model with selected hyperparameters
    if bagging[
        1] > 0:  # of course if max_samples is 0, we are not taking any samples in each bag, so we are not doing any bagging
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps),
                               # here we ensure we use the updated pipeline class so that the sample_weight param can be automatically passed through the base model's fit function
                               n_estimators=int(bagging[0]), max_samples=float(bagging[1]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][
                                                            0] + '__sample_weight'])  # here we need to get the sample_weight from fit_params
        gs = Pipeline(['bag',
                       gs])  # create the final pipeline which is just our bagging classifier (again we see here final step is an estimator)
    return gs


class LogUniformGen(rv_continuous):  # inherited from scipy
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(x / self.b)


def log_uniform(a=1, b=np.exp(1)):
    return LogUniformGen(a=a, b=b, name='logUniform')


ENCODING_CHARS = list(string.printable)


def quantile_encode_series(series: pd.Series, q: int = 10):
    """
    Encodes the series such that it is labelled with each quantile.
    This is currently only supported for q <= 100 (as we encode using characters available in the string module)

    Args:
    encoded_series: series of values, e.g. time series of returns
    q: number of quantiles to be used for encoding

    Returns:
    pd.Series of encoded characters
    """
    return pd.cut(series, bins=q, labels=ENCODING_CHARS[:q]).astype(str)


# same thing is also calculated in pyinfo.block_entropy.block_entropy
# we can also examine local entropy which is just the entropy of eac point before taking the sum across all i
# (remember we have a probability of all L length sequences in the k length series)
def shannon_entropy(encoded_series: pd.Series, sequence_length: int):
    """
    Calculates the probability mass function for an encoded series, for sequences of length sequence_length.
    Then calculates the single entropy value for the encoded series.
    This should be used in conjunction with pandas rolling where we seek to find a series of entropy values across
    a rolling window.

    Args:
    encoded_series: series of encoded values, there should be a fixed number of unique encodings, e.g. 2 (binary) or
                    m (m-quantile). Encodings should be chars/strings.
    sequence_length: integer value L for the length of the sequence examined for our pmf. There could be a total of
                        M^L possible sequences (pick L from M distinct possibilities)

    Returns:
    estimated Shannon entropy
    """
    pmf = {}
    # must turn encodings into string
    encoded_series = "".join(encoded_series)
    for i in range(sequence_length, len(encoded_series)):
        sequence = encoded_series[i-sequence_length:i]
        if sequence in pmf:
            pmf[sequence] += 1
        else:
            pmf[sequence] = 1
    num_sequences = len(encoded_series) - sequence_length
    pmf = np.array([pmf[seq]/num_sequences for seq in pmf.keys()])
    return -sum(pmf * np.log2(pmf))


def encode_and_entropy(series: pd.Series, sequence_length: int, q: int):
    return shannon_entropy(quantile_encode_series(series, q), sequence_length)
