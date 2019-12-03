# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:51:32 2018

@author: Andrew
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
#from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors


def loadModel(filename):
    return joblib.load(filename)

    
def matchRating(data):
    
    tm = time.time()
    for cl0, cl1, cl2, cl3 in [
                               ('teamATExplodePer', 'teamATScore', 'teamBCTScore', 'teamAExplode'), 
                               ('teamACTDifusePer', 'teamACTScore', 'teamBTScore', 'teamADifuse'), 
                               ('teamBTExplodePer', 'teamBTScore', 'teamACTScore', 'teamBExplode'), 
                               ('teamBCTDifusePer', 'teamBCTScore', 'teamATScore', 'teamBDifuse')
                              ]:
        index = (data[cl1]>=0) & (data[cl2]>=0) & (data[cl3]>=0)

        sm = (data.loc[index, cl1] + data.loc[index, cl2]).astype(float)
        confInt = 1 - sm/15.0
        if not cl3:
            data.loc[index, cl0] = np.round(np.mean([np.maximum(data.loc[index, cl1]/sm-confInt, 0), np.minimum(data.loc[index, cl1]/sm+confInt, 1)], axis=0), 2)
        else:
            data.loc[index, cl0] = np.round(np.mean([np.maximum(data.loc[index, cl3]/sm-confInt, 0), np.minimum(data.loc[index, cl3]/sm+confInt, 1)], axis=0), 2)
    
    print time.time()-tm

    data[['teamAKillavg', 'teamADeathavg', 'teamAassavg', 'teamAFirstKavg', 'teamBKillavg', 'teamBDeathavg', 'teamBassavg', 'teamBFirstKavg']] = data[['teamAKill', 'teamADeath', 'teamAass', 'teamAFirstK', 'teamBKill', 'teamBDeath', 'teamBass', 'teamBFirstK']].div(5*data['totalRnds'], axis=0)
    data[['teamAClutchPer', 'teamBClutchPer']] = data[['teamAClutch', 'teamBClutch']].div(data['totalRnds'], axis=0)

    cnt = 0
    for cl0, cl1, cl2 in [
                          ('teamAhsPer', 'teamAhs', 'teamAKill'), 
                          ('teamAFlashPer', 'teamAFlash', 'teamAass'), 
                          ('teamBhsPer', 'teamBhs', 'teamBKill'), 
                          ('teamBFlashPer', 'teamBFlash', 'teamBass')
    ]:
        cnt += 1
        index = data[cl2]>=0

        cond1 = data.loc[index, cl2].values == 0
        ind = data.loc[index][cond1].index
        data.loc[ind, cl0] = 0
        ind = data.loc[index][~cond1].index
        data.loc[ind, cl0] = data.loc[ind, cl1].values/data.loc[ind, cl2].values.astype(float)
        
    data.loc[data[data['teamAFlashPer']<0].index, 'teamAFlashPer'] = -10
    data.loc[data[data['teamBFlashPer']<0].index, 'teamBFlashPer'] = -10
    
#    retColumns = ['teamATExplodePer', 'teamACTDifusePer', 'teamAExplode', 'teamADifuse',
#                 'teamAKillavg', 'teamADeathavg', 'teamAassavg', 'teamAFirstKavg', 'teamAClutchPer',
#                 'teamAhsPer', 'teamAFlashPer', 
#                  'teamBTExplodePer', 'teamBCTDifusePer', 'teamBExplode', 'teamBDifuse',
#                 'teamBKillavg', 'teamBDeathavg', 'teamBassavg', 'teamBFirstKavg', 'teamBClutchPer',
#                 'teamBhsPer', 'teamBFlashPer', 
#                  'ratingType', 'teamAMatchRating', 'teamBMatchRating', 'difScore']
    return data

        

def teamRatingsPastTwoMonthsGameStats(teamsRating, mainDf, teamId='teamId'):
    def hf(localData):
        if localData.shape[0]==0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            lst = []
            lst += [localData[localData['difScore']>-17]['difScore'].sum(), localData[localData['difScore']>0].shape[0]/max(1, float(localData[localData['difScore']>-17].shape[0])), localData[localData['difScore']>-17]['difScore'].mean()]
            lst += [localData[localData['teamAadravg']>=0]['teamAadravg'].mean()]
            lst += [localData[localData['teamAMatchRating']>0]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0)]['teamAMatchRating'].mean()]
            mapPicker = -1
            lst += [localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].sum(), localData[(localData['difScore']>0) & (localData['mapPicker']==mapPicker)].shape[0]/max(1, float(localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)].shape[0])), localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].mean()]
            lst += [localData[(localData['teamAadravg']>=0) & (localData['mapPicker']==mapPicker)]['teamAadravg'].mean()]
            lst += [localData[(localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean()]
            mapPicker = 0
            lst += [localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].sum(), localData[(localData['difScore']>0) & (localData['mapPicker']==mapPicker)].shape[0]/max(1, float(localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)].shape[0])), localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].mean()]
            lst += [localData[(localData['teamAadravg']>=0) & (localData['mapPicker']==mapPicker)]['teamAadravg'].mean()]
            lst += [localData[(localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean()]
            mapPicker = 1
            lst += [localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].sum(), localData[(localData['difScore']>0) & (localData['mapPicker']==mapPicker)].shape[0]/max(1, float(localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)].shape[0])), localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].mean()]
            lst += [localData[(localData['teamAadravg']>=0) & (localData['mapPicker']==mapPicker)]['teamAadravg'].mean()]
            lst += [localData[(localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean()]
            return lst
        
    lst = []
    for _ in range(teamsRating.shape[0]):
        if _%1000==0:
            print _
            
        mask = (teamsRating.loc[_, teamId]==mainDf['teamAId']) & (mainDf['date']>=teamsRating.loc[_, 'date'] - np.timedelta64(60, 'D')) & (mainDf['date']<=teamsRating.loc[_, 'date'])
        lst += [hf(mainDf.loc[mask])]
        
    return pd.DataFrame(lst, columns=['validGameDifscoreSum', 'validGameDifscoreWin', 'validGameDifscoreMean', 'validGameadrMean', 'validGamePerfomMean', 'validGamePerfomWinMean', 'validGamePerfomLooseMean', \
                                     'validGameDifscoreSumEnemy', 'validGameDifscoreWinEnemy', 'validGameDifscoreMeanEnemy',  'validGameadrMeanEnemy', 'validGamePerfomMeanEnemy', 'validGamePerfomWinMeanEnemy', 'validGamePerfomLooseMeanEnemy', \
                                     'validGameDifscoreSumNet', 'validGameDifscoreWinNet', 'validGameDifscoreMeanNet', 'validGameadrMeanNet', 'validGamePerfomMeanNet', 'validGamePerfomWinMeanNet', 'validGamePerfomLooseMeanNet', \
                                     'validGameDifscoreSumMine', 'validGameDifscoreWinMine', 'validGameDifscoreMeanMine', 'validGameadrMeanMine', 'validGamePerfomMeanMine', 'validGamePerfomWinMeanMine', 'validGamePerfomLooseMeanMine'])


def hltWeekRatingvPositionPropogation(mainDf, ttt):
    mainDf['teamAPosition'] = 31
    sorted_dates = ttt['date'].sort_values()
    date_ind = 0
    
    for ind in range(mainDf.shape[0]):
        if ind%5000==0:
            print ind
    #     print mainDf.iloc[ind]['date']
        doc = mainDf.iloc[ind]
        if date_ind!=sorted_dates.shape[0]-1:
            if sorted_dates.iloc[date_ind]<=doc['date']:
                date_ind += 1
                
        if sorted_dates.iloc[date_ind]>doc['date']:
            continue
        
    #     print doc['date'], sorted_dates.iloc[date_ind]
        lcl = ttt[(ttt['date']==sorted_dates.iloc[date_ind]) & (ttt['position']<31) & (ttt['teamId']==doc['teamAId'])]
        if lcl.shape[0]:
    #         print lcl['position'].values[0]
            mainDf.loc[ind, 'teamAPosition'] = lcl['position'].values[0]
    #         print mainDf.loc[ind, 'teamAPosition']
    #         break
    return mainDf

def avgIncome(dats, ids, data, days):
    def hf(vl, mval, days):
        if vl.shape[0]==0:
            return 0
        else:
            tmp1 = np.maximum((vl['date']-mval).dt.days/float(days), 0.01)
            return (tmp1*vl['win']).sum()
        
    lastDats = dats - np.timedelta64(days, 'D')
    ttt = (ids.values.reshape((ids.shape[0], 1))==data['teamId'].values) & (data['date'].values>=lastDats.values.reshape((ids.shape[0], 1))) & (data['date'].values<=dats.values.reshape((ids.shape[0], 1)))
    ret = np.array(map(lambda mask, y: hf(data.loc[mask], y, days), ttt, lastDats.values))
    return ret

def preproc_data(ldf, pred_clmns, y_col, test_size=0.2, random_state=523, cond=None):
    #TODO preproc_data
    if cond is None:
        x_train, x_test, y_train, y_test = train_test_split(ldf[pred_clmns], ldf[y_col], test_size=test_size, random_state=random_state)
    else:
        
#     if 'alf' in y_col and 'istol' in y_col:
#         cond = ldf[y_col]!=0
        x_train, x_test, y_train, y_test = train_test_split(ldf[cond][pred_clmns], ldf[cond][y_col], test_size=test_size, random_state=random_state)
        
        
    print 'TRAIN SHAPE:', x_train.shape, 'TEST SHAPE:', x_test.shape
    if 'alf' in y_col and 'istol' in y_col:
        y_train.replace({-1: 0}, inplace=True)
        y_test.replace({-1: 0}, inplace=True)

    return x_train, x_test, y_train, y_test

def fit_model(params, x_train, x_test, y_train, y_test, weight_train=None, weight_test=None, modelname='lgb'):
    #TODO fit_model
    if modelname=='lgb':
        # define lgb train and validation Datasets
        d_train = lgb.Dataset(x_train, y_train, weight=weight_train)
        d_valid = lgb.Dataset(x_test, y_test, weight=weight_test)

        # train model
        print '\nSTART FITTING:'
        model = lgb.train(params=params, 
                        train_set=d_train, 
                        num_boost_round=5000, 
                        valid_sets=[d_valid],
                        verbose_eval=100, 
                        early_stopping_rounds=25)
        print 'FITTING HAS BEEN ENDED\n'
        
        print 'VALIDATION:'
        if params['objective']=='binary':
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

            print 'TRAIN ROC_AUC:', round(roc_auc_score(y_train, train_pred), 6), 'ACCURACY:', round(accuracy_score(y_train, np.round(train_pred)), 6)
            print 'TEST ROC_AUC:', round(roc_auc_score(y_test, test_pred), 6), 'ACCURACY:', round(accuracy_score(y_test, np.round(test_pred)), 6)
            
        else:#params['objective']=='regression'
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

            print 'TRAIN MAE:', round(mean_absolute_error(y_train, train_pred), 6), 'MSE:', round(mean_squared_error(y_train, train_pred), 6)
            print 'TEST MAE:', round(mean_absolute_error(y_test, test_pred), 6), 'MSE:', round(mean_squared_error(y_test, test_pred), 6)
            
        return model
    
    elif modelname=='xgb':
        model = XGBRegressor(**params)
        
        print '\nSTART FITTING:'
        model.fit(x_train, y_train)
        print 'FITTING HAS BEEN ENDED\n'
        
        print 'VALIDATION:'
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        print 'TRAIN MAE:', round(mean_absolute_error(y_train, train_pred), 6), 'MSE:', round(mean_squared_error(y_train, train_pred), 6)
        print 'TEST MAE:', round(mean_absolute_error(y_test, test_pred), 6), 'MSE:', round(mean_squared_error(y_test, test_pred), 6)
            
        return model
        

def teamRatingsPastTwoMonthsGameStatsHF(localData):
    #TODO teamRatingsPastTwoMonthsGameStatsHF
    if localData.shape[0]==0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        lst = []
        lst += [localData[localData['difScore']>-17]['difScore'].sum(), localData[localData['difScore']>0].shape[0]/max(1, float(localData[localData['difScore']>-17].shape[0])), localData[localData['difScore']>-17]['difScore'].mean()]
        lst += [localData[localData['teamAadravg']>=0]['teamAadravg'].mean()]
        lst += [localData[localData['teamAMatchRating']>0]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0)]['teamAMatchRating'].mean()]
        mapPicker = -1
        lst += [localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].sum(), localData[(localData['difScore']>0) & (localData['mapPicker']==mapPicker)].shape[0]/max(1, float(localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)].shape[0])), localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].mean()]
        lst += [localData[(localData['teamAadravg']>=0) & (localData['mapPicker']==mapPicker)]['teamAadravg'].mean()]
        lst += [localData[(localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean()]
        mapPicker = 0
        lst += [localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].sum(), localData[(localData['difScore']>0) & (localData['mapPicker']==mapPicker)].shape[0]/max(1, float(localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)].shape[0])), localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].mean()]
        lst += [localData[(localData['teamAadravg']>=0) & (localData['mapPicker']==mapPicker)]['teamAadravg'].mean()]
        lst += [localData[(localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean()]
        mapPicker = 1
        lst += [localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].sum(), localData[(localData['difScore']>0) & (localData['mapPicker']==mapPicker)].shape[0]/max(1, float(localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)].shape[0])), localData[(localData['difScore']>-17) & (localData['mapPicker']==mapPicker)]['difScore'].mean()]
        lst += [localData[(localData['teamAadravg']>=0) & (localData['mapPicker']==mapPicker)]['teamAadravg'].mean()]
        lst += [localData[(localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean(), localData[(localData['difScore']>-17) & (localData['difScore']<0) & (localData['teamAMatchRating']>0) & (localData['mapPicker']==mapPicker)]['teamAMatchRating'].mean()]
        return lst
        

def teamRatingsPastTwoMonthsGameStats(teamsRating, mainDf, teamId='teamId'):
    #TODO teamRatingsPastTwoMonthsGameStats
#     lastDats = teamsRating['date'] - np.timedelta64(60, 'D')
#     ttt = (teamsRating['teamId'].values.reshape((teamsRating.shape[0], 1))==mainDf['teamAId'].values) & (mainDf['date'].values>=lastDats.values.reshape((teamsRating.shape[0], 1))) & (mainDf['date'].values<=teamsRating['date'].values.reshape((teamsRating.shape[0], 1)))
#     print ttt.shape
    lst = []
    for _ in range(teamsRating.shape[0]):
        if _%1000==0:
            print _
            
#         print mainDf.loc[0, 'date'], teamsRating.loc[0, 'date'], teamsRating.loc[_, 'date'] - np.timedelta64(60, 'D'), ((mainDf['date']>=teamsRating.loc[_, 'date'] - np.timedelta64(60, 'D')) & (mainDf['date']<=teamsRating.loc[_, 'date'])).sum()
#         assert False
        mask = (teamsRating.loc[_, teamId]==mainDf['teamAId']) & (mainDf['date']>=teamsRating.loc[_, 'date'] - np.timedelta64(60, 'D')) & (mainDf['date']<=teamsRating.loc[_, 'date'])
        lst += [teamRatingsPastTwoMonthsGameStatsHF(mainDf.loc[mask])]
#         break
        
    return pd.DataFrame(lst, columns=['validGameDifscoreSum', 'validGameDifscoreWin', 'validGameDifscoreMean', 'validGameadrMean', 'validGamePerfomMean', 'validGamePerfomWinMean', 'validGamePerfomLooseMean', \
                                     'validGameDifscoreSumEnemy', 'validGameDifscoreWinEnemy', 'validGameDifscoreMeanEnemy',  'validGameadrMeanEnemy', 'validGamePerfomMeanEnemy', 'validGamePerfomWinMeanEnemy', 'validGamePerfomLooseMeanEnemy', \
                                     'validGameDifscoreSumNet', 'validGameDifscoreWinNet', 'validGameDifscoreMeanNet', 'validGameadrMeanNet', 'validGamePerfomMeanNet', 'validGamePerfomWinMeanNet', 'validGamePerfomLooseMeanNet', \
                                     'validGameDifscoreSumMine', 'validGameDifscoreWinMine', 'validGameDifscoreMeanMine', 'validGameadrMeanMine', 'validGamePerfomMeanMine', 'validGamePerfomWinMeanMine', 'validGamePerfomLooseMeanMine'])


    

#def avgIncome(x, prDistDf, mnthQnt=12):
##    x = [teamClmn, date]
#    #CALCULATE ONE TEAM AVERAGE INCOME PER MONTHQNT
#    tmp = prDistDf[(x[0]==prDistDf['teamId'].values) & (prDistDf['date']>=x[1]-dt.timedelta(days=mnthQnt*30)) & (prDistDf['date']<=x[1])]#[['date', 'win']]#.values
#    if tmp.shape[0]==0:
#        return 0
#    else:
#        td = dt.timedelta(days=mnthQnt*30)
#        tmp1 = (tmp['date']-(x[1]-td)).dt.days/float(mnthQnt*30)
#        return (tmp1*tmp['win']).sum()/tmp1.sum()
    


def preproc(newMainDf, ratingDifCols):
    #TODO preproc
    newMainDf['ratingDif'] = newMainDf[ratingDifCols[0]] - newMainDf[ratingDifCols[1]]
    # newMainDf['ratingDifB'] = newMainDf['teamBRating'] - newMainDf['teamARating']
    
#    newMainDf.loc[newMainDf[newMainDf['teamAmaxPrPoolRatio']>100].index, 'teamAmaxPrPoolRatio'] = 100
#    newMainDf.loc[newMainDf[newMainDf['teamBmaxPrPoolRatio']>100].index, 'teamBmaxPrPoolRatio'] = 100
#    
#    newMainDf.loc[newMainDf[newMainDf['teamAfirPlaceRatio']>50].index, 'teamAfirPlaceRatio'] = 50
#    newMainDf.loc[newMainDf[newMainDf['teamBfirPlaceRatio']>50].index, 'teamBfirPlaceRatio'] = 50
    newMainDf['matchFormat'] = newMainDf['matchFormat'].apply(abs)
    newMainDf['mapPicker'] = newMainDf['mapPicker'].astype(int)
    newMainDf['matchType'] = newMainDf['matchType'].replace({'Online': 0, 'LAN': 1, 'Lan': 1})#value_counts()
    return newMainDf

def createScaledDf(newMainDf, scaler, scaleClmns=['teamAfirPlaceRatio', 'teamBfirPlaceRatio', 'teamAmaxPrPoolRatio', 'teamBmaxPrPoolRatio', 'teamAMatchRatingMine', 'teamBMatchRatingMine'], \
                   unchngeClmns=['matchFormat', 'mapPicker', 'map', 'matchType']):
    #TODO createScaledDf
#    minMaxSc = MinMaxScaler()
#    clmns = [#'teamAMapLeftToWin', 'teamBMapLeftToWin', 
#               'teamAavgIncome', 'teamBavgIncome','teamAfirPlaceRatio', 'teamBfirPlaceRatio', 'teamAmaxPrPoolRatio', 
#               'teamBmaxPrPoolRatio']
    if not scaleClmns:
        scaleClmns = ['firPlace', 'maxPrizeUSD', u'avgIncomeDif', u'firPlaceRatioDif', u'maxPrPoolRatioDif']
    newMainDfScaled = scaler.fit_transform(newMainDf[scaleClmns].values)
    newMainDfScaled = pd.DataFrame(newMainDfScaled, columns=scaleClmns)
    
    for cl in unchngeClmns:
        newMainDfScaled[cl] = newMainDf[cl]
#    newMainDfScaled['map'] = newMainDf['map']
#    newMainDfScaled['mapPicker'] = newMainDf['mapPicker']
#    newMainDfScaled['matchFormat'] = newMainDf['matchFormat']
#    newMainDfScaled['matchType'] = newMainDf['matchType']
#    newMainDfScaled['teamAMatchRatingMine'] = newMainDf['teamAMatchRatingMine']
#    newMainDfScaled['teamBMatchRatingMine'] = newMainDf['teamBMatchRatingMine']
    
    if 'matchFormat' in newMainDfScaled.columns:
        matchFormatDummies = pd.get_dummies(newMainDfScaled['matchFormat'], prefix='matchFormat')
        for clmn in matchFormatDummies.columns:
            newMainDfScaled[clmn] = matchFormatDummies[clmn]
            
    if 'mapPicker' in newMainDfScaled.columns:
        mapPickerDummies = pd.get_dummies(newMainDfScaled['mapPicker'], prefix='mapPicker')
        for clmn in mapPickerDummies.columns:
            newMainDfScaled[clmn] = mapPickerDummies[clmn]
    
    if 'map' in newMainDfScaled.columns:
        mapDummies = pd.get_dummies(newMainDfScaled['map'])
        for clmn in mapDummies.columns:
            newMainDfScaled[clmn] = mapDummies[clmn]
    
    return newMainDfScaled, scaler

def scale(newMainDf, clmns, scaler):
    #TODO scale
    newMainDfScaled = scaler.fit_transform(newMainDf[clmns].values)
    newMainDfScaled = pd.DataFrame(newMainDfScaled, columns=clmns)
    return newMainDfScaled, scaler

def corralationMultiplier(newMainDfScaled, clmns, difclmns, tmpInd=np.array([])):
    #TODO corralationMultiplier
    #SCALE newMainDfScaled[clmns] ACCORDING TO PIERSO'N CORRELATION MATRIX
    if tmpInd.shape[0]:
        corr = newMainDfScaled.loc[tmpInd, clmns].corr()[['teamAMatchRatingMine', 'teamBMatchRatingMine']].apply(abs)
    else:
        corr = newMainDfScaled[clmns].corr()[['teamAMatchRatingMine', 'teamBMatchRatingMine']].apply(abs)
    
#     setclmns = list(set(clmns) - set(['teamAMatchRatingMine', 'teamBMatchRatingMine']))
#     columnA = [i for i in setclmns if 'A' in i]
#     columnB = [i for i in setclmns if 'B' in i]
#     ttlClmns = [1]*(len(columnA)+len(columnB))
#     ttlClmns[::2] = columnA
#     ttlClmns[1::2] = columnB
    corr.drop(['teamAMatchRatingMine', 'teamBMatchRatingMine'], inplace=True)
    
#    for i in range(len(ttlClmns)/2):
##        print ttlClmns[2*i:2*(i+1)]
#        corr.loc[ttlClmns[2*i:2*(i+1)], 'teamAMatchRatingMine'] = corr.loc[ttlClmns[2*i:2*(i+1)], 'teamAMatchRatingMine'].apply(abs).mean()

    corrMean = corr['teamAMatchRatingMine'].apply(abs).mean()*30
    for ind in corr.index:
        newMainDfScaled[ind] = newMainDfScaled[ind]*30*corr.loc[ind, 'teamAMatchRatingMine']
        
    mapDummies = pd.get_dummies(newMainDfScaled['map'])
    for clmn in mapDummies.columns:
        newMainDfScaled[clmn] = mapDummies[clmn]*corrMean
        
    return newMainDfScaled, corr, corrMean


def fitKNN(newMainDfScaled, n_neighbors):
    #TODO fitKNN
    knn = NearestNeighbors(n_neighbors=n_neighbors)#, radius=5)
    knn.fit(newMainDfScaled)
    return knn

#def relativeForceCalculationLocal(scaledData, knnInd, baseInd, n_neighbors, distance=[]):
#    #CALCULATE ONE TEAM RELATIVE FORCE BASED ON N NEAREST NEIGHBOURS
#    cl = 'teamAMatchRatingMine'
#    delInd = np.argwhere(knnInd==baseInd)
#
#    mnVal = scaledData.loc[baseInd, cl]
#    knnInd = np.delete(knnInd, delInd)
#    tmp = scaledData.loc[knnInd, 'teamAMatchRatingMine'].values<=mnVal
#    
#    if type(distance)!=list:#weighted
#        distance = np.delete(distance, delInd)
#        distance[np.isnan(distance)] = 0
#        disMax = distance.max()
#        if not disMax:
#            disMax = 1
#        distance = 0.5 + (disMax - distance)/(2*disMax)
#        return distance[tmp].sum()/distance.sum()
#    
#    if delInd:
#        return np.sum(tmp)/float(n_neighbors-1)
#    return np.sum(tmp)/float(n_neighbors)


def relativeForceCalculationLocal(scaledData, knnInd, baseInd, n_neighbors, cl, distance=[], mode=2):
    #CALCULATE ONE TEAM RELATIVE FORCE BASED ON N NEAREST NEIGHBOURS
#    cl = 'teamAMatchRatingMine'
    tmp = scaledData[cl].values
#     print 'teamAMatchRatingMine', tmp.shape, 'knnInd', knnInd.shape
#    if type(distance)!=list:
##         print 'dist', distance.shape
#        knnInd, distance, found = map(np.array, zip(*map(lambda x, y, z: (np.delete(x, np.argwhere(x==y)), np.delete(z, np.argwhere(x==y)), float(n_neighbors - np.argwhere(x==y).shape[0]) ), knnInd, baseInd, distance)))
#    else:
#        knnInd, found = map(np.array, zip(*map(lambda x, y: (np.delete(x, np.argwhere(x==y)), float(n_neighbors - np.argwhere(x==y).shape[0]) ), knnInd, baseInd)))
        

    cmprVal = scaledData.loc[baseInd, cl].values.reshape((baseInd.shape[0], 1))
    res = tmp[knnInd] <= cmprVal#).sum(axis=1)
    
    if type(distance)!=list:#weighted
        distance[np.isnan(distance)] = 0
        
        disMax = distance.max(axis=1)
        disMax[disMax==0] = 1
        disMax = disMax.reshape((disMax.shape[0], 1))
        
#        distance = 0.5 + (disMax - distance)/(2*disMax)
        distance = (disMax - distance)/disMax
        _ = np.array(map(lambda x, y: x[y].sum()/x.sum(), distance, res))
        if mode==1:
            return _
        
    if mode==2:
        return _, res.sum(axis=1)/float(n_neighbors)
    
    return res.sum(axis=1)/float(n_neighbors)#found

def relativeForceCalculation(newMainDf, newMainDfScaled, knn, clmns, iterateClmns, glblIndex=[], mode=''):
    #TODO relativeForceCalculation
    if type(glblIndex)==list:
        baseIndex = newMainDfScaled.index.values
    else:
#         print 'here'
        baseIndex = newMainDfScaled.loc[glblIndex].index.values

    dst, ind = knn.kneighbors(newMainDfScaled.loc[baseIndex, clmns].values)
    ind = baseIndex[ind]
    
    if type(glblIndex)!=list and not mode:
#        iterateClmns = [('teamARForceWeightedSpec', 'teamAMatchRatingMine'), ('teamARForceSpec', 'teamAMatchRatingMine'), 
#                        ('teamAKillavgRWeightedSpec', 'teamAKillavg'), ('teamAhsPerRWeightedSpec', 'teamAhsPer'),
#                        ('teamADeathavgRWeightedSpec', 'teamADeathavg'), ('teamAadravgRWeightedSpec', 'teamAadravg'), ('teamAKastavgRWeightedSpec', 'teamAKastavg'),
#                        ('teamAKillavgRSpec', 'teamAKillavg'), ('teamAhsPerRSpec', 'teamAhsPer'),
#                        ('teamADeathavgRSpec', 'teamADeathavg'), ('teamAadravgRSpec', 'teamAadravg'), ('teamAKastavgRSpec', 'teamAKastavg')]
#         print 'glblInd'
        dataShape = newMainDfScaled.shape[0]
        beforeInd = [i for i in baseIndex if i<dataShape/2]
        afterInd = [i for i in baseIndex if i>=dataShape/2]
#         print len(beforeInd), len(afterInd)
    else:
#        iterateClmns = [('teamARForceWeighted', 'teamAMatchRatingMine'), ('teamARForce', 'teamAMatchRatingMine'), 
#                        ('teamAKillavgRWeighted', 'teamAKillavg'), ('teamAhsPerRWeighted', 'teamAhsPer'),
#                        ('teamADeathavgRWeighted', 'teamADeathavg'), ('teamAadravgRWeighted', 'teamAadravg'), ('teamAKastavgRWeighted', 'teamAKastavg'),
#                        ('teamAKillavgR', 'teamAKillavg'), ('teamAhsPerR', 'teamAhsPer'),
#                        ('teamADeathavgR', 'teamADeathavg'), ('teamAadravgR', 'teamAadravg'), ('teamAKastavgR', 'teamAKastavg')]
        
        beforeInd = np.arange(newMainDf.shape[0]/2)
        afterInd = np.arange(newMainDf.shape[0]/2, newMainDf.shape[0])
    
#     print beforeInd.shape, afterInd.shape, newMainDf.loc[baseIndex].shape, newMainDf.loc[afterInd].shape
#     assert False
    
#    if len(iterateClmns)==2:
#        newMainDf.loc[baseIndex, iterateClmns[0][0]], newMainDf.loc[baseIndex, iterateClmns[1][0]] = relativeForceCalculationLocal(newMainDfScaled, ind, baseIndex, knn.n_neighbors, iterateClmns[0][1], distance=dst)
#        replaceClmn = iterateClmns[0][0].replace('A', 'B')
#        newMainDf.loc[beforeInd, replaceClmn], newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[afterInd, iterateClmns[0][0]].values, newMainDf.loc[beforeInd, iterateClmns[0][0]].values
#        replaceClmn = iterateClmns[1][0].replace('A', 'B')
#        newMainDf.loc[beforeInd, replaceClmn], newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[afterInd, iterateClmns[1][0]].values, newMainDf.loc[beforeInd, iterateClmns[1][0]].values
#    else:
    for clmn in iterateClmns:
        print clmn
        if 'Weight' in clmn[0]:
            newMainDf.loc[baseIndex, clmn[0]] = relativeForceCalculationLocal(newMainDfScaled, ind, baseIndex, knn.n_neighbors, clmn[1], distance=dst, mode=1)
#            if clmn[1]!='teamAMatchRatingMine':
#                newMainDf.loc[baseIndex[newMainDf.loc[baseIndex, clmn[1]]<0], clmn[0]] = np.nan
#            newMainDf.loc[baseIndex, clmn] = map(lambda knnInd, baseInd, weights: relativeForceCalculationLocal(newMainDfScaled, knnInd, baseInd, knn.n_neighbors, weights), ind, baseIndex, dst)
        else:
            newMainDf.loc[baseIndex, clmn[0]] = relativeForceCalculationLocal(newMainDfScaled, ind, baseIndex, knn.n_neighbors, clmn[1], distance=[], mode=1)
#            if clmn[1]!='teamAMatchRatingMine':
#                newMainDf.loc[baseIndex[newMainDf.loc[baseIndex, clmn[1]]<0], clmn[0]] = np.nan
#            newMainDf.loc[baseIndex, clmn] = map(lambda knnInd, baseInd: relativeForceCalculationLocal(newMainDfScaled, knnInd, baseInd, knn.n_neighbors), ind, baseIndex)

        replaceClmn = clmn[0].replace('A', 'B')
        newMainDf.loc[beforeInd, replaceClmn] = newMainDf.loc[afterInd, clmn[0]].values
        newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[beforeInd, clmn[0]].values
        
#        print newMainDf[newMainDf[clmn]>1].shape
        
#    print 
    return newMainDf


        
def hf1(tmp, mval, iterateClmns, days):
    if tmp.shape[0]==0:
        return [-0.5, 0.0]*len(iterateClmns) + [0, 2*days]
    returnValues = []
    
    n_matches = tmp.shape[0]
    trr = (tmp['date']-mval).dt.days
    last_match_passed_days = days - max(trr)
    trr = (trr/float(days)).values# .apply(lambda lcl: 0.001 if (lcl-mval).days==0 else (lcl-mval).days/float(days))
    trr[trr==0] = 0.001
    
    for col in iterateClmns:
        if col[-1]:#weighted=True
            returnVal = (trr*tmp[col[1]]).sum()/trr.sum()
            
            mean = tmp[col[1]].mean()
            std = ((trr*(tmp[col[1]].values - mean)**2).sum()/trr.sum())**0.5
            
            returnValues += [returnVal, std]
        else:#weighted=False
            returnValues += [tmp[col[1]].mean(), tmp[col[1]].std()]
            
    return returnValues + [n_matches, last_match_passed_days]
        
def relativeForceHistoryCalculationLocal(dats, ids, data, iterateClmns, days):#tmp = [['teamId', 'date']]
    lastDats = dats - np.timedelta64(days, 'D')
    ttt = (ids.values.reshape((ids.shape[0], 1))==data['teamAId'].values) & (data['date'].values>=lastDats.values.reshape((ids.shape[0], 1))) & (data['date'].values<dats.values.reshape((ids.shape[0], 1)))
    ret = np.array(map(lambda mask, y: hf1(data.loc[mask], y, iterateClmns, days), ttt, lastDats.values))
    return ret.T
    
def relativeForceHistoryCalculation(newMainDf, days, iterateClmns, glblIndex=[]):
    #TODO relativeForceHistoryCalculation
    if type(glblIndex)!=list:
        
        dataShape = newMainDf.shape[0]
        beforeInd = [i for i in glblIndex.values if i<dataShape/2]
        afterInd = [i for i in glblIndex.values if i>=dataShape/2]
    else:
#        glblIndex = newMainDf.index
        
        beforeInd = np.arange(newMainDf.shape[0]/2)
        afterInd = np.arange(newMainDf.shape[0]/2, newMainDf.shape[0])

    tm = time.time()
    
    if type(glblIndex)!=list:
        tmp = newMainDf.loc[glblIndex]
    else:
        tmp = newMainDf
    tmp = np.concatenate(map(lambda ind: relativeForceHistoryCalculationLocal(tmp.loc[ind, 'date'], tmp.loc[ind, 'teamAId'], tmp, iterateClmns, days), np.array_split(tmp.index, 100) ), axis=1)#.reshape((1, indA.shape[0]))
    print time.time() - tm
    
    for ind, clmn in enumerate(iterateClmns):
#        print dt.datetime.today()#, clmn[2]
    
        if type(glblIndex)!=list:
            newMainDf.loc[glblIndex, clmn[0]] = tmp[ind*2]
            newMainDf.loc[glblIndex, clmn[0]+'RD'] = tmp[ind*2+1]
        else:
            newMainDf[clmn[0]] = tmp[ind*2]
            newMainDf[clmn[0]+'RD'] = tmp[ind*2+1]
        
        replaceClmn = clmn[0].replace('A', 'B')
        newMainDf.loc[beforeInd, replaceClmn] = newMainDf.loc[afterInd, clmn[0]].values
        newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[beforeInd, clmn[0]].values
        newMainDf.loc[beforeInd, replaceClmn+'RD'] = newMainDf.loc[afterInd, clmn[0]+'RD'].values
        newMainDf.loc[afterInd, replaceClmn+'RD'] = newMainDf.loc[beforeInd, clmn[0]+'RD'].values

    if type(glblIndex)!=list:
        prefix = 'Spec'
    else:
        prefix = ''
        
    for ind, clmn in enumerate(['teamA_n_last_maps', 'teamA_last_match_passed_days']):
        if type(glblIndex)!=list:
            newMainDf.loc[glblIndex, clmn+prefix] = tmp[-2+ind]
        else:
            newMainDf[clmn] = tmp[-2+ind]

        replaceClmn = (clmn+prefix).replace('A', 'B')
        newMainDf.loc[beforeInd, replaceClmn] = newMainDf.loc[afterInd, clmn+prefix].values
        newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[beforeInd, clmn+prefix].values
    return newMainDf
    

def noHistoryDefine(newMainDf, iterateClmns):
    #TODO noHistoryDefine
    #NOT TIME WEIGHTED
    
    for clmn in iterateClmns:
        newMainDf[clmn[0]] = newMainDf[clmn[1]]<0
        newMainDf[clmn[0]] = newMainDf[clmn[1]]<0
        newMainDf.loc[pd.isna(newMainDf[clmn[1]]), clmn[0]] = True
        for lcl in clmn[1:-2]:
            if clmn[-1]:
                newMainDf.loc[newMainDf[lcl]<0, lcl] = 0.5
            else:
                newMainDf.loc[newMainDf[lcl]<0, lcl] = newMainDf.loc[~newMainDf[clmn[0]], lcl].mean()
                
            newMainDf.loc[newMainDf[clmn[-2]]<=1, lcl+'RD'] = 1
            newMainDf.loc[pd.isna(newMainDf[lcl+'RD']), lcl+'RD'] = 1
    
    if any(['Spec' not in clmn[0] for clmn in iterateClmns]):        
        newMainDf['teamA_n_last_maps'] = np.maximum(2, newMainDf['teamA_n_last_maps'].values)
        newMainDf['teamB_n_last_maps'] = np.maximum(2, newMainDf['teamB_n_last_maps'].values)
    
    if any(['Spec' in clmn[0] for clmn in iterateClmns]):
        newMainDf['teamA_n_last_mapsSpec'] = np.maximum(2, newMainDf['teamA_n_last_mapsSpec'].values)
        newMainDf['teamB_n_last_mapsSpec'] = np.maximum(2, newMainDf['teamB_n_last_mapsSpec'].values)
        
    return newMainDf

def absoluteForceCalculationLocal(data, dataScaled, knnInd, baseInd):
    mnVal = data[baseInd]
    return map(lambda x, y: np.percentile(dataScaled[x], y*100), knnInd, mnVal)

def absoluteForceCalculation(newMainDf, newMainDfScaled, knn, clmns, iterateClmns, glblIndex=[], tmpIndex=np.array([])):
    #TODO absoluteForceCalculation
    #tmpIndex - индексы  карт, которые были сыграны
    if type(glblIndex)==list:
        baseIndex = newMainDfScaled.index
    else:
        baseIndex = newMainDf.loc[glblIndex].index
#        baseIndex = allIndex
        
    dst, ind = knn.kneighbors(newMainDfScaled.loc[baseIndex, clmns].values)
    ind = tmpIndex[ind]
#    ind = baseIndex[ind]
    
    if type(glblIndex)!=list:
#        iterateClmns = [('teamAabsForceWeightedSpec', 'teamARelativeForceWeightedHistorySpec'), 
#                        ('teamAabsForceSpec', 'teamARelativeForceHistorySpec'), 
#                        ('teamAabsForceWeightedSpecW', 'teamARelativeForceWeightedHistorySpecW'), 
#                        ('teamAabsForceSpecW', 'teamARelativeForceHistorySpecW')]
        
        dataShape = newMainDf.shape[0]
        beforeInd = [i for i in glblIndex if i<dataShape/2]
        afterInd = [i for i in glblIndex if i>=dataShape/2]
    else:
#        iterateClmns = [('teamAabsForceWeighted', 'teamARelativeForceWeightedHistory'), 
#                        ('teamAabsForce', 'teamARelativeForceHistory'), 
#                        ('teamAabsForceWeightedW', 'teamARelativeForceWeightedHistoryW'), 
#                        ('teamAabsForceW', 'teamARelativeForceHistoryW')]
        
        beforeInd = np.arange(newMainDf.shape[0]/2)
        afterInd = np.arange(newMainDf.shape[0]/2, newMainDf.shape[0])
        
    for clmn in iterateClmns:
        print clmn
#        tm = time.time()
#        newMainDf.loc[baseIndex, clmn[0]] = np.concatenate(map(lambda knnInd, baseInd: absoluteForceCalculationLocal(newMainDf[clmn[1]].values, newMainDfScaled.loc[tmpIndex, 'teamAMatchRatingMine'].values, knnInd, baseInd), np.array_split(ind, 100), np.array_split(baseIndex, 100) ))
        newMainDf.loc[baseIndex, clmn[0]] = np.concatenate(map(lambda knnInd, baseInd: absoluteForceCalculationLocal(newMainDf[clmn[1]].values, newMainDfScaled[clmn[2]].values, knnInd, baseInd), np.array_split(ind, 100), np.array_split(baseIndex, 100) ))
#        print time.time()-tm
        replaceClmn = clmn[0].replace('A', 'B')
        newMainDf.loc[beforeInd, replaceClmn] = newMainDf.loc[afterInd, clmn[0]].values
        newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[beforeInd, clmn[0]].values
        
    
    return newMainDf


def inprob_scale_cond(rv, low, high, scaleLow, scaleHigh, cond):
    rv[cond] = scaleLow + (scaleHigh - scaleLow) * (rv[cond] - low)/(high - low)
    return rv

def probForceCalculationLocal(team, dataScaled, knnAInd, knnBInd, baseInd):#, clmnA, clmnB):
        
    sz = 1000
    sz1 = 10
    shp = knnAInd.shape[0]
    
    am = team[0].reshape(shp, 1)
    bm = team[2].reshape(shp, 1)
    
    astd = team[1].reshape(shp, 1)
    bstd = team[3].reshape(shp, 1)
    
    nA = team[4].reshape(shp, 1)
    nB = team[5].reshape(shp, 1)
    
    aTm = np.random.standard_t(nA-1, size=(shp, sz)) * (astd/((nA-1)**0.5)) + am
    bTm = np.random.standard_t(nB-1, size=(shp, sz)) * (bstd/((nB-1)**0.5)) + bm
    
#     Atppf = t.ppf(np.array([.001, .999]), nA-1)
#     alow, ahigh = (Atppf * (astd/(nA**0.5)) + am).T
#     Btppf = t.ppf(np.array([.001, .999]), nB-1)
#     blow, bhigh = (Btppf * (bstd/(nB**0.5)) + bm).T
    
#     aTm = np.array(map(lambda x: inprob_scale_cond(aTm[x], alow[x], am[x], -0.0001, am[x], aTm[x]<am[x]), range(shp)))
#     aTm = np.array(map(lambda x: inprob_scale_cond(aTm[x], am[x], ahigh[x], am[x], 1.0001, aTm[x]>am[x]), range(shp)))

#     bTm = np.array(map(lambda x: inprob_scale_cond(bTm[x], blow[x], bm[x], -0.0001, bm[x], bTm[x]<bm[x]), range(shp)))
#     bTm = np.array(map(lambda x: inprob_scale_cond(bTm[x], bm[x], bhigh[x], bm[x], 1.0001, bTm[x]>bm[x]), range(shp)))

    aTm = np.maximum(0.01, np.minimum(0.99, aTm))
    bTm = np.maximum(0.01, np.minimum(0.99, bTm))
    
    aTm = aTm.repeat(sz1, axis=1).reshape(shp, -1)
    bTm = bTm.repeat(sz1, axis=1).reshape(shp, -1)

    a = np.random.normal(size=(shp, sz*sz1)) * astd + aTm#.reshape(shp, -1)
    b = np.random.normal(size=(shp, sz*sz1)) * bstd + bTm#.reshape(shp, -1)
    
    a = np.maximum(0, np.minimum(1, a))
    b = np.maximum(0, np.minimum(1, b))
    
    datA = dataScaled[knnAInd]
    datB = dataScaled[knnBInd]
    
    aVal = np.array(map(lambda x, y: np.percentile(x, y*100), datA, a))
    bVal = np.array(map(lambda x, y: np.percentile(x, y*100), datB, b))

    return (aVal>=bVal).sum(axis=1)/float(sz*sz1)

def probForceCalculation(newMainDf, newMainDfScaled, knn, clmnA, iterateClmns, glblIndex=[], tmpIndex=np.array([]), mode=''):
    #TODO probForceCalculation
    if type(glblIndex)==list:
        baseIndex = newMainDfScaled.index
        beforeInd = baseIndex[:newMainDfScaled.shape[0]/2]
        afterInd = baseIndex[newMainDfScaled.shape[0]/2:]
    else:
        baseIndex = newMainDf.loc[glblIndex].index
        dataShape = newMainDf.shape[0]
        beforeInd = [i for i in baseIndex if i<dataShape/2]
        afterInd = [i for i in baseIndex if i>=dataShape/2]
        
    if not baseIndex.shape[0]:
        return newMainDf
#    clmnB = [i.replace('A', 'B') if 'A' in i else i.replace('B', 'A') for i in clmnA]
#    print clmnA
#    print clmnB
    dstA, indA = knn.kneighbors(newMainDfScaled.loc[beforeInd, clmnA].values)
    dstB, indB = knn.kneighbors(newMainDfScaled.loc[afterInd, clmnA].values)
        
    for clmn in iterateClmns:
        print clmn
#        tm = time.time()
        replaceClmn = clmn[1].replace('A', 'B')
        if 'Spec' in clmn[1]:
            prefix = 'Spec'
        else:
            prefix = ''
        
        newMainDf.loc[beforeInd, clmn[0]+'Prob'] = np.concatenate(map(lambda knnInd, baseInd: probForceCalculationLocal(newMainDf.loc[baseInd, [clmn[1], clmn[1]+'RD', replaceClmn, replaceClmn+'RD', 'teamA_n_last_maps'+prefix, 'teamB_n_last_maps'+prefix]].values.T, 
                         newMainDfScaled[clmn[2]].values, tmpIndex[indA[knnInd]], tmpIndex[indB[knnInd]], baseInd), np.array_split(np.arange(len(indA)), min(len(indA), 100)), np.array_split(beforeInd, min(len(indA), 100)) ))
        
        replaceClmn = (clmn[0]+'Prob').replace('A', 'B')
        newMainDf.loc[beforeInd, replaceClmn] = 1 - newMainDf.loc[beforeInd,  clmn[0]+'Prob']
        newMainDf.loc[afterInd, replaceClmn] = newMainDf.loc[beforeInd,  clmn[0]+'Prob'].values
        newMainDf.loc[afterInd,  clmn[0]+'Prob'] = newMainDf.loc[beforeInd, replaceClmn].values
#        print time.time()-tm
        print ((newMainDf.loc[baseIndex, clmn[0]+'Prob'].values>0.5)==(newMainDf.loc[baseIndex, 'isWin'].values>0)).sum()/float(newMainDf.loc[baseIndex].shape[0])
    
    return newMainDf

