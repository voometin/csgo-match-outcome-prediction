# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:11:57 2018

@author: Andrew
"""

import scipy.stats
import trueskill
import pandas as pd
import numpy as np
import datetime as dt
import time
from tqdm import tqdm
#from trueskill.mathematics import cdf

#import itertools
import math

#trueskill.BETA
#BETA = 25/6.0

#print trueskill.BETA
def win_probability(team1, team2):
#    delta_mu = team1['mu'] - team2['mu']
    delta_mu = team1.mu - team2.mu
#    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
#    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
#    sum_sigma = team1['sigma']**2 + team2['sigma']**2
    sum_sigma = team1.sigma**2 + team2.sigma**2
#    return trueskill.TrueSkill().cdf(delta_mu/sum_sigma)
    #normDist = scipy.stats.norm(delta_mu, sum_sigma).cdf(0)
#    size = len(team1) + len(team2)
#    size = 2
    denom = math.sqrt(2 * trueskill.BETA**2 + sum_sigma)
#    print delta_mu, denom
#    return normDist
#    ts = trueskill.global_env()
    return trueskill.TrueSkill().cdf(delta_mu / denom)

def win_probability1(raw):
    delta_mu = raw['ratingA'] - raw['ratingB']
    sum_sigma = raw['rdA']**2 + raw['rdB']**2
    denom = math.sqrt(2 * trueskill.BETA**2 + sum_sigma)
    return trueskill.TrueSkill().cdf(delta_mu / denom)


#team1 = {'mu': 25, 'sigma': 6.458}
#team2 = {'mu': 25, 'sigma': 6.458}
#print win_probability(team1, team2)
##trueskill.trueskill
#
t1 = trueskill.Rating(50, 6)
t2 = trueskill.Rating(25, 6)
#print trueskill.BETA, trueskill.SIGMA
#print win_probability(t2, t1)
#print t1.mu, t1.sigma, t2.mu, t2.sigma
#t2, t1 = trueskill.rate_1vs1(t2, t1)
##t1, t2 = trueskill.rate_1vs1(t1, t2)
#print t1.mu, t1.sigma, t2.mu, t2.sigma
##print trueskill.quality_1vs1(t2, t1, draw=True)
##C:\ProgramData\Anaconda2\lib\site-packages\trueskill\__init__.py

def ratingCalculation(lcldf, unratePer, unrate=False):
    global ratingDfMain
#    lstAErr, lstBErr = [], []
##    ttlSec = float(t * 24 * 3600.0)
#    tmdelta = dt.timedelta(days=ratePer)#период за который смотреть историю игр
    unratedTDelta = dt.timedelta(days=unratePer)#период за который команда считается unrated
    acc = []
    listErr = []
    for _, x in lcldf.iterrows():#.iloc[:100]
        pass
#        if np.isnan(x['difScore']):
#            continue
#        print trueskill.BETA
#        timeDlt = x['date'] - tmdelta
        stDate = x['date'] - unratedTDelta
#        dateDif = startDate
#        while startDate + tmdelta<x['date']:
#            rInd += 1
#            startDate = dt.datetime.strptime(rKeys[rInd], '%Y-%m-%d')
        
#        if str(x['date']).split()[0]>='2016' and not flag:
#            print x['date']
#            flag = True
#        timeDlt = x['date'] - tmdelta
#        timeDlt2 = x['date'] - tmdelta - tmdelta
        
#        rdA, ratingA = rdRatingReturn(x['teamAId'], timeDlt, timeDlt2, c)
#        rdB, ratingB = rdRatingReturn(x['teamBId'], timeDlt, timeDlt2, c)
                        
        
        if unrate:
            lclTtlA = lcldf.iloc[:_-1][((lcldf.iloc[:_-1]['teamAId']==x['teamAId']) | (lcldf.iloc[:_-1]['teamBId']==x['teamAId'])) & (lcldf.iloc[:_-1]['date']>=stDate) & (lcldf.iloc[:_-1]['date']<=x['date'])].copy()
            lclTtlB = lcldf.iloc[:_-1][((lcldf.iloc[:_-1]['teamAId']==x['teamBId']) | (lcldf.iloc[:_-1]['teamBId']==x['teamBId'])) & (lcldf.iloc[:_-1]['date']>=stDate) & (lcldf.iloc[:_-1]['date']<=x['date'])].copy() 
            
            
            if lclTtlA.shape[0]==0:
                ratingDfMain[x['teamAId']] = [trueskill.Rating(max(25, ratingDfMain[x['teamAId']][2].mu), max(6.458, ratingDfMain[x['teamAId']][2].sigma)),
                                                x['matchlinkId'], 
                                                trueskill.Rating(max(25, ratingDfMain[x['teamAId']][2].mu), max(6.458, ratingDfMain[x['teamAId']][2].sigma))]
            if lclTtlB.shape[0]==0:
                ratingDfMain[x['teamBId']] = [trueskill.Rating(max(25, ratingDfMain[x['teamBId']][2].mu), max(6.458, ratingDfMain[x['teamBId']][2].sigma)), 
                                                x['matchlinkId'], 
                                                trueskill.Rating(max(25, ratingDfMain[x['teamBId']][2].mu), max(6.458, ratingDfMain[x['teamBId']][2].sigma))]

        if x['matchlinkId']!=ratingDfMain[x['teamAId']][1]:
            ratingDfMain[x['teamAId']][0] = ratingDfMain[x['teamAId']][2]
        if x['matchlinkId']!=ratingDfMain[x['teamBId']][1]:
            ratingDfMain[x['teamBId']][0] = ratingDfMain[x['teamBId']][2]
            
        lcldf.loc[_, ['ratingA', 'rdA']] = ratingDfMain[x['teamAId']][0].mu, ratingDfMain[x['teamAId']][0].sigma
        lcldf.loc[_, ['ratingB', 'rdB']] = ratingDfMain[x['teamBId']][0].mu, ratingDfMain[x['teamBId']][0].sigma
        probA = win_probability(ratingDfMain[x['teamAId']][0], ratingDfMain[x['teamBId']][0])
            
        lcldf.loc[_, 'pred'] = probA > 0.5
        
        if not np.isnan(x['difScore']):
            if x['difScore']==0.5:#draw
                ratingDfMain[x['teamAId']][2], ratingDfMain[x['teamBId']][2] = trueskill.rate_1vs1(ratingDfMain[x['teamAId']][2], ratingDfMain[x['teamBId']][2], drawn = True)
            elif x['difScore']>0.5:#teamA win
                ratingDfMain[x['teamAId']][2], ratingDfMain[x['teamBId']][2] = trueskill.rate_1vs1(ratingDfMain[x['teamAId']][2], ratingDfMain[x['teamBId']][2])
            else:#teamB win
                ratingDfMain[x['teamBId']][2], ratingDfMain[x['teamAId']][2] = trueskill.rate_1vs1(ratingDfMain[x['teamBId']][2], ratingDfMain[x['teamAId']][2])
            
            if (probA>0.5 and x['difScore']>0.5) or (probA<0.5 and x['difScore']<0.5):
                acc.append(True)
            else:
                acc.append(False)
                
            if (probA>x['difScore'] and x['difScore']>0.5) or (probA<x['difScore'] and x['difScore']<0.5):
                err = 0
            else:
                err = abs(probA-x['difScore'])
            listErr.append(err)
    #        print probA, x['difScore'], err, sum(listErr)
            
            
        ratingDfMain[x['teamAId']][1] = x['matchlinkId']
        ratingDfMain[x['teamBId']][1] = x['matchlinkId']
            
        
#        expScore = expScoreCalc(gRDCalc(newRDB, True), newRatingA, newRatingB, True)
#        err = abs(x['difScore'] - expScore)
#        if (expScore>0.5 and x['difScore']>0.5) or (expScore<0.5 and x['difScore']<0.5):
#            lstAErr += [[err, err/2.0]]
#        else:
#            lstAErr += [[err, err]]
#            
#        expScore = expScoreCalc(gRDCalc(newRDA, True), newRatingB, newRatingA, True)
#        err = abs(1 - x['difScore'] - expScore)
#        if (expScore>0.5 and 1 - x['difScore']>0.5) or (expScore<0.5 and 1 - x['difScore']<0.5):
#            lstBErr += [[err, err/2.0]]
#        else:
#            lstBErr += [[err, err]]
#        lstBErr += [-x['difScore'] - (32 * expScore - 16)]
            
#    print lcldf.describe()
#    with open('glicko2.txt', 'a') as f:
##        f.write('winBonus, unratePer, tau, ratePer: %s\n'%str([winBonus, unratePer, tau, ratePer]))
#        f.write('unratePer, tau, ratePer: %s\n'%str([unratePer, tau, ratePer]))
#        f.write(str(lcldf.describe())+'\n')
#    print (lcldf['difScore']==0.5).sum()/float(lcldf.shape[0])
#    print 'shape before', lcldf.shape[0]a
#    lcldf.drop(lcldf[lcldf['difScore']==0.5].index, inplace=True)
    cond = lcldf['date']>dt.date(2015, 12, 31)
#    print 'shape after', lcldf.shape[0]
#    lcldf['pred'] = lcldf['ratingA'] > lcldf['ratingB']
    lcldf['pred1'] = lcldf['difScore'] > 0.5
#    with open('trueskill.txt', 'a') as f:
#        f.write('BETA: %s unratePer: %s\naccuracy: %s\n'%(str(trueskill.BETA), str(unratePer/7), str((lcldf[cond]['pred'] == lcldf[cond]['pred1']).sum()/float(lcldf[cond].shape[0]))))
#        f.write('abs error A: %s\n'%str(map(sum, zip(*lstAErr))))
#        f.write('abs error B: %s\n'%str(map(sum, zip(*lstBErr))))
#        f.write('abs error A MEAN: %s\n'%str(map(np.mean, zip(*lstAErr))))
#        f.write('abs error B MEAN: %s\n'%str(map(np.mean, zip(*lstBErr))))
    print 'BETA: %s unratePer: %s\naccuracy: %s'%(str(trueskill.BETA), str(unratePer/7), str((lcldf[cond]['pred'] == lcldf[cond]['pred1']).sum()/float(lcldf[cond].shape[0])))
    return {'acc': round(sum(acc)/float(len(acc)), 5), 'err': -sum(listErr)}
#    print 'accuracy', (lcldf[cond]['pred'] == lcldf[cond]['pred1']).sum()/float(lcldf[cond].shape[0]), lcldf[cond].shape, lcldf.shape
#    print 'abs error A', map(sum, zip(*lstAErr))
#    print 'abs error B', map(sum, zip(*lstBErr))
#    print 'abs error A MEAN', map(np.mean, zip(*lstAErr))
#    print 'abs error B MEAN', map(np.mean, zip(*lstBErr))
#    lcldf.drop(lcldf[lcldf['difScore']==0.5].index, inplace=True)
#            print 'YEAH'
#        return
def saveToFile(filename):
    global df
    with open(filename, 'w') as f:
        df[['matchlinkId', 'teamAId', 'teamBId', 'map', 'ratingA', 'ratingB', 'rdA', 'rdB', 'probA', 'probB']].to_csv(f, sep=';', index=False)        
        
import pymongo  
      
trueskill.BETA = 25/6.0
client = pymongo.MongoClient(port=33333)
db = client['csgo']  
        
def bayes_func(unratePer):#, days):#, par=400, inf=True):
    global ratingDfMain, ratingDf
    df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
    
#    print df.describe()
    df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    #                df = df.sort_values('date').reset_index(drop=True)
    #df['difScore'] = df['difScore'].apply(lambda x: 0.5 if x==0 else 0 if x<0 else 1)
    df.loc[~df.isnull().any(axis=1), 'difScore'] = df.loc[~df.isnull().any(axis=1), 'difScore']/32.0 + 0.5
#    df['difScore'] = df['difScore']/32.0 + 0.5
    
    df['ratingA'] = 25
    df['ratingB'] = 25
    df['rdA'] = 6.458
    df['rdB'] = 6.458
    #df['sigmaA'] = 0.06
    #df['sigmaB'] = 0.06
    
    #print df.info()
#    params = 7
#    unratePer = 112
#    print df.describe()
#    raise 'sdfsdf'
    #        ratePer = 8
    #                params = [df, winBonus, params*unratePer, tau, params*ratePer]
    params = [df, unratePer, True]
    
    teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
    ratingDfMain = {}
#    mindate = df['date'].min()
#    maxdate = df['date'].max()
#    lcldate = mindate
#    print maxdate
    for ids in teamIds:
        ratingDfMain[ids] = [trueskill.Rating(25, 6.458), 0, trueskill.Rating(25, 6.458)]
    
#    params = 
    print dt.datetime.today()
    print params[1:]
    tm = time.time()
    dc = ratingCalculation(*params)
    print 'time', time.time() - tm, dt.datetime.today()
    print dc
    
#    df['probA'] = np.round(df.apply(lambda x: win_probability1(x), axis=1).values, 6)
#    df['probB'] = 1 - df['probA'].values
    
    
    return dc['acc']
    return -dc['error']


#from bayes_opt import BayesianOptimization
##print bayes_func(0, 1, 400, 110)
#
##winBonus, k, par, days, inf
#bo = BayesianOptimization(bayes_func, {'unratePer': (45, 150)})#, 'days': (100, 200)})
#KAPPA = 5
##gp_params = {'kernel': 'cubic'}
#bo.maximize(init_points=5, n_iter=20, acq='ucb', kappa=KAPPA)#, **gp_params)
#print bo.res['max']

#{'max_params': {'unratePer': 45.01695450784352}, 'max_val': 0.60965}

df = {}
#for unratePer in np.arange(12, 21, 2):
#    for BETA in [3.5, 3.75, 4.0, 25/6.0, 4.25, 4.5]:
#        trueskill.BETA = BETA
def main_func():
    global ratingDfMain, ratingDf, df
    for unrate in [True, False]:
        df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
        df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        #                df = df.sort_values('date').reset_index(drop=True)
        #df['difScore'] = df['difScore'].apply(lambda x: 0.5 if x==0 else 0 if x<0 else 1)
        df.loc[~df.isnull().any(axis=1), 'difScore'] = df.loc[~df.isnull().any(axis=1), 'difScore']/32.0 + 0.5
    #    df['difScore'] = df['difScore']/32.0 + 0.5
        
        df['ratingA'] = 25
        df['ratingB'] = 25
        df['rdA'] = 6.458
        df['rdB'] = 6.458
        #df['sigmaA'] = 0.06
        #df['sigmaB'] = 0.06
        
        #print df.info()
    #    params = 7
        unratePer = 45
        #        ratePer = 8
        #                params = [df, winBonus, params*unratePer, tau, params*ratePer]
        params = [df, unratePer, unrate]
        
        teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
        ratingDfMain = {}
        maxdate = df['date'].max()
    #    print maxdate
        for ids in teamIds:
            ratingDfMain[ids] = [trueskill.Rating(25, 6.458), 0, trueskill.Rating(25, 6.458)]
        
        
        print dt.datetime.today()
        print params[1:]
        tm = time.time()
        dc = ratingCalculation(*params)
        print 'time', time.time() - tm, dt.datetime.today()
        print dc
        
        df['probA'] = np.round(df.apply(lambda x: win_probability1(x), axis=1).values, 6)
        df['probB'] = 1 - df['probA'].values
        
        if unrate:
            filename = 'trueskillRatingUnrate.csv'
        else:
            filename = 'trueskillRating.csv'
        saveToFile(filename)
        
        unratedTDelta = dt.timedelta(days=unratePer)
        stDate = maxdate - unratedTDelta 
        for ids in teamIds:
            lclTtl = df[((df['teamAId']==ids) | (df['teamBId']==ids)) & (df['date']>=stDate)].copy()
            if not lclTtl.shape[0]:
                ratingDfMain.pop(ids)
        
        #        with open('trueskillRating.txt', 'a') as f:
        #            print 'best'
        #            f.write('\nBETA: %s, unratePer: %s\n'%(str(BETA), str(unratePer)))
        for val in sorted(ratingDfMain.items(), key=lambda x: x[1][2].mu, reverse=True)[:10]:
        
            rs = db['fullMatchHistory'].find_one({'$or': [{'teamBId': str(val[0])}, {'teamAId': str(val[0])}]})
            if rs:
                if rs['teamAId']==str(val[0]):
        #                        f.write('%s %s\n'%(rs['teamAUrlName'], str((val[1].mu, val[1].sigma)) ))
                    print rs['teamAUrlName'], (val[1][2].mu, val[1][2].sigma)
                else:
        #                        f.write('%s %s\n'%(rs['teamBUrlName'], str((val[1].mu, val[1].sigma)) ))
                    print rs['teamBUrlName'], (val[1][2].mu, val[1][2].sigma)
                
      
main_func()
client.close()
      
#t1 = trueskill.Rating(1200, 50)
#t2 = trueskill.Rating(1100, 50)
##print trueskill.BETA, trueskill.SIGMA
#print win_probability1(t2, t1)
#        print 'worst'
#        for val in sorted(ratingDfMain.items(), key=lambda x: x[1].mu)[:10]:
#            print val
