# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:11:57 2018

@author: Andrew
"""

from random import random, sample, choice
from tqdm import tqdm
#from numpy import array, dot, mean
from numpy.linalg import pinv
import numpy as np
import pandas as pd
import datetime as dt
import time
import copy

import trueskill
import math


def win_probability(raw):
    delta_mu = raw['ratingA'] - raw['ratingB']
    sum_sigma = 2 * raw['rdA']**2
    denom = math.sqrt(sum_sigma)
    return trueskill.TrueSkill().cdf(delta_mu / denom)

def expScoreCalc(rA, rB, par):
#    try:
        if (rB-rA)/float(par)>8:
            return 0
        return 1/float(1 + 10**( (rB-rA)/float(par) ) )
#    except:
#        print rB, rA, par, (rB-rA)/float(par)
#        return 'hahahahah'

def difScoreCalc(actScore, expScore):
    return actScore - (32 * expScore - 16)

def changeEloRating(r, winBonus, k, difScore):
    return r + winBonus + k * difScore

def ratingInflation(rating, date1, date2, days):
    dateDif = (date2 - date1).days
    if dateDif>=days:
#        if rating<1000:
#            return rating
        return 1000
    if rating<1000:
        return rating
    return 1000 + (days-dateDif)/float(days)*(rating-1000)

def chageRating(x, winBonus, k, par, days, inf):
#    try:
    
    #    teamARt = ratingDf.loc[x['teamAId']].copy()
    #    teamBRt = ratingDf.loc[x['teamBId']].copy()
    #    teamARt = ratingDf[x['teamAId']]
    #    teamBRt = ratingDf[x['teamBId']]
            
        if inf:
            ratingA = ratingInflation(ratingDf[x['teamAId']][2], ratingDf[x['teamAId']][3], x['date'], days)
            ratingB = ratingInflation(ratingDf[x['teamBId']][2], ratingDf[x['teamBId']][3], x['date'], days)
        else:
            ratingA = ratingDf[x['teamAId']][2]
            ratingB = ratingDf[x['teamBId']][2]
        
        if x['matchlinkId']!=ratingDf[x['teamAId']][1]:
            ratingDf[x['teamAId']][0] = ratingA
        if x['matchlinkId']!=ratingDf[x['teamBId']][1]:
            ratingDf[x['teamBId']][0] = ratingB
#        expScore = expScoreCalc(ratingA, ratingB, par)
        if x['difScore']>0:
    #        ratingDf.loc[x['teamAId'], 'rating'] = changeEloRating(ratingA, winBonus, k, difScore)
    #        ratingDf.loc[x['teamBId'], 'rating'] = changeEloRating(ratingB, 0, k, difScore)
            expScore = expScoreCalc(ratingA, ratingB, par)
            difScore = difScoreCalc(x['difScore'], expScore)
            newRating = changeEloRating(ratingA, winBonus, k, difScore)
            if newRating>3000:
                newRating = 3000
            ratingDf[x['teamAId']][2] = newRating
            
            expScore1 = expScoreCalc(ratingB, ratingA, par)
            difScore1 = difScoreCalc(-x['difScore'], expScore1)
            newRating = changeEloRating(ratingB, 0, k, difScore1)
            if newRating>3000:
                newRating = 3000
            ratingDf[x['teamBId']][2] = newRating
        elif x['difScore']<0:
    #        ratingDf.loc[x['teamAId'], 'rating'] = changeEloRating(ratingA, 0, k, difScore)
    #        ratingDf.loc[x['teamBId'], 'rating'] = changeEloRating(ratingB, winBonus, k, difScore)
            expScore = expScoreCalc(ratingA, ratingB, par)
            difScore = difScoreCalc(x['difScore'], expScore)
            newRating = changeEloRating(ratingA, 0, k, difScore)
            if newRating>3000:
                newRating = 3000
            ratingDf[x['teamAId']][2] = newRating
            
            expScore1 = expScoreCalc(ratingB, ratingA, par)
            difScore1 = difScoreCalc(-x['difScore'], expScore1)
            newRating = changeEloRating(ratingB, winBonus, k, difScore1)
            if newRating>3000:
                newRating = 3000
            ratingDf[x['teamBId']][2] = newRating
        else:
    #        ratingDf.loc[x['teamAId'], 'rating'] = changeEloRating(ratingA, 0, k, difScore)
    #        ratingDf.loc[x['teamBId'], 'rating'] = changeEloRating(ratingB, 0, k, difScore)
            expScore = expScoreCalc(ratingA, ratingB, par)
            difScore = difScoreCalc(x['difScore'], expScore)
            newRating = changeEloRating(ratingA, 0, k, difScore)
            if newRating>3000:
                newRating = 3000
            ratingDf[x['teamAId']][2] = newRating
            
            expScore1 = expScoreCalc(ratingB, ratingA, par)
            difScore1 = difScoreCalc(-x['difScore'], expScore1)
            newRating = changeEloRating(ratingB, 0, k, difScore1)
            if newRating>3000:
                newRating = 3000
            ratingDf[x['teamBId']][2] = newRating
            
        ratingDf[x['teamAId']][3] = x['date']
        ratingDf[x['teamBId']][3] = x['date']
        
        ratingDf[x['teamAId']][1] = x['matchlinkId']
        ratingDf[x['teamBId']][1] = x['matchlinkId']
            
#        if abs(difScore)!=abs(difScore1) and (expScore>1 or expScore<0 or abs(difScore)>32 or abs(difScore1)>32):
#            print 'ABSSSSSSSS', difScore, difScore1, expScore, expScore1
            
        if difScore>0:
            kef = 0.5
        else:
            kef = 1
            
        if difScore1>0:
            kef1 = 0.5
        else:
            kef1 = 1
            
        if expScore>=x['difScore'] and x['difScore']>0 or expScore<=x['difScore'] and x['difScore']<0:
            err = 0
            err1 = 0
        else:
            err = np.mean([abs(difScore), abs(difScore1)])
            err1 = np.mean([kef*abs(difScore), kef1*abs(difScore1)])
            
#        return ratingDf[x['teamAId']][0], ratingDf[x['teamBId']][0], np.mean([kef*abs(difScore), kef1*abs(difScore1)]), np.mean([abs(difScore), abs(difScore1)]), (32 * expScore - 16) * x['difScore'] >= 0, (32 * expScore1 - 16) * -x['difScore'] >= 0
        return ratingDf[x['teamAId']][0], ratingDf[x['teamBId']][0], err, err1, (32 * expScore - 16) * x['difScore'] > 0, (32 * expScore1 - 16) * -x['difScore'] > 0
#        if (32 * expScore - 16) * x['difScore'] < 0:
#            return ratingA, ratingB, np.mean([abs(difScore), abs(difScore1)])
#        else:
#            return ratingA, ratingB, 0
#        return pd.Series({'ratingA': ratingA, 'ratingB': ratingB, 'errScore': difScore})
#    except:
#        print x.shape
#        return 0, 0, 0
#        return pd.Series({'ratingA': 0, 'ratingB': 0, 'errScore': 0})
#        return (0, 0, 0, )



def get_fitness(lcldf, kefs, mode=False, inf=True):
    global ratingDf
    winBonus, k, par, days = kefs
#    print 'winBonus, k, par, days:', winBonus, k, par, days
    tttt, tttt1 = [], []
    lstA, lstB, lst, lst1 = [], [], [], []
    for _, x in lcldf.iterrows():
        if np.isnan(x['difScore']):
#            lstA += [ratingA]
            if lcldf.loc[_-1, 'matchlinkId']!=x['matchlinkId']:
                
#                print lcldf.loc[:2].index.values
#                return 
                tmpAIndex = lcldf.loc[:_-1][(lcldf.loc[:_-1, 'teamAId']==x['teamAId']) | (lcldf.loc[:_-1, 'teamAId']==x['teamBId'])].index.values
                tmpBIndex = lcldf.loc[:_-1][(lcldf.loc[:_-1, 'teamBId']==x['teamBId']) | (lcldf.loc[:_-1, 'teamBId']==x['teamAId'])].index.values
#                print tmpAIndex
                if not tmpAIndex.shape[0]:
                    lstA += [1000]
                else:
#                if tmpAIndex<_:
                    tmpAIndex = tmpAIndex[-1]
                    if lcldf.loc[tmpAIndex, 'teamAId']==x['teamAId']:
                        lstA += [lstA[tmpAIndex]]
                    else:
                        lstA += [lstB[tmpAIndex]]
                    
                if not tmpBIndex.shape[0]:
                    lstB += [1000]
                else:
#                if tmpBIndex<_:
                    tmpBIndex = tmpBIndex[-1]
                    if lcldf.loc[tmpBIndex, 'teamBId']==x['teamBId']:
                        lstB += [lstB[tmpBIndex]]
                    else:
                        lstB += [lstA[tmpBIndex]]
                    
#                print _, tmpAIndex, tmpBIndex#, lcldf.loc[tmpAIndex, 'teamAId'], lcldf.loc[tmpAIndex, 'teamBId'], lcldf.loc[tmpBIndex, 'teamAId'], lcldf.loc[tmpBIndex, 'teamBId']
#                print
            else:
                lstA += [lstA[-1]]
                lstB += [lstB[-1]]
            lst += [0]
            lst1 += [0]
            tttt += [True]
            tttt1 += [True]
        else:        
            ratingA, ratingB, difScore, difScore1, wrong, wrong1 = chageRating(x, winBonus, k, par, days, inf)
            lstA += [ratingA]
            lstB += [ratingB]
            lst += [abs(difScore)]
            lst1 += [abs(difScore1)]
            tttt += [wrong]
            tttt1 += [wrong1]
#    return
    lcldf.loc[lcldf.index, ['ratingA', 'ratingB', 'errScore', 'err2', 'wrongPred', 'wrongPred1']] = pd.DataFrame({'ratingA': lstA, 'ratingB': lstB, 'errScore': lst, 'err2': lst1, 'wrongPred': tttt, 'wrongPred1': tttt1})
    
    error = lcldf[lcldf['date']>dt.date(2015, 12, 31)]['errScore'].sum()
    error1 = lcldf[lcldf['date']>dt.date(2015, 12, 31)]['err2'].sum()
    acc = round(lcldf.loc[lcldf[lcldf['date']>dt.date(2015, 12, 31)].index, 'wrongPred'].sum()/float(lcldf[lcldf['date']>dt.date(2015, 12, 31)].index.shape[0]), 5)
    sigma = sorted(abs(np.array(lstA) - np.array(lstB)))[:int(len(lstA)*0.995)][-1]/3.0
    print '3 sigma', sorted(abs(np.array(lstA) - np.array(lstB)))[:int(len(lstA)*0.995)][-1]
    if mode:
        print 'SUM', lcldf[lcldf['date']>dt.date(2015, 12, 31)]['wrongPred'].sum(), lcldf[lcldf['date']>dt.date(2015, 12, 31)].shape, 1-lcldf[lcldf['date']>dt.date(2015, 12, 31)]['wrongPred'].sum()/float(lcldf[lcldf['date']>dt.date(2015, 12, 31)].shape[0]), 1-lcldf[lcldf['date']>dt.date(2015, 12, 31)]['wrongPred1'].sum()/float(lcldf[lcldf['date']>dt.date(2015, 12, 31)].shape[0])
    else:
        ratingDf = copy.deepcopy(ratingDfMain)
#    print 'ERROR', error
    return {'error': error, 'err2': error1, 'acc': acc, 'coeff': kefs, 'sigma': sigma}, lcldf









#df = pd.read_csv('../cs_new/csgoEloDS.csv', sep=';')
#df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
#df = df.sort_values('date').reset_index(drop=True)
#df['ratingA'] = 1000
#df['ratingB'] = 1000
#df['errScore'] = 1000
#df['err2'] = 1000
#df['wrongPred'] = False
#df['wrongPred1'] = False
##print df.info()
#teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
#ratingDfMain = {}
#mindate = df['date'].min()
#print teamIds.shape
#for ids in teamIds:
#    ratingDfMain[ids] = [1000, mindate]
#    
#ratingDf = copy.deepcopy(ratingDfMain)
##winBonus, k, par, days = 8, 1, 400, 60
#
#
#population_size = 100
#selection_size = floor(0.2*population_size)
#max_generations = 50
#probability_of_individual_mutating = 0.1
#probability_of_gene_mutating = 0.25
#best_individuals_stash = [create_individual()]
#individual_size = len(best_individuals_stash[0])
#initial_population = create_population(population_size)
#current_population = initial_population
#termination = False
#generation_count = 0
#for i in range(20):
##while termination is False:
##    current_best_individual = get_fitness(df.copy(), best_individuals_stash[-1])
#    print 'Generation: ', generation_count
#    best_individuals = evaluate_population(current_population)
#    current_population = get_new_generation(best_individuals)
##    termination = check_termination_condition(current_best_individual)
#    generation_count += 1
#else:
#    print(get_fitness(df.copy(), best_individuals_stash[-1]))
##best_individuals[0][]

def saveToFile(filename):
    global df
    with open(filename, 'w') as f:
        df[['matchlinkId', 'teamAId', 'teamBId', 'map', 'ratingA', 'ratingB']].to_csv(f, sep=';', index=False)
        

import pymongo
client = pymongo.MongoClient(port=33333)

ratingDfMain = {}
ratingDf = {}

def bayes_func(winBonus, k):#, days):#, par=400, inf=True):
    global ratingDfMain, ratingDf
    df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
    df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    #df = df.sort_values('date').reset_index(drop=True)
    df['ratingA'] = 1000
    df['ratingB'] = 1000
    df['errScore'] = 1000
    df['err2'] = 1000
    df['wrongPred'] = False
    df['wrongPred1'] = False
    #print df.info()
    teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
    ratingDfMain = {}
    mindate = df['date'].min()
    #print teamIds.shape, df.info()
    for ids in teamIds:
        ratingDfMain[ids] = [1000, 0, 1000, mindate]
#                [
#                рейтинг_на_момент_начала_серии_BO, 
#                matchlinkId_последней_окончевшейся_серии_BO, 
#                рейтинг_по_окончанию_последней_карты,
#                дата_последнего_изменения_рейтинга
#                ]
        
    days = 100
    par = 400
    inf = True
    ratingDf = copy.deepcopy(ratingDfMain)#.copy()
    
    dc, df = get_fitness(df.copy(), [winBonus, k, par, days], True, inf = inf)
    print dc
    
#    df['rdA'] = dc['sigma']
#    df['rdB'] = dc['sigma']
#    df['probA'] = np.round(df.apply(lambda x: win_probability(x), axis=1).values, 6)
#    df['probB'] = 1 - df['probA'].values
    
    #{'error': error, 'err2': error1, 'acc': acc, 'coeff': kefs, 'sigma': sigma}, lcldf
    return -dc['error']

#from bayes_opt import BayesianOptimization
##print bayes_func(0, 1, 400, 110)
#
##winBonus, k, par, days, inf
#bo = BayesianOptimization(bayes_func, {'winBonus': (0, 15), 'k': (1, 3)})#, 'days': (100, 200)})
#KAPPA = 5
##gp_params = {'kernel': 'cubic'}
#bo.maximize(init_points=10, n_iter=20, acq='ucb', kappa=KAPPA)#, **gp_params)
#print bo.res['max']

#k=1.71, winBonus=0.97, inf=False {'acc': 0.64225, 'err2': 221481.11446846146, 'coeff': [0.9760232361103232, 1.7085533446413426, 400, 100], 'sigma': 96.2841540304449, 'error': 295308.1526246153}
#k=1.8649, winBonus=4.9, inf=True {'acc': 0.63409, 'err2': 223917.12033401077, 'coeff': [4.915826358773401, 1.8648619275308889, 400, 100], 'sigma': 91.72462980456044, 'error': 298556.1604453476}

#{'max_params': {'days': 149.69300685682333,
#  'k': 2.4732934494029317,
#  'winBonus': 15.702419761130475},
# 'max_val': 0.69309}, inf=True
    
#{'max_params': {'days': 158.33052511831815,
#  'k': 1.0,
#  'winBonus': 3.7042647500227246},
# 'max_val': -246710.4880625388}
    
#{{'k': 1.9480784967848763, 'winBonus': 3.9517177392501113},
# 'max_val': 0.69827}

#{'max_params': {'k': 1.7361422609247033, 'winBonus': 1.1152652867610442},
# 'max_val': -244281.77907327304}

    
def main_func():
#cnt = 0
    global ratingDfMain, ratingDf, df
    for winBonus, k, par, days, inf, filename in [
#                                                  (15.7, 2.47, 400, 150, True, 'eloRating0Inf.csv'), 
#                                                  (3.7, 1, 400, 150, True, 'eloRating5Inf.csv'), 
#                                                  (3.95, 1.95, 400, 150, False, 'eloRating0.csv'), 
#                                                  (1.11, 1.73, 400, 150, False, 'eloRating5.csv'),
                                                  (4.9, 1.8649, 400, 150, True, 'eloRating0Inf.csv'), 
                                                  (0.97, 1.71, 400, 150, False, 'eloRating0.csv'), 
                                                  ]:
        df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
        df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
        #df = df.sort_values('date').reset_index(drop=True)
        df['ratingA'] = 1000
        df['ratingB'] = 1000
        df['errScore'] = 1000
        df['err2'] = 1000
        df['wrongPred'] = False
        df['wrongPred1'] = False
        #print df.info()
        teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
        ratingDfMain = {}
        mindate = df['date'].min()
        #print teamIds.shape, df.info()
        for ids in teamIds:
            ratingDfMain[ids] = [1000, 0, 1000, mindate]
    #                [
    #                рейтинг_на_момент_начала_серии_BO, 
    #                matchlinkId_последней_окончевшейся_серии_BO, 
    #                рейтинг_по_окончанию_последней_карты,
    #                дата_последнего_изменения_рейтинга
    #                ]
            
        ratingDf = copy.deepcopy(ratingDfMain)#.copy()
#        winBonus, k, par, days = [winBonus, 1, 400, 110]
        #winBonus, k, par, days = [5, 1, 400, 110]
        #winBonus, k, par, days = [8, 1, 400, 100]
        #winBonus, k, par, days = [8.965746514148387, 4.709022656266433, 648.0666192870394, 105.47184393819215]
        #print get_fitness(df.copy(), [winBonus, k, par, days])
        ##    
        ##
        ##
        ##
        #minError = {'error': 100000000}
        #minError1 = {'err2': 100000000}
        #for winBonus in  range(9)[::-1]:
        #    for days in [40, 50, 60, 70, 80, 90, 100, 110, 120]:
        #        for par in [200, 250, 300, 400, 450, 500, 550, 600]:
        #            dc = get_fitness(df.copy(), [winBonus, k, par, days], True)
        #            ratingDf = copy.deepcopy(ratingDfMain)
        ##            print sorted(ratingDf.values(), key=lambda x: x[0], reverse=True)[:10]
        ##            print df['ratingA'].value_counts()
        #            if dc['error']<minError['error']:
        #                print '!!!!!!NEW MIN ERROR!!!!!', dc
        #                minError = dc.copy()
        #            if dc['err2']<minError1['err2']:
        #                minError1 = dc.copy()
        #                print '!!!!!!NEW MIN ERR2!!!!!', dc
        #            print 'ERROR', dc
        
        
        
        #print df.head()
        dc, df = get_fitness(df.copy(), [winBonus, k, par, days], True, inf = inf)
        print dc
        
        df['rdA'] = dc['sigma']
        df['rdB'] = dc['sigma']
        df['probA'] = np.round(df.apply(lambda x: win_probability(x), axis=1).values, 6)
        df['probB'] = 1 - df['probA'].values
        
        saveToFile(filename)
        
        db = client['csgo']
        for val in sorted(ratingDf.items(), key=lambda x: x[1][2], reverse=True)[:10]:
            print val, 
            rs = db['fullMatchHistory'].find_one({'$or': [{'teamBId': str(val[0])}, {'teamAId': str(val[0])}]})
            if rs:
                if rs['teamAId']==str(val[0]):
        #                        f.write('%s %s\n'%(rs['teamAUrlName'], str((val[1].mu, val[1].sigma)) ))
                    print rs['teamAUrlName']
                else:
        #                        f.write('%s %s\n'%(rs['teamBUrlName'], str((val[1].mu, val[1].sigma)) ))
                    print rs['teamBUrlName']
                    
        ratingDf = copy.deepcopy(ratingDfMain)
            
main_func()

client.close()
##print df['date'].min()
##print ratingDf[6665]
##print ratingDf[4608]
##print ratingDf[6667]
##print ratingDf[5973]
##print ratingDf[4494]
##print ratingDf[7532]

