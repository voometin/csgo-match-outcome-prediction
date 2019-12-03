# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 16:12:47 2018

@author: Andrew
"""

import pandas as pd
import numpy as np
import datetime as dt
import time
from tqdm import tqdm
import pymongo
import math, trueskill

def win_probability(raw):
    delta_mu = raw['ratingA'] - raw['ratingB']
    sum_sigma = raw['rdA']**2 + raw['rdB']**2
    denom = math.sqrt(sum_sigma)
    return trueskill.TrueSkill().cdf(delta_mu / denom)

client = pymongo.MongoClient(port=33333)
db = client['csgo']

def cCalc(t):#t - days - time period
    return ((350**2 - 130**2)/float(t))**0.5

def RDCalc(rdOld, c, t):
    return min((rdOld**2 + t * c**2)**0.5, 350)

def scale_down(rating, rd, ratio=173.7178):
    return (rating - 1500) / ratio, rd / ratio
    
def scaleUp(rating, rd, ratio=173.7178):
    return ratio*rating + 1500, ratio*rd


def gRDCalc(RD, mode=''):
#    print 'RD', RD, RD**2, (1 + 3*(q**2)*(RD**2)/(np.pi**2))**0.5, 1/float((1 + 3*(q**2)*(RD**2)/(np.pi**2))**0.5)
    if not mode:
        return 1/((1 + 3 * RD**2 / (np.pi**2))**0.5).astype(float)
    else:
        return 1/float((1 + 3*(RD**2)/(np.pi**2))**0.5)

def expScoreCalc(gRD, rA, rB, mode=''):
    if not mode:
        tmp = np.array(map(lambda z: 0 if z>8 else 1 if z<-8 else z,  -gRD*(rA - rB)))
        return 1/(1 + np.e**tmp).astype(float)
    else:
        tmp =  -gRD*(rA - rB)
        if tmp>8:
            tmp = 0
        elif tmp<-8:
            tmp = 1
        return 1/float(1 + np.e**tmp)

def expActScoreCalc(gRDList, expScoreList, resList):#, winBonusList):
#    return (gRDList * (winBonusList + resList - expScoreList) ).sum()
    return (gRDList * (resList - expScoreList) ).sum()

def deltaCalc(V, expActScore):
    return expActScore * V

def VCalc(gRDList, expScoreList):
    return 1/float( (gRDList**2 * expScoreList * (1 - expScoreList)).sum() )

def aCalc(sigma):
    return np.log(sigma**2)

def fxCalc(delta, rd, V, tau, a, x):
    firstFraction = (np.e**x * (delta**2 - rd**2 - V - np.e**x))/float(2 * (rd**2 + V + np.e**x)**2)
    secondFraction = (x - a)/float(tau**2)
    return firstFraction - secondFraction

def secondIter(delta, rd, V, tau, a, epsilon=0.000001):
    #2 Iteration
    A = a
    deltaSqr = delta**2
    rdSqr = rd**2
    if deltaSqr>rdSqr + V:
        B = np.log(deltaSqr - rdSqr - V)
    else:
        k = 1
        while fxCalc(delta, rd, V, tau, a, a - k*tau)<0:
            k += 1
            
        B = a - k*tau
    #3 Iteration
    fA = fxCalc(delta, rd, V, tau, a, A)
    fB = fxCalc(delta, rd, V, tau, a, B)
    #4 Iteration
    while abs(B - A)>epsilon:
        C = A + (A - B)*fA/float(fB - fA)
        fC = fxCalc(delta, rd, V, tau, a, C)
        if fC*fB<0:
            A = B
            fA = fB
        else:
            fA /= 2.0
        B = C
        fB = fC
    
#    print A
    #5 Iteration
    return np.e**(A/2.0)

def rdStarCalc(rd, sigma):
    return (rd**2 + sigma**2)**0.5

def rdUpdate(rdStar, V):
    return 1/float((rdStar**-2 + 1/float(V))**0.5)

def ratingChangeCalc(rOld, rd, expActScore):
    return rOld + expActScore * rd**2



#def rdRatingReturn(x, timeDlt, timeDlt2, c):
#    while ratingDfMain[x]:#True:
#        if ratingDfMain[x][0][2]<=timeDlt2:
#            if ratingDfMain[x][-1]!=ratingDfMain[x][0]:
#                if ratingDfMain[x][1][2]<=timeDlt:
#                    ratingDfMain[x].pop(0)
#                elif ratingDfMain[x][1][2]<=timeDlt2:
#                    ratingDfMain[x][0] = [1500, 350, timeDlt]
#                    break
#                else:
#                    break
#            else:
#                ratingDfMain[x][0] = [1500, 350, timeDlt]
#                break
#                    
#        elif ratingDfMain[x][0][2]<=timeDlt:
#            if ratingDfMain[x][-1]!=ratingDfMain[x][0]:
#                if ratingDfMain[x][1][2]<=timeDlt:
#                    ratingDfMain[x].pop(0)
#                else:
#                    break
#            else:
#                break
#            
#        else:
#            if ratingDfMain[x][0][:2] != [1500, 350]:
#                ratingDfMain[x] = [[1500, 350, timeDlt]] + ratingDfMain[x]
#            break
#        
##    print ratingDfMain[x]
#    rd = RDCalc(ratingDfMain[x][0][1], c)
#    rating = ratingDfMain[x][0][0]
#    
#    return rd, rating

def ratingCalculation(lcldf, unratePer, tau, ratePer):
#    global ratingDfMain
    c = cCalc(unratePer/float(ratePer))
        
        
    lstAErr, lstBErr = [], []
    acc = []
#    ttlSec = float(t * 24 * 3600.0)
    tmdelta = dt.timedelta(days=ratePer)#период за который смотреть историю игр
    unratedTDelta = dt.timedelta(days=unratePer)#период за который команда считается unrated
    for _, x in lcldf.iterrows():#.iloc[:100]
        pass
        timeDlt = x['date'] - tmdelta
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
                        
        
#        lclTtlA = lcldf[((lcldf['teamAId']==x['teamAId']) | (lcldf['teamBId']==x['teamAId'])) & (lcldf['date']>=stDate) & (lcldf['date']<x['date'])].copy()
#        lclTtlB = lcldf[((lcldf['teamAId']==x['teamBId']) | (lcldf['teamBId']==x['teamBId'])) & (lcldf['date']>=stDate) & (lcldf['date']<x['date'])].copy() 
        lclTtlA = lcldf.iloc[:_][((lcldf.iloc[:_]['teamAId']==x['teamAId']) | (lcldf.iloc[:_]['teamBId']==x['teamAId'])) & (lcldf.iloc[:_]['date']>=stDate) & (lcldf.iloc[:_]['date']<x['date']) & (~np.isnan(lcldf.iloc[:_]['difScore'].values))].copy()
        lclTtlB = lcldf.iloc[:_][((lcldf.iloc[:_]['teamAId']==x['teamBId']) | (lcldf.iloc[:_]['teamBId']==x['teamBId'])) & (lcldf.iloc[:_]['date']>=stDate) & (lcldf.iloc[:_]['date']<x['date']) & (~np.isnan(lcldf.iloc[:_]['difScore'].values))].copy() 
        lclA = lclTtlA[lclTtlA['date']>=timeDlt].copy()
        lclB = lclTtlB[lclTtlB['date']>=timeDlt].copy()
        
        
        if lclTtlA.shape[0]==0:
            newRDA = 350.0
            newRatingA = 1500.0
            newSigmaA = 0.06
        elif lclA.shape[0]==0:
            lclTtlA = lclTtlA.iloc[-1]
            if lclTtlA['teamAId']==x['teamAId']:
                newRatingA = lclTtlA['ratingA']
                newSigmaA = lclTtlA['sigmaA']
                newRDA = lclTtlA['rdA']
                
                lcltm = (x['date'] - lclTtlA['date']).days/float(ratePer)
                newRDA = RDCalc(lclTtlA['rdA'], c, lcltm)
                
                preRDA = scale_down(newRatingA, newRDA)[1]
                rdStarA = rdStarCalc(preRDA, newSigmaA)
                newRDA = scaleUp(newRatingA, rdStarA)[1]
            else:
                newRatingA = lclTtlA['ratingB']
                newSigmaA = lclTtlA['sigmaB']
                
                lcltm = (x['date'] - lclTtlA['date']).days/float(ratePer)
                newRDA = RDCalc(lclTtlA['rdB'], c, lcltm)
                
                preRDA = scale_down(newRatingA, newRDA)[1]
                rdStarA = rdStarCalc(preRDA, newSigmaA)
                newRDA = scaleUp(newRatingA, rdStarA)[1]
        else:
#            print lclA['date']
            exchInd = lclA[lclA['teamBId']==x['teamAId']].index
            lclA.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB', 'sigmaA', 'sigmaB']] = lclA.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA', 'sigmaB', 'sigmaA']].values
#            lclA.loc[exchInd, 'difScore'] = lclA.loc[exchInd, 'difScore'].replace({0: 1, 1: 0})
            lclA.loc[exchInd, 'difScore'] = (1 - lclA.loc[exchInd, 'difScore']).values
#            ratingA = lclA.iloc[-1, 'ratingA']
            lastRow = lclA.iloc[0]
#            ratingA = lastRow['ratingA']
            preRatingA, preRDA = scale_down(lastRow['ratingA'], lastRow['rdA'])
            sigmaA = lastRow['sigmaA']
            a = aCalc(sigmaA)
            lclARes = lclA['difScore'].values
#            winBonusList = np.array(map(lambda x: winBonus if x>0.5 else 0, lclARes))
            
            preRatingListA, preRDListA = scale_down(lclA['ratingA'].values, lclA['rdA'].values)
            preRatingListAOpp, preRDListAOpp = scale_down(lclA['ratingB'].values, lclA['rdB'].values)
            
            gRDListAOpp = gRDCalc(preRDListAOpp)
            expScoreAList = expScoreCalc(gRDListAOpp, preRatingListA, preRatingListAOpp)
            
            V = VCalc(gRDListAOpp, expScoreAList)
            expActScore = expActScoreCalc(gRDListAOpp, expScoreAList, lclARes)#, winBonusList)
            delta = deltaCalc(V, expActScore)
            
            newSigmaA = secondIter(delta, preRDA, V, tau, a)
            
            rdStarA = rdStarCalc(preRDA, newSigmaA)
            
            preNewRDA = rdUpdate(rdStarA, V)
            preNewRatingA = ratingChangeCalc(preRatingA, preNewRDA, expActScore)
            
            newRatingA, newRDA = scaleUp(preNewRatingA, preNewRDA)

        lcldf.loc[_, 'ratingA'] = newRatingA
        lcldf.loc[_, 'rdA'] = newRDA
        lcldf.loc[_, 'sigmaA'] = newSigmaA
            
#            print 'YEAH'
        
        if lclTtlB.shape[0]==0:
            newRDB = 350.0
            newRatingB = 1500.0
            newSigmaB = 0.06
        elif lclB.shape[0]==0:
            lclTtlB = lclTtlB.iloc[-1]
            if lclTtlB['teamBId']==x['teamBId']:
                newRatingB = lclTtlB['ratingB']
                newSigmaB = lclTtlB['sigmaB']
                newRDB = lclTtlB['rdB']
                
                lcltm = (x['date'] - lclTtlB['date']).days/float(ratePer)
                newRDB = RDCalc(lclTtlB['rdB'], c, lcltm)
                
                preRDB = scale_down(newRatingB, newRDB)[1]
                rdStarB = rdStarCalc(preRDB, newSigmaB)
                newRDB = scaleUp(newRatingB, rdStarB)[1]
            else:
                newRatingB = lclTtlB['ratingA']
                newSigmaB = lclTtlB['sigmaA']
                
                lcltm = (x['date'] - lclTtlB['date']).days/float(ratePer)
                newRDB = RDCalc(lclTtlB['rdA'], c, lcltm)
                
                preRDB = scale_down(newRatingB, newRDB)[1]
                rdStarB = rdStarCalc(preRDB, newSigmaB)
                newRDB = scaleUp(newRatingB, rdStarB)[1]
        else:
#            print lclB['date']
            exchInd = lclB[lclB['teamAId']==x['teamBId']].index
            lclB.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB', 'sigmaA', 'sigmaB']] = lclB.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA', 'sigmaB', 'sigmaA']].values
#            lclB.loc[exchInd, 'difScore'] = lclB.loc[exchInd, 'difScore'].replace({0: 1, 1: 0})
            lclB.loc[exchInd, 'difScore'] = (1 - lclB.loc[exchInd, 'difScore']).values
            lastRow = lclB.iloc[0]
#            ratingA = lastRow['ratingA']
            preRatingB, preRDB = scale_down(lastRow['ratingB'], lastRow['rdB'])
            sigmaB = lastRow['sigmaB']
            a = aCalc(sigmaB)
            lclBRes = (1 - lclB['difScore']).values
#            winBonusList = np.array(map(lambda x: winBonus if x>0.5 else 0, lclBRes))
            
            preRatingListB, preRDListB = scale_down(lclB['ratingB'].values, lclB['rdB'].values)
            preRatingListBOpp, preRDListBOpp = scale_down(lclB['ratingA'].values, lclB['rdA'].values)
            
            gRDListBOpp = gRDCalc(preRDListBOpp)
            expScoreBList = expScoreCalc(gRDListBOpp, preRatingListB, preRatingListBOpp)
            
            V = VCalc(gRDListBOpp, expScoreBList)
            expActScore = expActScoreCalc(gRDListBOpp, expScoreBList, lclBRes)#, winBonusList)
            delta = deltaCalc(V, expActScore)
            
            newSigmaB = secondIter(delta, preRDB, V, tau, a)
            
            rdStarB = rdStarCalc(preRDB, newSigmaB)
            
            preNewRDB = rdUpdate(rdStarB, V)
            preNewRatingB = ratingChangeCalc(preRatingB, preNewRDB, expActScore)
            
            newRatingB, newRDB = scaleUp(preNewRatingB, preNewRDB)

        lcldf.loc[_, 'ratingB'] = newRatingB
        lcldf.loc[_, 'rdB'] = newRDB
        lcldf.loc[_, 'sigmaB'] = newSigmaB
            
        if not np.isnan(x['difScore']):#x['difScore']!=np.nan:
            expScore = expScoreCalc(gRDCalc(newRDB, True), newRatingA, newRatingB, True)
            err = abs(x['difScore'] - expScore)
            
            if expScore>=x['difScore'] and x['difScore']>0.5 or expScore<=x['difScore'] and x['difScore']<0.5:
                err = 0
                
            if (expScore>0.5 and x['difScore']>0.5) or (expScore<0.5 and x['difScore']<0.5):
#                lstAErr += [[err, err/2.0]]
                acc.append(True)
            else:
#                lstAErr += [[err, err]]
                acc.append(False)
                
            lstAErr.append(err)
                
#            expScore = expScoreCalc(gRDCalc(newRDA, True), newRatingB, newRatingA, True)
#            err = abs(1 - x['difScore'] - expScore)
#            
#            if (expScore>0.5 and 1 - x['difScore']>0.5) or (expScore<0.5 and 1 - x['difScore']<0.5):
#                lstBErr += [[err, err/2.0]]
#            else:
#                lstBErr += [[err, err]]
#        else:
#            lstAErr.append(0)
#            lstAErr += [[0, 0]]
#            lstBErr += [[0, 0]]
            
        if _%10000==0:
            print _
#        lstBErr += [-x['difScore'] - (32 * expScore - 16)]

    return {'acc': round(sum(acc)/float(len(acc)), 5), 'error': sum(lstAErr)}, lstAErr
#    return -sum(lstAErr)
            
#    print lcldf.describe()
#    with open('glicko2.txt', 'a') as f:
##        f.write('winBonus, unratePer, tau, ratePer: %s\n'%str([winBonus, unratePer, tau, ratePer]))
#        f.write('unratePer, tau, ratePer: %s\n'%str([unratePer, tau, ratePer]))
#        f.write(str(lcldf.describe())+'\n')
#    print (lcldf['difScore']==0.5).sum()/float(lcldf.shape[0])
#    print 'shape before', lcldf.shape[0]a
#    lcldf.drop(lcldf[lcldf['difScore']==0.5].index, inplace=True)



#!!!!!!!!!
#    cond = lcldf['date']>dt.date(2015, 12, 31)
##    print 'shape after', lcldf.shape[0]
#    lcldf['pred'] = lcldf['ratingA'] > lcldf['ratingB']
#    lcldf['pred1'] = lcldf['difScore'] > 0.5
##    with open('glicko2.txt', 'a') as f:
##        f.write('accuracy: %s\n'%str((lcldf[cond]['pred'] == lcldf[cond]['pred1']).sum()/float(lcldf[cond].shape[0])))
##        f.write('abs error A: %s\n'%str(map(sum, zip(*lstAErr))))
##        f.write('abs error B: %s\n'%str(map(sum, zip(*lstBErr))))
##        f.write('abs error A MEAN: %s\n'%str(map(np.mean, zip(*lstAErr))))
##        f.write('abs error B MEAN: %s\n'%str(map(np.mean, zip(*lstBErr))))
#    print 'accuracy', (lcldf[cond]['pred'] == lcldf[cond]['pred1']).sum()/float(lcldf[cond].shape[0])
#    print 'abs error A', map(sum, zip(*lstAErr))
#    print 'abs error B', map(sum, zip(*lstBErr))
#    print 'abs error A MEAN', map(np.mean, zip(*lstAErr))
#    print 'abs error B MEAN', map(np.mean, zip(*lstBErr))
#!!!!!!!!!    
    
    
#    lcldf.drop(lcldf[lcldf['difScore']==0.5].index, inplace=True)
#            print 'YEAH'
#        return
        
        
        
        
            
        
#def ratingDistribution(filename, per):
#    global df
#    print 'START ratingDistribution'
#    td = dt.timedelta(days=per)
#    
#    lcldf = pd.read_csv(filename, sep=';')
#    lcldf['date'] = pd.to_datetime(lcldf['date'])
#    for ind, row in lcldf.iterrows():
#        tmpIndex = df[(df['teamAId']==row['teamId']) & (df['date']>=row['date']) & (df['date']<row['date']+td)].index
#        df.loc[tmpIndex, 'ratingA'] = row['rating']
#        df.loc[tmpIndex, 'rdA'] = row['rd']
#        if tmpIndex.shape[0]!=row['gamesQnt']:
#            tmpIndex = df[(df['teamBId']==row['teamId']) & (df['date']>=row['date']) & (df['date']<row['date']+td)].index
#            df.loc[tmpIndex, 'ratingB'] = row['rating']
#            df.loc[tmpIndex, 'rdB'] = row['rd']
#            
#    
#    df['pred'] = df['ratingA'] > df['ratingB']
#    df['pred1'] = df['difScore'] > 0
#    print (df['pred'] == df['pred1']).sum()/float(df.shape[0])
#    print (df['difScore']==0.5).sum()/float(df.shape[0])
    
#    print lcldf.info()
#    print lcldf.head()
def saveToFile(filename):
    global df
    with open(filename, 'w') as f:
        df[['matchlinkId', 'teamAId', 'teamBId', 'map', 'ratingA', 'ratingB','rdA', 'rdB', 'sigmaA', 'sigmaB', 'probA', 'probB']].to_csv(f, sep=';', index=False)
    
client = pymongo.MongoClient(port=33333)  
db = client['csgo']
    

def bayes_func(ratePer):#, days):#, par=400, inf=True):
    global ratingDfMain, ratingDf
    df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
    df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    #                df = df.sort_values('date').reset_index(drop=True)
    #df['difScore'] = df['difScore'].apply(lambda x: 0.5 if x==0 else 0 if x<0 else 1)
#    df['difScore'] = df['difScore']/32.0 + 0.5
    df.loc[~df.isnull().any(axis=1), 'difScore'] = df.loc[~df.isnull().any(axis=1), 'difScore']/32.0 + 0.5
    
    df['ratingA'] = 1500
    df['ratingB'] = 1500
    df['rdA'] = 350
    df['rdB'] = 350
    df['sigmaA'] = 0.06
    df['sigmaB'] = 0.06
    
    #print df.info()
#    params = 7
    #                params = [df, winBonus, params*unratePer, tau, params*ratePer]
#    unratePer = 112
    #ratePer = 14
    params = [df, 110, 0.3, int(ratePer)]
    
#    teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
#    maxdate = df['date'].max()
    print params[1:]


    print dt.datetime.today()
    tm = time.time()
    dc = ratingCalculation(*params)[0]
    print time.time() - tm
    print dc
    
        
    
    #{'error': error, 'err2': error1, 'acc': acc, 'coeff': kefs, 'sigma': sigma}, lcldf
#    return dc['acc']
    return -dc['error']


#from bayes_opt import BayesianOptimization
##print bayes_func(0, 1, 400, 110)
#
##winBonus, k, par, days, inf
##'unratePer': (70, 140), 
#bo = BayesianOptimization(bayes_func, {'ratePer': (7, 28)})#, 'days': (100, 200)})
#KAPPA = 5
##gp_params = {'kernel': 'cubic'}
#bo.maximize(init_points=4, n_iter=10, acq='ucb', kappa=KAPPA)#, **gp_params)
#print bo.res['max']

#{'max_params': {'ratePer': 7.00212003065896, 'unratePer': 111.00733190810075},
# 'max_val': 0.55837}

#
#for unratePer in np.arange(16, 21, 4):
##    for winBonus in np.arange(0, 0.26, 0.05):
#        for ratePer in np.arange(2, 18, 4):
#            for tau in [0.3, 0.5, 0.7]:#, 0.9, 1.1]:
#                if ratePer>unratePer:
#                    continue

def main_func():
    global ratingDfMain, df
#    for tau in [0.2, 0.3, 0.4, 0.8]:
#    print 'tau', tau
#    for ratePer, unratePer in [(7, 111), (14, 111)]:#14, 21, 28, 36]:
    ratePer, unratePer = 14, 111
    df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
    df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    #                df = df.sort_values('date').reset_index(drop=True)
    #df['difScore'] = df['difScore'].apply(lambda x: 0.5 if x==0 else 0 if x<0 else 1)
#    df['difScore'] = df['difScore']/32.0 + 0.5
    df.loc[~df.isnull().any(axis=1), 'difScore'] = df.loc[~df.isnull().any(axis=1), 'difScore']/32.0 + 0.5
    
    df['ratingA'] = 1500
    df['ratingB'] = 1500
    df['rdA'] = 350
    df['rdB'] = 350
    df['sigmaA'] = 0.06
    df['sigmaB'] = 0.06
    
    #print df.info()
#    params = 7
    #                params = [df, winBonus, params*unratePer, tau, params*ratePer]
#        unratePer = 112
    #ratePer = 14
    params = [df, unratePer, 0.3, ratePer]
    
#        teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
    maxdate = df['date'].max()
    print params[1:]


    print dt.datetime.today()
    tm = time.time()
    eror = ratingCalculation(*params)[1]
    print time.time() - tm
    print
    
    df['probA'] = np.round(df.apply(lambda x: win_probability(x), axis=1).values, 6)
    df['probB'] = 1 - df['probA'].values
    
    print df[['ratingA', 'ratingB', 'probA', 'rdA', 'rdB']].describe()
    print (((df['probA']>0.5) & (df['difScore']>0.5)) | ((df['probA']<0.5) & (df['difScore']<0.5))).value_counts(normalize=True)
    print pd.Series(eror).describe()
    filename = 'glicko2.csv'
    saveToFile(filename)
        
    #c = cCalc(unratePer)
    #ttlSec = float(unratePer * 24 * 3600.0)
    stDate = maxdate - dt.timedelta(days=unratePer)
    lclTtl = np.unique(np.append(df[df['date']>=stDate]['teamAId'].value_counts().index.values, df[df['date']>=stDate]['teamBId'].value_counts().index.values))
    tms = []
    for ids in lclTtl:
        lcl = df[((df['teamAId']==ids) | (df['teamBId']==ids)) & (df['date']>=stDate)].iloc[-1]
        if lcl['teamAId']==ids:
    #        lcltm = (maxdate - lcl['date']).total_seconds()/ttlSec * unratePer
    #        lcl['rdA'] = RDCalc(lcl['rdA'], c, lcltm)
            tms += [ [lcl['teamAId'], lcl['ratingA'], lcl['rdA']] ]
        elif lcl['teamBId']==ids:
    #        lcltm = (maxdate - lcl['date']).total_seconds()/ttlSec * unratePer
    #        lcl['rdB'] = RDCalc(lcl['rdB'], c, lcltm)
            tms += [ [lcl['teamBId'], lcl['ratingB'], lcl['rdB']] ]
            
            
    
    for val in sorted(tms, key=lambda x: (x[1], -x[2]), reverse=True)[:10]:
        print val, 
        rs = db['fullMatchHistory'].find_one({'$or': [{'teamBId': str(val[0])}, {'teamAId': str(val[0])}]})
        if rs:
            if rs['teamAId']==str(val[0]):
    #                        f.write('%s %s\n'%(rs['teamAUrlName'], str((val[1].mu, val[1].sigma)) ))
                print rs['teamAUrlName']
            else:
    #                        f.write('%s %s\n'%(rs['teamBUrlName'], str((val[1].mu, val[1].sigma)) ))
                print rs['teamBUrlName']
                
    print 'ASTRALIS', [i for i in tms if i[0]==6665]


#main_func()
#tau = 0.5
#lst = [[1500.0, 1400.0, 200.0, 30.0, 0.06, 1],
#       [1500.0, 1550.0, 200.0, 100.0, 0.06, 0],
#       [1500.0, 1700.0, 200.0, 300.0, 0.06, 0],
#       ]
#
#clmns = ['ratingA', 'ratingB', 'rdA', 'rdB', 'sigmaA', 'difScore']
#lclA = pd.DataFrame(lst, columns = clmns).head()
#
#lastRow = lclA.iloc[-1]
#preRatingA, preRDA = scale_down(lastRow['ratingA'], lastRow['rdA'])
#sigmaA = lastRow['sigmaA']
#a = aCalc(sigmaA)
#lclARes = lclA['difScore'].values
#
#
#preRatingListA, preRDListA = scale_down(lclA['ratingA'].values, lclA['rdA'].values)
#preRatingListAOpp, preRDListAOpp = scale_down(lclA['ratingB'].values, lclA['rdB'].values)
#
#gRDListAOpp = gRDCalc(preRDListAOpp)
#expScoreAList = expScoreCalc(gRDListAOpp, preRatingListA, preRatingListAOpp)
#
#V = VCalc(gRDListAOpp, expScoreAList)
#expActScore = expActScoreCalc(gRDListAOpp, expScoreAList, lclARes)
#delta = deltaCalc(V, expActScore)
#
#newSigma = secondIter(delta, preRDA, V, tau, a)
#
#rdStarA = rdStarCalc(preRDA, newSigma)
#
#preNewRDA = rdUpdate(rdStarA, V)
#preNewRatingA = ratingChangeCalc(preRatingA, preNewRDA, expActScore)
#
#newRatingA, newRDA = scaleUp(preNewRatingA, preNewRDA)



#tm = time.time()
#ratingDistribution('../cs_new/glicko.csv', 7)
#print time.time() - tm


#print df.iloc[:1000][(df['teamAId']==6385) | (df['teamBId']==6385)]
#for val in sorted(ratingDfMain.items(), key=lambda x: x[1][-1][0], reverse=True)[:10]:
#    print val[0], val[1][-1]
#print np.dot(np.array([2, 3]), np.array([5, 6]), np.array([2, 3]))
#print np.linalg.multi_dot([np.array([2, 3]), np.array([5, 6]), np.array([2, 3])])

client.close()