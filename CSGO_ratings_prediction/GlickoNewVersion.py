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
import trueskill, math

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']

def win_probability(raw):
    delta_mu = raw['ratingA'] - raw['ratingB']
    sum_sigma = raw['rdA']**2 + raw['rdB']**2
    denom = math.sqrt(sum_sigma)
    return trueskill.TrueSkill().cdf(delta_mu / denom)

def cCalc(t):#t - days - time period
    return ((350**2 - 130**2)/float(t))**0.5

def RDCalc(rdOld, c, t):
    return min((rdOld**2 + t * c**2)**0.5, 350)

def qCalc(par):
    return np.log(10)/float(par)

def gRDCalc(q, RD, mode=''):
#    print 'RD', RD, RD**2, (1 + 3*(q**2)*(RD**2)/(np.pi**2))**0.5, 1/float((1 + 3*(q**2)*(RD**2)/(np.pi**2))**0.5)
    if not mode:
        return 1/((1 + 3*(q**2)*(RD**2)/(np.pi**2))**0.5).astype(float)
    else:
        return 1/float((1 + 3*(q**2)*(RD**2)/(np.pi**2))**0.5)

def expScoreCalc(gRD, rA, rB, par, mode=''):
    if not mode:
        tmp = np.array(map(lambda z: 0 if z>10 else 1 if z<-10 else z,  -gRD*(rA - rB)/float(par)))
        return 1/(1 + 10**tmp).astype(float)
    else:
        tmp =  -gRD*(rA - rB)/float(par)
        return 1/float(1 + 10**tmp)
#    return 1/(1 + 10**(-gRD*(rA - rB)/float(par))).astype(float)


def dSqrCalc(q, gRDList, expScoreList):
#    print gRDList
#    print expScoreList
    return 1/(q**2 * (gRDList**2 * expScoreList * (1 - expScoreList)).sum()).astype(float)

def rdSqrDSqrCalc(RD, dSqr):
#    print float(1/(RD**2) + 1/dSqr)
    return max(.000000000001, float(1/(RD**2) + 1/dSqr))

#def ratingChangeCalc(rOld, q, rdSqrDSqr, gRDList, expScoreList, resList, winBonusList):
#    return rOld + (gRDList * (winBonusList + resList - expScoreList) ).sum() * q/rdSqrDSqr

def ratingChangeCalc(rOld, q, rdSqrDSqr, gRDList, expScoreList, resList):
    return rOld + (gRDList * (resList - expScoreList) ).sum() * q/rdSqrDSqr

def RDChangeCalc(rdSqrDSqr):
    return (1/rdSqrDSqr)**0.5

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

def ratingCalculation(lcldf, unratePer, par, ratePer, unchange):
#    global ratingDfMain
    c = cCalc(unratePer/float(ratePer))
    q = qCalc(par)
#    lst = []
#    clmns = ['date', 'teamId', 'rating', 'rd', 'gamesQnt']#еще дифьюзы и тд.
#    
#    tmpdf = pd.DataFrame(columns=clmns)
#    
#    with open('../cs_new/glicko.csv', 'w') as f:
#        tmpdf.to_csv(f, sep=';', index=False)
#        
#    td = dt.timedelta(days=7)
#    tmdelta = dt.timedelta(days=timePeriod)
##    flag = False
##    lenKeys = len(ratingDfMain)
#    rKeys = sorted(ratingDfMain.keys())
##    startDate = dt.datetime.strptime(rKeys[rInd], '%Y-%m-%d')
#    for ind, date in enumerate(rKeys[:-1]):
##    for ind, date in enumerate(rKeys[:11]):
#        print date
#        ratingDfMain[date] = {}
#        dateDif = date-tmdelta
#        tmpDf = lcldf[(lcldf['date']<date) & (lcldf['date']>dateDif)].copy()
#        if not tmpDf.shape[0]:
#            continue
#        teamIds = np.unique(np.append(tmpDf['teamAId'].value_counts().index.values, tmpDf['teamBId'].value_counts().index.values))
#        for ids in teamIds:
##            preRate = ratingDfMain.get(dateDif, '')
##            if not preRate:
##                rd = 350.0
##                rating = 1500.0
##            else:
##                teamPreRate = preRate.get(ids, [])
##                if teamPreRate:
###                    print teamPreRate, ids
##                    rd = RDCalc(teamPreRate[1], c)
##                    rating = teamPreRate[0]
##                else:
##                    rd = 350.0
##                    rating = 1500.0
#                
#            tmp = tmpDf[(tmpDf['teamAId']==ids) | (tmpDf['teamBId']==ids)].copy()
#            tmpshape = tmp.shape[0]
#            
##            tmpA = tmp[tmpDf['teamAId']==ids].index
##            tmpB = tmp[tmpDf['teamBId']==ids].index
##            lcldf.loc[tmpA, 'ratingA'] = rating
##            lcldf.loc[tmpA, 'rdA'] = rd
##            lcldf.loc[tmpB, 'ratingB'] = rating
##            lcldf.loc[tmpB, 'rdB'] = rd
#            
#            exchInd = tmp[tmp['teamBId']==ids].index
##            print lclA.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA']]
#            tmp.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB']] = tmp.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA']].values
##            print lclA.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB']]
#            tmp.loc[exchInd, 'difScore'] = tmp.loc[exchInd, 'difScore'].replace({0: 1, 1: 0})
##            lclA.loc[exchInd, 'difScore'] = 1 - lclA.loc[exchInd, 'difScore']
##            print 'AFTER EXCHANG'#, exchInd
##            print lclA.head()
#            rating = tmp['ratingA'].values
#            ratingOpp = tmp['ratingB'].values#ratings of opponents
#            rd = tmp['rdA'].values
#            rdOpp = tmp['rdB'].values#ratings of opponents
##            if not preRate:
##                ratingListOpp = np.array([1500.0]*tmpshape)
##                gRDListOpp = np.array([350.0]*tmpshape)#RD of opponents
##            else:
##                ratingListOpp = []
##                gRDListOpp = []
##                for i in tmp['teamBId'].values:
##                    oppTmp = preRate.get(i, [])
##                    if oppTmp:
###                        print 'wokr'
##                        ratingListOpp += [oppTmp[0]]
##                        gRDListOpp += [RDCalc(oppTmp[1], c)]
##                    else:
##                        ratingListOpp += [1500.0]
##                        gRDListOpp += [350.0]
##                ratingListOpp = np.array(ratingListOpp)
##                gRDListOpp = np.array(gRDListOpp)
#                
#            gRDListOpp = gRDCalc(q, gRDListOpp)
#                
#                        
#            #ratings of opponents
##            print ratingListA, ratingListAOpp
##            gRDListOpp = gRDCalc(q, np.vectorize(lambda z: RDCalc(z, c))(tmp['rdB'].values))#RD of opponents
##            print 'gRDListAOpp', lclA['rdB'].values, gRDListAOpp
#            expScoreList = expScoreCalc(gRDListOpp, rating, ratingListOpp, par)
##            print 'expScoreAList', expScoreAList, ratingListA, ratingListAOpp
#            lclRes = tmp['difScore'].values
##            print 'lclARes', lclARes
#            dSqr = dSqrCalc(q, gRDListOpp, expScoreList)
##            print 'dSqrA', dSqrA
#            rdSqrDSqr = rdSqrDSqrCalc(rd, dSqr)
##            print 'rdSqrDSqrA', rdSqrDSqrA
#            newRD = RDChangeCalc(rdSqrDSqr)
##            print 'newRDA', newRDA, rdA
#            newRating = ratingChangeCalc(rating, q, rdSqrDSqr, gRDListOpp, expScoreList, lclRes)
##            print 'newRatingA', newRatingA, ratingA
##            print 'A', newRatingA, newRDA, _
#            ratingDfMain[date][ids] = [newRating, newRD, tmpshape]
#            
#            tmpIndex = df[(df['teamAId']==ids) & (df['date']>=date) & (df['date']<date+td)].index
#            df.loc[tmpIndex, 'ratingA'] = newRating
#            df.loc[tmpIndex, 'rdA'] = newRD
#            if tmpIndex.shape[0]!=tmpshape:
#                tmpIndex = df[(df['teamBId']==ids) & (df['date']>=date) & (df['date']<date+td)].index
#                df.loc[tmpIndex, 'ratingB'] = newRating
#                df.loc[tmpIndex, 'rdB'] = newRD
#            
#            lst += [[date, ids, newRating, newRD, tmpshape]]
#            
#        if len(lst)>100:
#            tmpdf = pd.DataFrame(lst, columns=clmns)
#            with open('../cs_new/glicko.csv', 'a') as f:
#                tmpdf.to_csv(f, sep=';', index=False, header=False)
#                
#            lst = []
#            
#    if lst:
#        tmpdf = pd.DataFrame(lst, columns=clmns)
#        with open('../cs_new/glicko.csv', 'a') as f:
#            tmpdf.to_csv(f, sep=';', index=False, header=False)
#            
#        lst = []
#        
##    ratingDistribution('../cs_new/glicko.csv', 7)
#            
##            lcldf.loc[_, 'ratingA'] = newRatingA
##            lcldf.loc[_, 'rdA'] = newRDA
        
        
#    print ind, rKeys[ind], sorted(ratingDfMain[rKeys[ind]].values(), key=lambda x: x[0], reverse=True)[:10]
#    print ind, rKeys[ind-1], sorted(ratingDfMain[rKeys[ind-1]].values(), key=lambda x: x[0], reverse=True)[:10]
    lstAErr, lstBErr = [], []
#    ttlSec = float(unratePer * 24 * 3600.0)
    unratedTDelta = dt.timedelta(days=unratePer)#период за который команда считается unrated
    tmdelta = dt.timedelta(days=ratePer)#период за который смотреть историю игр
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
                        
        
        lclTtlA = lcldf.iloc[:_][((lcldf.iloc[:_]['teamAId']==x['teamAId']) | (lcldf.iloc[:_]['teamBId']==x['teamAId'])) & (lcldf.iloc[:_]['date']>=stDate) & (lcldf.iloc[:_]['date']<x['date']) & (~np.isnan(lcldf.iloc[:_]['difScore'].values))].copy()
        lclTtlB = lcldf.iloc[:_][((lcldf.iloc[:_]['teamAId']==x['teamBId']) | (lcldf.iloc[:_]['teamBId']==x['teamBId'])) & (lcldf.iloc[:_]['date']>=stDate) & (lcldf.iloc[:_]['date']<x['date']) & (~np.isnan(lcldf.iloc[:_]['difScore'].values))].copy() 
        lclA = lclTtlA[lclTtlA['date']>=timeDlt].copy()
        lclB = lclTtlB[lclTtlB['date']>=timeDlt].copy()
        
        
        if lclTtlA.shape[0]==0:
            newRDA = 350.0
            newRatingA = 1500.0
        elif lclA.shape[0]==0:
            lclTtlA = lclTtlA.iloc[0]
            if lclTtlA['teamAId']==x['teamAId']:
                newRatingA = lclTtlA['ratingA']
                if unchange:
                    newRDA = lclTtlA['rdA']
                else:
                    lcltm = (x['date'] - lclTtlA['date']).days/float(ratePer)
                    newRDA = RDCalc(lclTtlA['rdA'], c, lcltm)
            else:
                newRatingA = lclTtlA['ratingB']
                if unchange:
                    newRDA = lclTtlA['rdB']
                else:
                    lcltm = (x['date'] - lclTtlA['date']).days/float(ratePer)
                    newRDA = RDCalc(lclTtlA['rdB'], c, lcltm)
        else:
#            print 'teamAId', x['teamAId']
#            print lclA.head()
            exchInd = lclA[lclA['teamBId']==x['teamAId']].index
#            print lclA.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA']]
            lclA.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB']] = lclA.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA']].values
#            print lclA.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB']]
#            lclA.loc[exchInd, 'difScore'] = lclA.loc[exchInd, 'difScore'].replace({0: 1, 1: 0})
            lclA.loc[exchInd, 'difScore'] = (1 - lclA.loc[exchInd, 'difScore']).values
#            print 'AFTER EXCHANG'#, exchInd
#            print lclA.head()
#            ratingA = lclA.iloc[-1, 'ratingA']
            lastRow = lclA.iloc[0]
#            if lclA['date'].value_counts().shape[0]>1:
#                print 'last', lclA.iloc[-1]
#                print 'first', lclA.iloc[0]
#            print '======='
#            print lclA.iloc[0]
#            print lclA.iloc[-1]
#            print '======='
            ratingA = lastRow['ratingA']
#            if (x['date'] - lastRow['date']).total_seconds()/ttlSec * t>1:
#                print (x['date'] - lastRow['date']).total_seconds()/ttlSec * t, (x['date'] - lastRow['date']).total_seconds(), ttlSec, t, x['date'], lastRow['date']
            if unchange:
                rdA = lastRow['rdA']
            else:
                lcltm = (x['date'] - lastRow['date']).days/float(ratePer)
                rdA = RDCalc(lastRow['rdA'], c, lcltm)
#            timeKefs = map(lambda z: (x['date'] - z).total_seconds()/ttlSec * t, lclA['date'])
            ratingListA = lclA['ratingA'].values
            ratingListAOpp = lclA['ratingB'].values#ratings of opponents
#            gRDListAOpp = gRDCalc(q, np.vectorize(lambda z: RDCalc(z, c))(lclA['rdB'].values))#RD of opponents
            gRDListAOpp = gRDCalc(q, lclA['rdB'].values)#RD of opponents
#            experiment = gRDCalc(q, (lclB['rdB'].values**2 + lclB['rdA'].values**2)**0.5)
#            expScoreAList = expScoreCalc(experiment, ratingListA, ratingListAOpp, par)
            expScoreAList = expScoreCalc(gRDListAOpp, ratingListA, ratingListAOpp, par)
            lclARes = lclA['difScore'].values
#            winBonusList = np.array(map(lambda x: winBonus if x>0.5 else 0, lclARes))
            dSqrA = dSqrCalc(q, gRDListAOpp, expScoreAList)
            rdSqrDSqrA = rdSqrDSqrCalc(rdA, dSqrA)
            newRDA = min(RDChangeCalc(rdSqrDSqrA), 350)
            newRatingA = max(-1500.0, min(ratingChangeCalc(ratingA, q, rdSqrDSqrA, gRDListAOpp, expScoreAList, lclARes), 4500.0))

        lcldf.loc[_, 'ratingA'] = newRatingA
        lcldf.loc[_, 'rdA'] = newRDA
            
#            print 'YEAH'
        
        if lclTtlB.shape[0]==0:
            newRDB = 350.0
            newRatingB = 1500.0
        elif lclB.shape[0]==0:
            lclTtlB = lclTtlB.iloc[0]
            if lclTtlB['teamBId']==x['teamBId']:
                newRatingB = lclTtlB['ratingB']
                if unchange:
                    newRDB = lclTtlB['rdB']
                else:
                    lcltm = (x['date'] - lclTtlB['date']).days/float(ratePer)
                    newRDB = RDCalc(lclTtlB['rdB'], c, lcltm)
            else:
                newRatingB = lclTtlB['ratingA']
                if unchange:
                    newRDB = lclTtlB['rdA']
                else:
                    lcltm = (x['date'] - lclTtlB['date']).days/float(ratePer)
                    newRDB = RDCalc(lclTtlB['rdA'], c, lcltm)
        else:
            exchInd = lclB[lclB['teamAId']==x['teamBId']].index
            lclB.loc[exchInd, ['teamAId', 'teamBId', 'ratingA', 'ratingB', 'rdA', 'rdB']]= lclB.loc[exchInd, ['teamBId', 'teamAId', 'ratingB', 'ratingA', 'rdB', 'rdA']].values
#            lclB.loc[exchInd, 'difScore'] = lclB.loc[exchInd, 'difScore'].replace({0: 1, 1: 0})
            lclB.loc[exchInd, 'difScore'] = (1 - lclB.loc[exchInd, 'difScore']).values
            lastRow = lclB.iloc[0]
            ratingB = lastRow['ratingB']
#            if (x['date'] - lastRow['date']).total_seconds()/ttlSec * t>1:
#                print (x['date'] - lastRow['date']).total_seconds()/ttlSec * t, (x['date'] - lastRow['date']).total_seconds(), ttlSec, t, x['date'], lastRow['date']
            
            if unchange:
                rdB = lastRow['rdB']
            else:
                lcltm = (x['date'] - lastRow['date']).days/float(ratePer)
                rdB = RDCalc(lastRow['rdB'], c, lcltm)
            
            ratingListB = lclB['ratingB'].values
            ratingListBOpp = lclB['ratingA'].values#ratings of opponents
#            gRDListBOpp = gRDCalc(q, np.vectorize(lambda z: RDCalc(z, c))(lclB['rdA'].values))#RD of opponents
            gRDListBOpp = gRDCalc(q, lclB['rdA'].values)#RD of opponents
#            experiment = gRDCalc(q, (lclB['rdB'].values**2 + lclB['rdA'].values**2)**0.5)
#            expScoreBList = expScoreCalc(experiment, ratingListB, ratingListBOpp, par)
            expScoreBList = expScoreCalc(gRDListBOpp, ratingListB, ratingListBOpp, par)
#            lclBRes = lclB['difScore'].replace({0: 1, 1: 0}).values
            lclBRes = (1 - lclB['difScore']).values
#            winBonusList = np.array(map(lambda x: winBonus if x>0.5 else 0, lclBRes))
            dSqrB = dSqrCalc(q, gRDListBOpp, expScoreBList)
            rdSqrDSqrB = rdSqrDSqrCalc(rdB, dSqrB)
            newRDB = min(RDChangeCalc(rdSqrDSqrB), 350)
            newRatingB = max(-1500.0, min(ratingChangeCalc(ratingB, q, rdSqrDSqrB, gRDListBOpp, expScoreBList, lclBRes), 4500.0))

        lcldf.loc[_, 'ratingB'] = newRatingB
        lcldf.loc[_, 'rdB'] = newRDB
            
        if not np.isnan(x['difScore']):#x['difScore']!=np.nan:
            expScore = expScoreCalc(gRDCalc(q, newRDB, True), newRatingA, newRatingB, par, True)
            err = abs(x['difScore'] - expScore)
            if (expScore>0.5 and x['difScore']>0.5) or (expScore<0.5 and x['difScore']<0.5):
                lstAErr += [[err, err/2.0]]
            else:
                lstAErr += [[err, err]]
                
#            expScore = expScoreCalc(gRDCalc(q, newRDA, True), newRatingB, newRatingA, par, True)
#            err = abs(1 - x['difScore'] - expScore)
#            if (expScore>0.5 and 1 - x['difScore']>0.5) or (expScore<0.5 and 1 - x['difScore']<0.5):
#                lstBErr += [[err, err/2.0]]
#            else:
#                lstBErr += [[err, err]]
#        else:
##            if _ in [2556, 3364, 4004]:
##                print _
#            lstAErr += [[0, 0]]
#            lstBErr += [[0, 0]]
        if _%5000==0:
            print _
#        lstBErr += [-x['difScore'] - (32 * expScore - 16)]
            
    return lstAErr
#    with open('glicko.txt', 'a') as f:
##        f.write('winBonus, unratePer, par, ratePer: %s\n'%str([winBonus, unratePer, par, ratePer]))
#        f.write('unratePer, par, ratePer: %s\n'%str([unratePer, par, ratePer]))
#        f.write(str(lcldf.describe())+'\n')



#!!!!!!
#    print lcldf.describe()
##    print (lcldf['difScore']==0.5).sum()/float(lcldf.shape[0])
##    print 'shape before', lcldf.shape[0]
##    lcldf.drop(lcldf[lcldf['difScore']==0.5].index, inplace=True)
#    cond = lcldf['date']>dt.date(2015, 12, 31)
##    print 'shape after', lcldf.shape[0]
#    lcldf['pred'] = lcldf['ratingA'] > lcldf['ratingB']
#    lcldf['pred1'] = lcldf['difScore'] > 0.5
##    with open('glicko.txt', 'a') as f:
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
#!!!!!!
    
    
    
#    lcldf.drop(lcldf[lcldf['difScore']==0.5].index, inplace=True)
#            print 'YEAH'
#        return
        
        
        
        
            
        
def ratingDistribution(filename, per):
    global df
    print 'START ratingDistribution'
    td = dt.timedelta(days=per)
    
    lcldf = pd.read_csv(filename, sep=';')
    lcldf['date'] = pd.to_datetime(lcldf['date'])
    for ind, row in lcldf.iterrows():
        tmpIndex = df[(df['teamAId']==row['teamId']) & (df['date']>=row['date']) & (df['date']<row['date']+td)].index
        df.loc[tmpIndex, 'ratingA'] = row['rating']
        df.loc[tmpIndex, 'rdA'] = row['rd']
        if tmpIndex.shape[0]!=row['gamesQnt']:
            tmpIndex = df[(df['teamBId']==row['teamId']) & (df['date']>=row['date']) & (df['date']<row['date']+td)].index
            df.loc[tmpIndex, 'ratingB'] = row['rating']
            df.loc[tmpIndex, 'rdB'] = row['rd']
            
    
    df['pred'] = df['ratingA'] > df['ratingB']
    df['pred1'] = df['difScore'] > 0
    print (df['pred'] == df['pred1']).sum()/float(df.shape[0])
    print (df['difScore']==0.5).sum()/float(df.shape[0])
    
#    print lcldf.info()
#    print lcldf.head()
           
    
def saveToFile(filename):
    with open(filename, 'w') as f:
        df[['matchlinkId', 'teamAId', 'teamBId', 'map', 'ratingA', 'ratingB', 'rdA', 'rdB', 'probA', 'probB']].to_csv(f, sep=';', index=False)

import pymongo
client = pymongo.MongoClient(port=33333)  
db = client['csgo']

cnt = 0
#for filename in ['glickoRatingChangeExp.csv', 'glickoRatingChangeKefExp.csv', 'glickoRatingUnchangeExp.csv']:
#for filename in ['glickoRatingChange.csv', 'glickoRatingChangeKef.csv', 'glickoRatingUnchange.csv']:
#for unratePer in np.arange(12, 21, 4):
##    for winBonus in np.arange(0, 0.26, 0.05):
#        for ratePer in np.arange(2, 18, 4):
#            if ratePer>unratePer:
#                continue
df = pd.read_csv('../data/csgoEloDS.csv', sep=';')
df['date'] = df['date'].apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
#            df = df.sort_values('date').reset_index(drop=True)
#df['difScore'] = df['difScore'].apply(lambda x: 0.5 if x==0 else 0 if x<0 else 1)
#    print (~df.isnull().any(axis=1)).sum(), df.shapek
df.loc[~df.isnull().any(axis=1), 'difScore'] = df.loc[~df.isnull().any(axis=1), 'difScore']/32.0 + 0.5
#    break

df['ratingA'] = 1500
df['ratingB'] = 1500
df['rdA'] = 350
df['rdB'] = 350

#    print df.info()
#            params = [df, winBonus, params*unratePer, 400, params*ratePer]
#params = [df, params*unratePer, 400, params*ratePer]
unratePer = 112
ratePer = 30
params = [df] + [unratePer, 400, ratePer]

teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
maxdate = df['date'].max()


tm = time.time()
print dt.datetime.today()
unchange = False
#if cnt<2:
#    unchange = False
#else:
#    unchange = True
params += [unchange]
print params[1:]
eror = ratingCalculation(*params)
cnt += 1
print time.time() - tm
print
df['probA'] = np.round(df.apply(lambda x: win_probability(x), axis=1).values, 6)
df['probB'] = 1 - df['probA'].values

print df[['ratingA', 'ratingB', 'probA', 'rdA', 'rdB']].describe()
print (((df['probA']>0.5) & (df['difScore']>0.5)) | ((df['probA']<0.5) & (df['difScore']<0.5))).value_counts(normalize=True)
print pd.DataFrame(eror).describe()
#print df.info()
#print df.describe()
saveToFile('glicko.csv')

c = cCalc(unratePer)
stDate = maxdate - dt.timedelta(days=unratePer)
lclTtl = np.unique(np.append(df[df['date']>=stDate]['teamAId'].value_counts().index.values, df[df['date']>=stDate]['teamBId'].value_counts().index.values))
tms = []
for ids in lclTtl:
    lcl = df[((df['teamAId']==ids) | (df['teamBId']==ids)) & (df['date']>=stDate)].iloc[-1]
    if lcl['teamAId']==ids:
        lcltm = (maxdate - lcl['date']).days/float(ratePer)
        lcl['rdA'] = RDCalc(lcl['rdA'], c, lcltm)
        tms += [ [lcl['teamAId'], lcl['ratingA'], lcl['rdA']] ]
    elif lcl['teamBId']==ids:
        lcltm = (maxdate - lcl['date']).days/float(ratePer)
        lcl['rdB'] = RDCalc(lcl['rdB'], c, lcltm)
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
            
client.close()  

#tm = time.time()
#ratingDistribution('../cs_new/glicko.csv', 7)
#print time.time() - tm


#print df.iloc[:1000][(df['teamAId']==6385) | (df['teamBId']==6385)]
#for val in sorted(ratingDfMain.items(), key=lambda x: x[1][-1][0], reverse=True)[:10]:
#    print val[0], val[1][-1]
#print np.dot(np.array([2, 3]), np.array([5, 6]), np.array([2, 3]))
#print np.linalg.multi_dot([np.array([2, 3]), np.array([5, 6]), np.array([2, 3])])

#client.close()