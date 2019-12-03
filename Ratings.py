# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:18:16 2018

@author: Andrew
"""

from random import random, sample, choice
from tqdm import tqdm
#from numpy import array, dot, mean
from numpy.linalg import pinv
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import time
import re
import copy

import trueskill
import math
import pymongo

from Parser import Parser
from DataBase import DataBase

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']

trueskill.BETA = 25/6.0

class RatingParser(Parser):
    def __init__(self, db=DataBase()):
        Parser.__init__(self, db)
        self.db = db# mongodb by default
        
    def __updateTeamRating(self, res, datestr):
        _ = re.findall(r'position">#(\d+).*?\((\d+) points.*?href="/team/(\d+)/(.*?)"', res, re.DOTALL|re.IGNORECASE)
        if not _:
            print 'not found anny ratings'
#        print re.findall(r'position">#(\d+).*?\((\d+) points.*?href="/team/(\d+)/(.*?)"', res, re.DOTALL|re.IGNORECASE)
        for team in _:
            position, teamRating, teamId, teamUrlName = team
            condition = {'teamId': int(teamId), 'date': datestr}
            update_vals = {'teamId': int(teamId), 
                           'teamUrlName': teamUrlName, 
                           'teamRating': int(teamRating), 
                           'position': int(position),
                           'date': datestr
                           }
            self.db.update('teamsRating', condition, update_vals, upsert=True)
        
        countries = re.findall(r'ranking-country"><a href="(.*?)"', res, re.DOTALL|re.IGNORECASE)
        
        for cntry in countries:
            teamRanks = self.request('https://www.hltv.org%s'%cntry)
            
            _ = re.findall(r'\((\d+) points.*?href="/team/(\d+)/(.*?)"', teamRanks, re.DOTALL|re.IGNORECASE)
            if not _:
                print 'not found country %s ratings'%cntry
            for team in _:
                teamRating, teamId, teamUrlName,  = team
                condition = {'teamId': int(teamId), 'date': datestr}
                update_vals = {'teamId': int(teamId), 
                               'teamUrlName': teamUrlName, 
                               'teamRating': int(teamRating), 
                               'date': datestr
                               }
                self.db.update('teamsRating', condition, update_vals, upsert=True)
                
    def getTeamsRating(self, updateAll=False):
        #TODO TOP 30 RANKING HISTORY FROM HLTV
        print '#getTeamsRating update week TeamsRating'
    #    res = csgo.request('https://www.hltv.org/ranking/teams/2017/december/18')
    #    print re.findall(r'position">#(\d+).*?name js-link" data-url="/team/(\d+)/(.*?)".*?\((\d+) points', res, re.DOTALL|re.IGNORECASE)
    #    res, url = csgo.request('https://www.hltv.org/ranking/teams/2015/december/28', retUrl=True)
        res, url = self.request('https://www.hltv.org/ranking/teams', retUrl=True)
        if not url:
            return
    #    db['teamsRating'].delete_many({}) -> self.db.delete('teamsRating', {})
        dturl = '/'.join(url.split('/')[-3:])
        date = datetime.datetime.strptime(dturl, "%Y/%B/%d")
    #    if db['teamsRating'].find_one({'date': str(date), 'position': {'$lte': 15}}) and not updateAll:
    #        return
        print date
        if updateAll:
            stopdate = datetime.datetime.strptime('2015-08-15', '%Y-%m-%d')
        
        visitedDates = []
        links = sorted(set(map(lambda x: datetime.datetime.strptime('/'.join(x), "%Y/%B/%d"), re.findall(r'/ranking/teams/(\d+)/(\w+)/(\d+)"', res, re.DOTALL|re.IGNORECASE))), reverse=True)
        
        while True:
            
            datestr = str(date)
        
            self.__updateTeamRating(res, datestr)
        
    #        date -= datetime.timedelta(days=7)
            while links:
                if links[0]>=date:
                    links.pop(0)
                    continue
                date = links.pop(0)
                break
    #        if date<todayDate-datetime.timedelta(days=380):
            if (lambda: date<=stopdate if updateAll else False)() or visitedDates==date:
                return
            print date.strftime('%Y/%B/%d').lower(), date
    #        print date
            visitedDates = date
            
            if self.db.find('teamsRating', {'date': str(date), 'position': {'$lte': 15}}, multi=False) and not updateAll:
                return
#            if db['teamsRating'].find_one({'date': str(date), 'position': {'$lte': 15}}) and not updateAll:
#                return
            res = self.request('https://www.hltv.org/ranking/teams/%s'%date.strftime('%Y/%B/%d').lower())
            while not res:
                print 'one more try'
                time.sleep(2)
                res = self.request('https://www.hltv.org/ranking/teams/%s'%date.strftime('%Y/%B/%d').lower())
            
            newlinks = sorted(set(map(lambda x: datetime.datetime.strptime('/'.join(x), "%Y/%B/%d"), re.findall(r'/ranking/teams/(\d+)/(\w+)/(\d+)"', res, re.DOTALL|re.IGNORECASE))), reverse=True)
    #        print newlinks
            for i in newlinks[1:]:
    #            if (lambda: links[0]>i if links else date>i)() :
    #                break
                if visitedDates<=i:
                    continue                
                links.insert(0, i)
                
            links = sorted(links, reverse=True)
    
    
    def updateTeamRatingSpecDate(self, url='https://www.hltv.org/ranking/teams/2018/may/7'):
        #TODO UPDATE RATINGS FOR A SPECIFIC DATE
        dturl = '/'.join(url.split('/')[-3:])
        date = datetime.datetime.strptime(dturl, "%Y/%B/%d")
        datestr = str(date)
        res = self.request('https://www.hltv.org/ranking/teams')
        
        self.__updateTeamRating(res, datestr)
        
    def __getRankHistory(self, teamId):
        try:
            #https://www.hltv.org/team/8355/kolding
            res = self.request('https://www.hltv.org/team/%s/asddhhffsd'%teamId)
            rankHistory = re.findall(r'3e(\d+)\w\w? (\w+) (\d+).*?#(\d+)', res, re.IGNORECASE|re.DOTALL)
            rankHistory = [[str(datetime.datetime.strptime('%s %s %s'%i[:3], '%d %B %Y')), int(i[3])] for i in rankHistory]
            return rankHistory
        except Exception as e:
            print e
            return []
    
    def makeRankHistory(self, date=''):
        # TODO POSITION HISTORY CHANGES
        print '#makeRankHistory',
        if not date:
            teams = set([i['teamAId'] for i in self.db.find('fullMatchHistory')])
            teams = teams | set([i['teamBId'] for i in self.db.find('fullMatchHistory')])
        else:
            teams = set([i['teamAId'] for i in self.db.find('fullMatchHistory', {'date': {'$gte': date}})])
            teams = teams | set([i['teamBId'] for i in self.db.find('fullMatchHistory', {'date': {'$gte': date}})])
    #    if not date:
    #        teams = set([i['teamAId'] for i in db['fullMatchHistory'].find()])
    #        teams = teams | set([i['teamBId'] for i in db['fullMatchHistory'].find()])
    ##        teams = set([(i['teamAId'], i['teamAUrlName']) for i in db['fullMatchHistory'].find({'position': {'$exists': False}})])
    ##        teams = teams | set([(i['teamBId'], i['teamBUrlName']) for i in db['fullMatchHistory'].find({'position': {'$exists': False}})])
    #    else:
    #        teams = set([i['teamAId'] for i in db['fullMatchHistory'].find({'date': {'$gte': date}})])
    #        teams = teams | set([i['teamBId'] for i in db['fullMatchHistory'].find({'date': {'$gte': date}})])
            
        print 'scan %s'%len(teams)
        for ind, teamId in enumerate(teams):
            if not ind%20:
                print '%s/%s'%(ind, len(teams))
                
#            print teamId
            rankHstr = self.__getRankHistory(teamId)
            if rankHstr:
                rankHstr = sorted(rankHstr, key=lambda x: x[0], reverse=True)
                    
            flag = True
            for dt, position in rankHstr:
                if dt<'2015-09-01 00:00:00':
                    break
                if (lambda: dt<date if date else False)():
    #                lcldt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    #                lcldate = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    if flag:# and lcldate-lcldt>datetime.timedelta(days=10):
                        flag = False
                    else:
                        break
                    
                self.db.update('teamsRating', {'teamId': teamId, 'date': dt}, {'teamId': teamId, 'date': dt, 'position': position})
    #            db['teamsRating'].update_one({'teamId': teamId, 'date': dt}, {'$set': 
    #                                                                            {'teamId': teamId, 
    #                                                                             'date': dt,
    #                                                                             'position': position}}, upsert=True)
    
    
class RatingSystem():
    #TODO RatingSystem
    def __init__(self):
        pass
    
    def win_probability(self, ratingA, ratingB, rdA, rdB):
        delta_mu = ratingA - ratingB
        sum_sigma = rdA**2 + rdB**2
        denom = math.sqrt(2 * trueskill.BETA**2 + sum_sigma)
        return trueskill.TrueSkill().cdf(delta_mu / denom)
    
    
class HLTV(RatingSystem):
    # TODO HLTV
    def __init__(self, db=DataBase(), *args):
        super(RatingSystem, self).__init__()
        self.db = db
        self.model = lgb.Booster(model_file='LGBHLTVRatingPrediction.model')
        self.dataPreprocessor = DataPreprocessor(self.db)
        self.rSystem, self.unratePer, self.unrate = args
        
    def __teamRatingsPastTwoMonthsGameStatsHF(self, localData):
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
            
    
    def updateRating(self, x, ratingA, ratingB):
        # Добавить информацию, чтобы 'difScore']>-17, ....
        lclTtlA = list(self.db.find('fullMatchHistory', {'difScore': {'$gt': -17}, 'teamAadravg': {'$gte': 0}, 'teamAMatchRating': {'$gt': 0}, '$or': [{'teamAId': x['teamAId']}, {'teamBId': x['teamAId']}], 
                                                         '$and': [{'date': {'$gte': str(x['date'] - datetime.timedelta(days=64))}}, {'date': {'$lte': str(x['date'])}}]},
                                                         {'difScore': 1, 'teamAadravg': 1, 'teamBadravg': 1, 'teamAMatchRating': 1, 'teamBMatchRating': 1, 'teamAId': 1, 'teamBId': 1}))
        lclTtlB = list(self.db.find('fullMatchHistory', {'difScore': {'$gt': -17}, 'teamAadravg': {'$gte': 0}, 'teamAMatchRating': {'$gt': 0}, '$or': [{'teamAId': x['teamBId']}, {'teamBId': x['teamBId']}], 
                                                         '$and': [{'date': {'$gte': str(x['date'] - datetime.timedelta(days=64))}}, {'date': {'$lte': str(x['date'])}}]},
                                                         {'difScore': 1, 'teamAadravg': 1, 'teamBadravg': 1, 'teamAMatchRating': 1, 'teamBMatchRating': 1, 'teamAId': 1, 'teamBId': 1}))
        
        mainDf = pd.DataFrame(lclTtlA + lclTtlB)
        columns = ['difScore', 'teamAadravg', 'teamBadravg', 'teamAMatchRating', 'teamBMatchRating', 'teamAId', 'teamBId']
        oversample_columns = ['difScore', 'teamBadravg', 'teamAadravg', 'teamBMatchRating', 'teamAMatchRating', 'teamBId', 'teamAId']
        mainDf = pd.concat([mainDf[columns], mainDf[oversample_columns]])
        
        mainDf.drop_duplicates(inplace=True)
        
        lstA = self.__teamRatingsPastTwoMonthsGameStatsHF(mainDf.loc[mainDf['teamAId']==x['teamAId']])
        lstB = self.__teamRatingsPastTwoMonthsGameStatsHF(mainDf.loc[mainDf['teamAId']==x['teamBId']])
        
        #add teamAvgIncome
        tmp = self.dataPreprocessor([x['teamAId'], x['teamBId']], x['date'])
        lstA = [tmp['teamAId']] + lstA
        lstB = [tmp['teamBId']] + lstB
        
        #add last team HLTV positions -------
        positionA = self.db.find('ratings', {'teamId': x['teamAId'], 'position': {'$exists': 1}, 'type': 'HLTV'}, multi=False, sort=[('date', -1)])
        positionB = self.db.find('ratings', {'teamId': x['teamBId'], 'position': {'$exists': 1}, 'type': 'HLTV'}, multi=False, sort=[('date', -1)])
        
        if not positionA:
            lstA = [31] + lstA
        else:
            positionA['date'] = datetime.datetime.strptime(positionA['date'], '')
            if positionA['date'] + datetime.timedelta(days=10)<datetime.datetime.today():# инфа о рейтинге команды была получена давно, а свежей инфы нету возможно из-за того, что у команды плохой рейтинг
                lstA = [31] + lstA
            else:
                if positionA['position']<1 or positionA['position']>30:
                    lstA = [31] + lstA
                else:
                    lstA = [positionA['position']] + lstA
            
        if not positionB:
            lstB = [31] + lstB
        else:
            positionB['date'] = datetime.datetime.strptime(positionB['date'], '')
            if positionB['date'] + datetime.timedelta(days=10)<datetime.datetime.today():# инфа о рейтинге команды была получена давно, а свежей инфы нету возможно из-за того, что у команды плохой рейтинг
                lstB = [31] + lstB
            else:
                if positionB['position']<1 or positionB['position']>30:
                    lstB = [31] + lstB
                else:
                    lstB = [positionB['position']] + lstB
        # -------            
        
        ratingA['%s_rating'%self.rSystem], ratingB['%s_rating'%self.rSystem] = ratingA['%s_ratinglast'%self.rSystem], ratingB['%s_ratinglast'%self.rSystem]
        
        #prediction
        ratingA['%s_ratinglast'%self.rSystem], ratingB['%s_ratinglast'%self.rSystem] = self.model.predict([lstA, lstB])
        
        ratingA['%s_date'%self.rSystem] = str(x['date'])
        ratingB['%s_date'%self.rSystem] = str(x['date'])
        
        ratingA['%s_matchId'%self.rSystem] = x['matchlinkId']
        ratingB['%s_matchId'%self.rSystem] = x['matchlinkId']
            
        return ratingA, ratingB, {'ratingA_%s'%self.rSystem: ratingA['%s_rating'%self.rSystem], 'ratingB_%s'%self.rSystem: ratingB['%s_rating'%self.rSystem]}
    


class Trueskill(RatingSystem):
    # TODO Trueskill
    def __init__(self, db=DataBase(), *args):
        super(RatingSystem, self).__init__()
        self.db = db
        self.rSystem, self.unratePer, self.unrate = args
        
    
    
    def updateRating(self, x, ratingA, ratingB):
        rSystem, unratePer, unrate = self.rSystem, self.unratePer, self.unrate
        
        if unrate:
            unratedTDelta = datetime.timedelta(days=unratePer)#период за который команда считается unrated
            stDate = x['date'] - unratedTDelta
            lclTtlA = self.db.find('fullMatchHistory', {'difScore': {'$gte': -16}, '$or': [{'teamAId': x['teamAId']}, {'teamBId': x['teamAId']}], '$and': [{'date': {'$gte': str(stDate)}}, {'date': {'$lt': str(x['date'])}}]})
            lclTtlB = self.db.find('fullMatchHistory', {'difScore': {'$gte': -16}, '$or': [{'teamAId': x['teamBId']}, {'teamBId': x['teamBId']}], '$and': [{'date': {'$gte': str(stDate)}}, {'date': {'$lt': str(x['date'])}}]})
#            lclTtlA = db['fullMatchHistory'].find({'difScore': {'$gte': -16}, '$or': [{'teamAId': x['teamAId']}, {'teamBId': x['teamAId']}], '$and': [{'date': {'$gte': str(stDate)}}, {'date': {'$lt': str(x['date'])}}]})
#            lclTtlB = db['fullMatchHistory'].find({'difScore': {'$gte': -16}, '$or': [{'teamAId': x['teamBId']}, {'teamBId': x['teamBId']}], '$and': [{'date': {'$gte': str(stDate)}}, {'date': {'$lt': str(x['date'])}}]})
            
            
            if lclTtlA.count()==0:
                ratingA['%s_rating'%rSystem] = max(25, ratingA['%s_ratinglast'%rSystem])
                ratingA['%s_rd'%rSystem] = max(6.458, ratingA['%s_rdlast'%rSystem])
                ratingA['%s_matchId'%rSystem] = x['matchlinkId']
                ratingA['%s_ratinglast'%rSystem] = max(25, ratingA['%s_ratinglast'%rSystem])
                ratingA['%s_rdlast'%rSystem] = max(6.458, ratingA['%s_rdlast'%rSystem])
            if lclTtlB.count()==0:
                ratingB['%s_rating'%rSystem] = max(25, ratingB['%s_ratinglast'%rSystem])
                ratingB['%s_rd'%rSystem] = max(6.458, ratingB['%s_rdlast'%rSystem])
                ratingB['%s_matchId'%rSystem] = x['matchlinkId']
                ratingB['%s_ratinglast'%rSystem] = max(25, ratingB['%s_ratinglast'%rSystem])
                ratingB['%s_rdlast'%rSystem] = max(6.458, ratingB['%s_rdlast'%rSystem])
    
        if x['matchlinkId']!=ratingA['%s_matchId'%rSystem]:
            ratingA['%s_rating'%rSystem] = ratingA['%s_ratinglast'%rSystem]
            ratingA['%s_rd'%rSystem] = ratingA['%s_rdlast'%rSystem]
        if x['matchlinkId']!=ratingB['%s_matchId'%rSystem]:
            ratingB['%s_rating'%rSystem] = ratingB['%s_ratinglast'%rSystem]
            ratingB['%s_rd'%rSystem] = ratingB['%s_rdlast'%rSystem]
            
    
        if x['difScore']>=0:
            teamA = trueskill.Rating(ratingA['%s_ratinglast'%rSystem], ratingA['%s_rdlast'%rSystem])
            teamB = trueskill.Rating(ratingB['%s_ratinglast'%rSystem], ratingB['%s_rdlast'%rSystem])
            if x['difScore']==0.5:#draw
                teamA, teamB = trueskill.rate_1vs1(teamA, teamB, drawn = True)
            elif x['difScore']>0.5:#teamA win
                teamA, teamB = trueskill.rate_1vs1(teamA, teamB)
            else:#teamB win
                teamB, teamA = trueskill.rate_1vs1(teamB, teamA)
                
            ratingA['%s_ratinglast'%rSystem] = teamA.mu
            ratingA['%s_rdlast'%rSystem] = teamA.sigma
            
            ratingB['%s_ratinglast'%rSystem] = teamB.mu
            ratingB['%s_rdlast'%rSystem] = teamB.sigma
            
        ratingA['%s_date'%rSystem] = str(x['date'])
        ratingB['%s_date'%rSystem] = str(x['date'])
        ratingA['%s_matchId'%rSystem] = x['matchlinkId']
        ratingB['%s_matchId'%rSystem] = x['matchlinkId']
            
        return ratingA, ratingB, {'ratingA_%s'%rSystem: ratingA['%s_rating'%rSystem], 'ratingB_%s'%rSystem: ratingB['%s_rating'%rSystem]}
            
     

class Elo(RatingSystem):
    # TODO Elo
    def __init__(self, db=DataBase(), *args):
        super(RatingSystem, self).__init__()
        self.db = db
        self.rSystem, self.winBonus, self.k, self.par, self.days, self.inf = args
        
    def __elo_glicko2_expScoreCalc(self, rA, rB, par):
        if (rB-rA)/float(par)>8:
            return 0
        return 1/float(1 + 10**( (rB-rA)/float(par) ) )
    
    def __elo_difScoreCalc(self, actScore, expScore):
        return actScore - (32 * expScore - 16)
    
    def __elo_changeRating(self, r, winBonus, k, difScore):
        return r + winBonus + k * difScore
    
    def elo_ratingInflation(self, rating, date1, date2, days):
        dateDif = (date2 - date1).days
        if dateDif>=days:
    #        if rating<1000:
    #            return rating
            return 1000
        if rating<1000:
            return rating
        return 1000 + (days-dateDif)/float(days)*(rating-1000)
    
    def updateRating(self, x, ratingA, ratingB):
        rSystem, winBonus, k, par, days, inf = self.rSystem, self.winBonus, self.k, self.par, self.days, self.inf
        
        if inf:
            ratingA['%s_ratinglast'%rSystem] = self.elo_ratingInflation(ratingA['%s_ratinglast'%rSystem], ratingA['%s_date'%rSystem], x['date'], days)
            ratingB['%s_ratinglast'%rSystem] = self.elo_ratingInflation(ratingB['%s_ratinglast'%rSystem], ratingB['%s_date'%rSystem], x['date'], days)
        
        if x['matchlinkId']!=ratingA['%s_matchId'%rSystem]:
            ratingA['%s_rating'%rSystem] = ratingA['%s_ratinglast'%rSystem]
        if x['matchlinkId']!=ratingB['%s_matchId'%rSystem]:
            ratingB['%s_rating'%rSystem] = ratingB['%s_ratinglast'%rSystem]
    
        if x['difScore']>0:
            expScore = self.__elo_glicko2_expScoreCalc(ratingA['%s_ratinglast'%rSystem], ratingB['%s_ratinglast'%rSystem], par)
            difScore = self.__elo_difScoreCalc(x['difScore'], expScore)
            newRating = self.__elo_changeRating(ratingA['%s_ratinglast'%rSystem], winBonus, k, difScore)
            if newRating>3000:
                newRating = 3000
            ratingA['%s_ratinglast'%rSystem] = newRating
            
            expScore1 = self.__elo_glicko2_expScoreCalc(ratingB['%s_ratinglast'%rSystem], ratingA['%s_ratinglast'%rSystem], par)
            difScore1 = self.__elo_difScoreCalc(-x['difScore'], expScore1)
            newRating = self.__elo_changeRating(ratingB['%s_ratinglast'%rSystem], 0, k, difScore1)
            if newRating>3000:
                newRating = 3000
            ratingB['%s_ratinglast'%rSystem] = newRating
            
        elif x['difScore']<0:
            expScore = self.__elo_glicko2_expScoreCalc(ratingA['%s_ratinglast'%rSystem], ratingB['%s_ratinglast'%rSystem], par)
            difScore = self.__elo_difScoreCalc(x['difScore'], expScore)
            newRating = self.__elo_changeRating(ratingA['%s_ratinglast'%rSystem], 0, k, difScore)
            if newRating>3000:
                newRating = 3000
            ratingA['%s_ratinglast'%rSystem] = newRating
            
            expScore1 = self.__elo_glicko2_expScoreCalc(ratingB['%s_ratinglast'%rSystem], ratingA['%s_ratinglast'%rSystem], par)
            difScore1 = self.__elo_difScoreCalc(-x['difScore'], expScore1)
            newRating = self.__elo_changeRating(ratingB['%s_ratinglast'%rSystem], winBonus, k, difScore1)
            if newRating>3000:
                newRating = 3000
            ratingB['%s_ratinglast'%rSystem] = newRating
            
        else:
            expScore = self.__elo_glicko2_expScoreCalc(ratingA['%s_ratinglast'%rSystem], ratingB['%s_ratinglast'%rSystem], par)
            difScore = self.__elo_difScoreCalc(x['difScore'], expScore)
            newRating = self.__elo_changeRating(ratingA['%s_ratinglast'%rSystem], 0, k, difScore)
            if newRating>3000:
                newRating = 3000
            ratingA['%s_ratinglast'%rSystem] = newRating
            
            expScore1 = self.__elo_glicko2_expScoreCalc(ratingB['%s_ratinglast'%rSystem], ratingA['%s_ratinglast'%rSystem], par)
            difScore1 = self.__elo_difScoreCalc(-x['difScore'], expScore1)
            newRating = self.__elo_changeRating(ratingB['%s_ratinglast'%rSystem], 0, k, difScore1)
            if newRating>3000:
                newRating = 3000
            ratingB['%s_ratinglast'%rSystem] = newRating
            
        ratingA['%s_date'%rSystem] = str(x['date'])
        ratingB['%s_date'%rSystem] = str(x['date'])
        
        ratingA['%s_matchId'%rSystem] = x['matchlinkId']
        ratingB['%s_matchId'%rSystem] = x['matchlinkId']
            
        return ratingA, ratingB, {'ratingA_%s'%rSystem: ratingA['%s_rating'%rSystem], 'ratingB_%s'%rSystem: ratingB['%s_rating'%rSystem]}

def RatingPreprocessor():
    # TODO RatingPreprocessor
    def __init__(self, db=DataBase()):
        self.db = db
        self.mainRatingSystemsInit = {
                                 'HLTV': {'rating': 0}, 
                                 'elo0': {'rating': 1000.0},#, 'rd': 67.6877001717},
                                 'elo0Inf': {'rating': 1000.0},#, 'rd': 51.2712302976},
                                 'trueskill': {'rating': 25, 'rd': 6.458},
                                 'trueskillUnrate': {'rating': 25, 'rd': 6.458},
                                 }
        
        self.mainRatingSystemsConf = {
                                 'elo0': ['elo0', 0, 1, 400, 110, False], #[winBonus, k, par, days, isRatingInflation, Elo]
                                 'elo0Inf': ['elo0Inf', 0, 1, 400, 110, True], #[winBonus, k, par, days, isRatingInflation, Elo]
                                 'trueskill': ['trueskill', 45, False], #[unratePer, unrate, Trueskill]
                                 'trueskillUnrate': ['trueskillUnrate', 45, True], #[unratePer, unrate, Trueskill]
                                 }
        for rSstr, rSclass in zip(*[list(self.mainRatingSystemsConf.keys()), [Elo, Elo, Trueskill, Trueskill] ]):
            self.mainRatingSystemsConf[rSstr].append(rSclass(*self.mainRatingSystemsConf[rSstr]))

    def createGlblUpdateDic(self, mindate):
        glblupdateDic = {}
        for key1 in self.mainRatingSystemsInit:
            for key2 in self.mainRatingSystemsInit[key1]:
                glblupdateDic['%s_%s'%(key1, key2)] = self.mainRatingSystemsInit[key1][key2]
#                if 'glicko2' not in key1:
                glblupdateDic['%s_%slast'%(key1, key2)] = self.mainRatingSystemsInit[key1][key2]
    #        if 'elo' in key1:
            glblupdateDic['%s_date'%key1] = str(mindate)
#            if 'glicko2' not in key1:
            glblupdateDic['%s_matchId'%key1] = 0
    #        [
    #        рейтинг_на_момент_начала_серии_BO, - elo0_ratinglast
    #        matchlinkId_последней_окончевшейся_серии_BO, - elo0_matchId
    #        рейтинг_по_окончанию_последней_карты, - elo0_rating
    #        дата_последнего_изменения_рейтинга - elo0_date
    #        ]
        return glblupdateDic
            
    def initSpecificTeamRatings(self, teamId, mindate, glblupdateDic={}):
        if not glblupdateDic:
            glblupdateDic = createGlblUpdateDic(mindate)
                
        lclUpdateDic = {'teamId': str(teamId)}
        lclUpdateDic.update(glblupdateDic)
        self.db.update('ratings', {'teamId': str(teamId)}, lclUpdateDic)
        return lclUpdateDic
    
    def initAllTeamsRatings(self):
        with open('csgoEloDS.csv', 'r') as f:
            df = pd.read_csv(f, sep=';')
            df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
            mindate = df['date'].min()
            teamIds = np.unique(np.append(df['teamAId'].value_counts().index.values, df['teamBId'].value_counts().index.values))
        glblupdateDic = createGlblUpdateDic(mindate)
        for ids in teamIds:
            self.initSpecificTeamRatings(ids, mindate, glblupdateDic)
    #        lclUpdateDic = {'teamId': str(ids)}
    #        lclUpdateDic.update(glblupdateDic)
    #        db['ratings'].update_one({'teamId': str(ids)}, {'$set': lclUpdateDic}, upsert=True)
        
    def getCurrentRatings(self, rating, matchId, date):
        for rS in self.mainRatingSystemsConf.keys():
            if 'trueskill' in rS:
                if self.mainRatingSystemsConf[rS][1]:
                    unratedTDelta = datetime.timedelta(days=self.mainRatingSystemsConf[rS][0])#период за который команда считается unrated
                    stDate = date - unratedTDelta
                    lclTtlA = self.db.find('fullMatchHistory', {'difScore': {'$gte': -16}, '$or': [{'teamAId': rating['teamId']}, {'teamBId': rating['teamId']}], '$and': [{'date': {'$gte': str(stDate)}}, {'date': {'$lt': str(date)}}]})
                    
                    
                    if lclTtlA.count()==0:
                        rating['%s_ratinglast'%rS] = max(25, rating['%s_ratinglast'%rS])
                        rating['%s_rdlast'%rS] = max(6.458, rating['%s_rdlast'%rS])
                        
                
                rating['%s_rd'%rS] = rating['%s_rdlast'%rS]
            elif 'elo' in rS:
                if self.mainRatingSystemsConf[rS][-2]:
                    rating['%s_ratinglast'%rS] = self.mainRatingSystemsConf[rS][-1].elo_ratingInflation(rating['%s_ratinglast'%rS], datetime.datetime.strptime(rating['%s_date'%rS], '%Y-%m-%d %H:%M:%S'), date, self.mainRatingSystemsConf[rS][-3])
                    
                rating['%s_rd'%rS] = rating['%s_rd'%rS]
                
            rating['%s_rating'%rS] = rating['%s_ratinglast'%rS]
        return rating
            
        
    def updateRating(self, rSystem):
    #    lastDate = db['ratings']
        res = self.db.find('fullMatchHistory', {'ratingA_%s'%rSystem: {'$exists': False}, 'ratingB_%s'%rSystem: {'$exists': False}}, 
                                               {'matchlinkId': 1, 'map': 1, 'teamAId': 1, 'teamBId': 1, 'date': 1, 'teamAResScore': 1, 'teamBResScore': 1})#.sort([('date', 1)])
        res = sorted(list(res), key=lambda x: x.get('date', ''))
        for cnt, raw in enumerate(res):
            if cnt%2000==0 and cnt:
                print cnt
            if 'date' not in raw:
                continue
            teamArating = self.db.find('ratings', {'teamId': raw['teamAId']}, multi=False)
            if not teamArating:#init ratingA
                teamArating = self.initSpecificTeamRatings(raw['teamAId'], raw['date'])
            teamBrating = self.db.find('ratings', {'teamId': raw['teamBId']}, multi=False)
            if not teamBrating:#init ratingB
                teamBrating = self.initSpecificTeamRatings(raw['teamBId'], raw['date'])
            
    #        print teamArating
    #        print teamArating
            if teamArating['%s_date'%rSystem]>raw['date'] or teamBrating['%s_date'%rSystem]>raw['date']:
                continue
            
            raw['totalRnds'] = raw.get('teamAResScore', -10) + raw.get('teamBResScore', -10)
            if raw['totalRnds']<=15:
                lclDic = {'%s_date'%rSystem: str(raw['date'])}
#                if 'glicko2' not in rSystem:
                lclDic.update({'%s_matchId'%rSystem: raw['matchlinkId']})
                self.db.update('ratings', {'teamId': raw['teamAId']}, {'%s_matchId': raw['matchlinkId'], '%s_date': raw['date']})
                self.db.update('ratings', {'teamId': raw['teamBId']}, {'%s_matchId': raw['matchlinkId'], '%s_date': raw['date']})
                continue
                
    #        print 123
    #        return
            raw['date'] = datetime.datetime.strptime(raw['date'], '%Y-%m-%d %H:%M:%S')
            if raw['totalRnds']>30:
                if raw['teamAResScore']>raw['teamBResScore']:
                    raw['difScore'] = 1
                else:
                    raw['difScore'] = -1
            else:
                raw['difScore']  = raw['teamAResScore'] - raw['teamBResScore']
    
            
            teamArating['%s_date'%rSystem] = datetime.datetime.strptime(teamArating['%s_date'%rSystem], '%Y-%m-%d %H:%M:%S')
            teamBrating['%s_date'%rSystem] = datetime.datetime.strptime(teamBrating['%s_date'%rSystem], '%Y-%m-%d %H:%M:%S')
            updateDic = {}
            if 'elo' in rSystem:
                teamArating, teamBrating = self.mainRatingSystemsConf[rSystem][-1](raw, teamArating, teamBrating, rSystem, *self.mainRatingSystemsConf[rSystem][:-1])
#                win_probability_func = 'elo_glicko2_win_probability'
#                pass#run elo algorithm
            elif 'trueskill' in rSystem:
                teamArating, teamBrating = self.mainRatingSystemsConf[rSystem][-1](raw, teamArating, teamBrating, rSystem, *self.mainRatingSystemsConf[rSystem][:-1])
#                win_probability_func = 'trueskill_win_probability'
#                pass#run trueskill algorithm
#            else:#unknows rSystem
#                continue
            
            self.db.update('ratings', {'teamId': raw['teamAId']}, teamArating)
            self.db.update('ratings', {'teamId': raw['teamBId']}, teamBrating)
    #        probA = np.round(elo_win_probability(teamArating['%s_rating'%rSystem], teamBrating['%s_rating'%rSystem], teamArating['%s_rd'%rSystem]), 6)
#            probA = np.round(globals()[win_probability_func](teamArating['%s_rating'%rSystem], teamBrating['%s_rating'%rSystem], teamArating['%s_rd'%rSystem], teamBrating['%s_rd'%rSystem]), 6)
#            probB = 1 - probA
            updateDic.update({
                              '%s_ratingA'%rSystem: teamArating['%s_rating'%rSystem], 
                              '%s_ratingB'%rSystem: teamBrating['%s_rating'%rSystem],
                              })
            self.db.update('fullMatchHistory', {'matchlinkId': raw['matchlinkId'], 'map': raw['map']}, updateDic)
    #        print 'here'
        
    #initAllTeamsRatings()
    
    #tm = time.time()
    #updateRating('elo0')
    #print time.time() - tm
    
    def updateAllRatings(self):
        dat = self.db.find('fullMatchHistory', {'elo0_ratingA': {'$exists': True}, 'date': {'$exists': True}}, {'date': 1}, multi=False, sort=[('date', -1)])['date']
        print 'LAST RATING UPDATES WAS IN', dat
    #    dat = '2018-09-09 22:10:00'
    #    return
        res = self.db.find('fullMatchHistory', {'elo0_ratingA': {'$exists': False}, 'elo0_ratingB': {'$exists': False}, 'date': {'$gte': dat}}, 
                           {'matchlinkId': 1, 'difScore': 1, 'totalRnds': 1, 'map': 1, 'teamAId': 1, 'teamBId': 1, 'date': 1, 'teamAResScore': 1, 'teamBResScore': 1})#.sort([('date', 1)])
    #    res = db['fullMatchHistory'].find({'elo0_ratingA': {'$exists': False}, 'difScore': {'$exists': True}, 'totalRnds': {'$exists': True}, 'map': {'$exists': True}, 'date': {'$exists': True}}, {'matchlinkId': 1, 'difScore': 1, 'totalRnds': 1, 'map': 1, 'teamAId': 1, 'teamBId': 1, 'date': 1, 'teamAResScore': 1, 'teamBResScore': 1})#.sort([('date', 1)])
        res = sorted(list(res), key=lambda x: (x.get('date', ''), x.get('matchlinkId', ''), x.get('teamAMapLeftToWin', 50)+x.get('teamBMapLeftToWin', 50)))
        print len(res)
        
    #    return
        for cnt, raw in enumerate(res):
    #        if cnt%10==0 and cnt:
    #            print cnt#, raw['matchlinkId']
            if cnt%500==0 and cnt:
                print cnt
                
            if not raw.get('map', '') or not raw.get('date', ''):# or not doc.get('teamARating', ''):#   or not doc.get('mapPicker', '')
                continue
            
    #        if 'date' not in raw:
    ##            print 'firCon'
    #            continue
            teamArating = self.db.find('ratings', {'teamId': raw['teamAId']}, multi=False)
            if not teamArating:#init ratingA
                teamArating = initSpecificTeamRatings(raw['teamAId'], raw['date'])
    #            print teamArating
            teamBrating = self.db.find('ratings', {'teamId': raw['teamBId']}, multi=False)
            if not teamBrating:#init ratingB
                teamBrating = initSpecificTeamRatings(raw['teamBId'], raw['date'])
            
    #        raw['totalRnds'] = raw.get('teamAResScore', -10) + raw.get('teamBResScore', -10)
    #        if raw['totalRnds']>30:
    #            if raw['teamAResScore']>raw['teamBResScore']:
    #                raw['difScore'] = 1
    #            else:
    #                raw['difScore'] = -1
    #        elif raw['totalRnds']>15:
    ##            raw['difScore']  = raw['teamAResScore'] - raw['teamBResScore']
    #        else:
    #            raw['difScore'] = -100
    #        updateDic = {'difScore': raw['difScore']}
            raw['date'] = datetime.datetime.strptime(raw['date'], '%Y-%m-%d %H:%M:%S')
            reservDifScore = raw['difScore']
            updateDic = {}
    #        print teamArating
    #        print teamArating
            for rSystem in self.mainRatingSystemsConf:
    #            if 'elo' in rSystem:
                teamArating['%s_date'%rSystem] = datetime.datetime.strptime(teamArating['%s_date'%rSystem], '%Y-%m-%d %H:%M:%S')
                teamBrating['%s_date'%rSystem] = datetime.datetime.strptime(teamBrating['%s_date'%rSystem], '%Y-%m-%d %H:%M:%S')
                if teamArating['%s_date'%rSystem]>raw['date'] or teamBrating['%s_date'%rSystem]>raw['date']:
    #                teamArating['%s_date'%rSystem] = str(raw['date'])
    #                teamBrating['%s_date'%rSystem] = str(raw['date'])
                    teamArating.pop('%s_date'%rSystem)
                    teamBrating.pop('%s_date'%rSystem)
    #                print 'secCon'
                    continue
                
                if raw['totalRnds']<=15 or raw['difScore']<-16:
    #                lclDic = {'%s_date'%rSystem: str(raw['date'])}
    #                if 'glicko2' not in rSystem:
    #                    lclDic.update({'%s_matchId'%rSystem: raw['matchlinkId']}) 
    #                db['ratings'].update_one({'teamId': raw['teamAId']}, {'$set': lclDic})
    #                db['ratings'].update_one({'teamId': raw['teamBId']}, {'$set': lclDic})
    #                teamArating['%s_date'%rSystem] = str(raw['date'])
    #                teamBrating['%s_date'%rSystem] = str(raw['date'])
                    teamArating.pop('%s_date'%rSystem)
                    teamBrating.pop('%s_date'%rSystem)
                    continue
                    
        #        print 123
        #        return
        
                raw['difScore'] = reservDifScore
    #            teamArating['%s_date'%rSystem] = datetime.datetime.strptime(teamArating['%s_date'%rSystem], '%Y-%m-%d %H:%M:%S')
    #            teamBrating['%s_date'%rSystem] = datetime.datetime.strptime(teamBrating['%s_date'%rSystem], '%Y-%m-%d %H:%M:%S')
                if 'trueskill' in rSystem:
                    raw['difScore'] = raw['difScore']/32.0 + 0.5
                
    #            print rSystem
    #            print raw
                teamArating, teamBrating, lclDic = self.mainRatingSystemsConf[rSystem][-1].updateRating(raw, teamArating, teamBrating)
    #            print teamArating, teamBrating
                updateDic.update(lclDic)
    #            print updateDic
            if teamArating:
                self.db.update('ratings', {'teamId': raw['teamAId']}, teamArating)
            if teamBrating:
                self.db.update('ratings', {'teamId': raw['teamBId']}, teamBrating)
            if updateDic:
                self.db.update('fullMatchHistory', {'matchlinkId': raw['matchlinkId'], 'map': raw['map']}, updateDic)
    #        print 'here'
    #    dc, df = get_fitness(df.copy(), [winBonus, k, par, days], True, inf = cnt%2==0)
    #    
    #    df['probA'] = np.round(df.apply(lambda x: win_probability(x), axis=1).values, 6)
    #    df['probB'] = 1 - df['probA'].values
        
#db['ratings'].delete_many({})
#initAllTeamsRatings()
#keysDel = {i: 1 for i in pd.DataFrame(list(db['fullMatchHistory'].find())).columns if '_rating' in i}
###print keysDel
#if keysDel:
#    db['fullMatchHistory'].update_many({}, {'$unset': keysDel}, upsert=True)

#tm = time.time()
#updateAllRatings()
#print time.time() - tm
#client.close()