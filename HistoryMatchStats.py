# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 08:33:19 2017

@author: Андрей
"""
import time, re, copy, ast
#import pymongo
import datetime
import numpy as np

from DataBase import DataBase 
from Parser import Parser

#from calendar import month_name
#months = {val.lower(): key+1 for key, val in enumerate(month_name[1:])}
#print months

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']

todayDate = datetime.datetime.today()
base = 'https://www.hltv.org'

#db = DataBase()

def catDefine(desc):
    #0 - На вылет, 1 - группа. за очки, 2 - верхняя сетка с правом на ошибку, 3 - борьа за выход с первого места из группы
    desc = desc.lower()
    if 'eliminat' in desc:#OK
        cat = 0
    elif 'decid' in desc:#OK
        cat = 0 
    elif 'lower' in desc:#OK
        cat = 0
    elif 'upper' in desc and not 'grand final' in desc and 'from the upper' not in desc:#OK
        cat = 2
    elif 'round of' in desc:#OK
        cat = 0
    elif 'final' in desc or 'from the upper' in desc:
        cat = 0
    elif 'playoff' in desc:#OK
        cat = 0
    elif 'swiss' in desc:
        if 'win' in desc:
            cat = 0
        else:
            cat = 1
    elif 'win' in desc:# and 'group' in desc:
        cat = 3
    else:#group
        cat = 1
    return cat


#TODO PARSER
class HistoryMatchStats(Parser):
    def __init__(self, db=DataBase()):
        Parser.__init__(self, db)
        self.db = db# mongodb by default
            
    def getStatsMatchistory(self, dat='2015-09-01 00:00:00'):#from stats/results page
        #TODO GET ONLY MAP_LINK_ID, MATCH_TYPE, MATCH_DATE
        print '#getStatsMatchistory and Lan or Online matchType'
    #    stopdate = datetime.datetime.strptime(sorted(list(db['teamsRating'].find()), key=lambda x: x['date'])[0]['date'], "%Y-%m-%d %H:%M:%S")
        
    #    if dat:
        stopdate = datetime.datetime.strptime(dat, "%Y-%m-%d %H:%M:%S")
    #    elif updateAll:
    #        stopdate = datetime.datetime.strptime('2015-09-01 00:00:00', "%Y-%m-%d %H:%M:%S")
    #    print stopdate
    #    https://www.hltv.org/results?offset=0&content=stats&matchType=Lan
        start = 0
    #    db['matchHistory'].delete_many({})
    #    if matchType!='all':
        matchType = 'Lan'
        while True:
    #        print matchType, start
    #        if matchType=='all':
    #            res = csgo.request('https://www.hltv.org/stats/matches?offset=%s'%start)
            res = self.request('https://www.hltv.org/stats/matches?offset=%s&matchType=%s'%(start, matchType))
    #        with open('tttt%s.txt'%matchType, 'w') as f:
    #            f.write(res.encode('utf-8'))
                
            matches = re.findall(r'href="/stats/matches/mapstatsid/(\d+)/.*?>(\d+/\d+/\d+)<', res, re.DOTALL|re.IGNORECASE)
            
            for matchId, date in matches:
                date = date[:-2]+'20'+date[-2:]
                date = datetime.datetime.strptime(date, '%d/%m/%Y')
    #            print date
    #            print matches, type(matchId), date
    #            return
                self.db.update('matchMapStatsHistory', {'maplinkId': matchId}, {'maplinkId': matchId, 'matchType': matchType, 'date': str(date).split('.')[0]})
#                db['matchMapStatsHistory'].update_one({'maplinkId': matchId}, {'$set': {'maplinkId': matchId, 'matchType': matchType, 'date': str(date).split('.')[0]}}, upsert=True)
    #        return
            lastDate = datetime.datetime.strptime(matches[-1][1], '%d/%m/%y')
            start += 50
    #        print lastDate
    #        break
    
    #        if not updateAll:
    #            if db['matchHistory'].find_one({'matchlinkId': matches[-1]}):
    #                if matchType!='Lan':
    #                    break
    #                else:
    ##                    print
    #                    start = 0
    #                    matchType = 'Online'
    #            print lastDate, matchType, start
    #            continue
                        
            if lastDate<=stopdate:
                if matchType!='Lan':
                    break
                else:
                    start = 0
                    matchType = 'Online'
                    
            print lastDate, start
            
    def getMatchistory(self, dat='2015-09-01 00:00:00'):#from results page
        #TODO GET ONLY MATCH_LINK_ID, MATCH_TYPE, MATCH_DATE
        print '#getMatchHistoryLinks and Lan or Online matchType'
    #    stopdate = datetime.datetime.strptime(sorted(list(db['teamsRating'].find()), key=lambda x: x['date'])[0]['date'], "%Y-%m-%d %H:%M:%S")
    #    if dat:
        stopdate = datetime.datetime.strptime(dat, "%Y-%m-%d %H:%M:%S")
    #    elif updateAll:
    #        stopdate = datetime.datetime.strptime('2015-09-01 00:00:00', "%Y-%m-%d %H:%M:%S")
        print stopdate
    #    https://www.hltv.org/results?offset=0&content=stats&matchType=Lan
        start = 0
    #    db['matchHistory'].delete_many({})
        matchType = 'Lan'
        while True:
    #        print matchType, start
            res = self.request('https://www.hltv.org/results?offset=%s&matchType=%s'%(start, matchType))
    #        with open('tttt%s.txt'%matchType, 'w') as f:
    #            f.write(res.encode('utf-8'))
                
            res = re.findall(r'>Results for (.+?) (\d+).*? (\d+)<(.*?)</a></div>\s*</div>', res, re.DOTALL|re.IGNORECASE)
            lastDate = datetime.datetime.strptime('%s %s %s'%res[-1][:-1], '%B %d %Y')
            print lastDate
    #        res = ''.join(res.split('Results for')[1:])
    #        matches = re.findall(r'result-con".*?href="/matches/(\d+)/', res, re.DOTALL|re.IGNORECASE)
            for datMatches in res:
                lclDate = datetime.datetime.strptime('%s %s %s'%datMatches[:-1], '%B %d %Y')
    #            print lclDate#, datMatches
                for matchId in re.findall(r'result-con".*?href="/matches/(\d+)/', datMatches[-1], re.DOTALL|re.IGNORECASE):
    #                print matchId#, type(matchId)
    #                return
        #            pass
                    self.db.update('matchHistory', {'matchlinkId': matchId},{'matchlinkId': matchId, 'matchType': matchType, 'date': str(lclDate).split('.')[0]})
#                    db['matchHistory'].update_one({'matchlinkId': matchId}, {'$set': {'matchlinkId': matchId, 'matchType': matchType, 'date': str(lclDate).split('.')[0]}}, upsert=True)
    #            return
            start += 100
    #        return
    #        if not updateAll:
    #            if db['matchHistory'].find_one({'matchlinkId': matches[-1]}):
    #                if matchType!='Lan':
    #                    break
    #                else:
    ##                    print
    #                    start = 0
    #                    matchType = 'Online'
    #            print lastDate, matchType, start
    #            continue
                        
            if lastDate<=stopdate:
                if matchType!='Lan':
                    break
                else:
                    start = 0
                    matchType = 'Online'
                    
            print lastDate, start
            
    
    def __getMapStats(self, teamId, teamUrlName):
        dct = {}
        try:
            res = self.request('https://www.hltv.org/stats/teams/maps/%s/%s'%(teamId, teamUrlName))
            mapStats = re.findall(r'map-pool-map-name">(\w+\d?)<.*?>(\d+) / (\d+) / (\d+)<', res, re.IGNORECASE|re.DOTALL)
            for i in mapStats:
                dct.update({'%sWin'%i[0]: int(i[1]), '%sDraw'%i[0]: int(i[2]), '%sLoose'%i[0]: int(i[3])})
        except:
            pass
        return dct
    
    def makeMapStats(self, date=''):
        #TODO GET TEAM MAP STATISTICS
        print '#makeMapStats',
        todayDate = str(datetime.datetime.today()).split('.')[0]
    #    db['mapStats'].delete_many({})
        if not date:
            teams = set([(i['teamAId'], i['teamAUrlName']) for i in self.db.find('fullMatchHistory')])
            teams = teams | set([(i['teamBId'], i['teamBUrlName']) for i in self.db.find('fullMatchHistory')])
        else:
            teams = set([(i['teamAId'], i['teamAUrlName']) for i in self.db.find('fullMatchHistory', {'date': {'$gte': date}})])
            teams = teams | set([(i['teamBId'], i['teamBUrlName']) for i in self.db.find('fullMatchHistory', {'date': {'$gte': date}})])
    #    if not date:
    #        teams = set([(i['teamAId'], i['teamAUrlName']) for i in db['fullMatchHistory'].find()])
    #        teams = teams | set([(i['teamBId'], i['teamBUrlName']) for i in db['fullMatchHistory'].find()])
    #    else:
    #        teams = set([(i['teamAId'], i['teamAUrlName']) for i in db['fullMatchHistory'].find({'date': {'$gte': date}})])
    #        teams = teams | set([(i['teamBId'], i['teamBUrlName']) for i in db['fullMatchHistory'].find({'date': {'$gte': date}})])
            
        print 'scan %s'%len(teams)
        for ind, (teamId, teamUrlName) in enumerate(teams):
            if not ind%20:
                print '%s/%s'%(ind, len(teams))
                
            teamMapStat = self.__getMapStats(teamId, teamUrlName)
            while not teamMapStat:
                print 'one more try'
                time.sleep(5)
                teamMapStat = self.__getMapStats(teamId, teamUrlName)
                
            teamMapStat.update({'teamId': teamId, 'teamUrlName': teamUrlName, 'date': todayDate})
            self.db.update('mapStats', {'teamId': teamId}, teamMapStat)
    #        db['mapStats'].update_one({'teamId': teamId}, {'$set': teamMapStat}, upsert=True)
            
    
    def parseMatchInfo(self, matchHistory, proxy={}):
        #TODO GET BASIC MATCH STATISTICS
#            print 123
#        import 
        try:
#            #https://www.hltv.org/matches/2313224/sabertooth-vs-utm-rog-masters-2017-vietnam-qualifier
#            res = csgo.request('https://www.hltv.org/matches/2313224/sabertooth-vs-utm-rog-masters-2017-vietnam-qualifier')
            res = self.request(base + '/matches/%s/asd'%matchHistory['matchlinkId'], proxy=proxy)
            if not res:
                with open('repeatMatchHistory.txt', 'a') as f:
                    f.write('%s\n'%str(matchHistory))
                return datetime.datetime.today()
#                returnList.append(matchHistory)
#            print res
#            print base + '/matches/%s/asd'%matchHistory['matchlinkId']
#            tm = time.time()
#            print tm
            hourMin, strDate = re.findall(r'data-unix=".*?">(.*?)<', res, re.DOTALL|re.IGNORECASE)[:2]#час и минуты начала матча
            strDate = ' '.join(re.findall(r'(\d+).*?of (\w+) (\d+)', strDate, re.DOTALL|re.IGNORECASE)[0])#день-месяц-год начала матча
            date = datetime.datetime.strptime('%s %s'%(strDate, hourMin), '%d %B %Y %H:%M')
            eventUrl = re.search(r'/events/(\d+)/', res).group(1)#id турнира
#            results = re.findall(r'"results".*?(\d+).*?(\d+).*?class="(\w+)">(\d+).*?class="(\w+)">(\d+).*?class="(\w+)">(\d+).*?class="(\w+)">(\d+)', res, re.DOTALL|re.IGNORECASE)
#            results = [[int(i[0]), int(i[1]), int(i[-3]), int(i[3]), int(i[5]), int(i[-1])] if i[2]=='t' else [int(i[0]), int(i[1]), int(i[3]), int(i[-3]), int(i[-1]), int(i[5])] for i in results ]
            results = re.findall(r'"results".*?(\d+).*?(\d+).*?class="(\w+)">(\d+).*?class="(\w+)">(\d+).*?class="(\w+)">(\d+).*?class="(\w+)">(\d+)<.*?(?:mapstatsid/(\d+)/.*?)?</div', res, re.DOTALL|re.IGNORECASE)
#            mapIds = 
#            print results
            results = [[int(i[0]), int(i[1]), int(i[-4]), int(i[3]), int(i[5]), int(i[-2]), i[-1]] if i[2]=='t' else [int(i[0]), int(i[1]), int(i[3]), int(i[-4]), int(i[-2]), int(i[5]), i[-1]] for i in results ]
#            
#            print results
#            return
            #results = [resAScore, resBScore, resACT, resAT, resBCT, resBT]
#            maps = re.findall(r'" played".*?class="mapname">(\w+)<', res, re.DOTALL|re.IGNORECASE)
            maps = re.findall(r'mapname">(\w+)<', res, re.DOTALL|re.IGNORECASE)
            bo = len(maps)
            maps = [i for i in maps if i.lower()!='default' and i.lower()!='tba']
#            print maps
#            print maps
#            return
            #карты которые были сыграны - например BO3, но сыграли 2 карты (команда А или В выйграла 2 карты) - на выходе 2 карты
            mapsDict = {val:results[ind] for ind, val in enumerate(maps[:len(results)])}#[:len(results)]
#            print mapsDict
#            print mapsDict, maps
            mapBox = re.search(r'Maps(.*?)<div class="mapholder">', res, re.DOTALL|re.IGNORECASE)
            if mapBox:
                mapBox = mapBox.groups()[0]
#                bo = re.search(r'Best of\s+(\d)', mapBox, re.DOTALL|re.IGNORECASE)
#                if bo:
#                    bo = int(bo.groups()[0])
            else:
#                bo = ''
                mapBox = ''
                
            matchFormat = mapBox.strip().replace('\n', ' ')
#            bo, matchFormat = re.search(r'padding preformatted-text">.*?Best of\s+(\d)(.*?)<div class="mapholder">', res, re.DOTALL|re.IGNORECASE).groups()#([0, 1])
#            bo = int(bo)
#            matchFormat = matchFormat.strip().replace('\n', ' ')
            matchFormatCat = catDefine(matchFormat)
#            print bo, matchFormat, matchFormatCat
            
            teamA, teamB = re.findall(r'href="/team/(\d+)/(.+?)">.*?alt="(.*?)"', res, re.DOTALL|re.IGNORECASE)[:2]
            matchResult = re.findall(r'class="(won|tie|lost)"', res, re.DOTALL|re.IGNORECASE)[:2]
#            print matchResult
#            print teamA, teamB
            teamA = [teamA[0], teamA[1], teamA[2], 0, [], []]#teamId, teamUrlName, teamShortName, mapQntToWinMatch, maps Picked, lineup
            teamB = [teamB[0], teamB[1], teamB[2], 0, [], []]#teamId, teamUrlName, teamShortName, mapQntToWinMatch, maps Picked, lineup
            
            
            prewin = False
            if bo==5:
                if '%s has 1-0 map advantage'%teamA[2] in matchFormat:
#                    if not bo:
#                        bo = 5
                    teamA[3] -= 1
                    prewin = True
                elif '%s has 1-0 map advantage'%teamB[2] in matchFormat:
#                    if not bo:
#                        bo = 5
                    teamB[3] -= 1
                    prewin = True
#                elif not bo:
#                    mapsLen = len(maps)
#                    if mapsLen>1:
#                        if mapsLen>3:
#                            bo = 5
#                        else:
#                            bo = 3
#                    else:
#                        bo = 1
                       
            if bo==2:
                teamA[3] = 1
                teamB[3] = 1
            else:
                teamA[3] += bo/2+1
                teamB[3] += bo/2+1
                        
            teamA[-2] = re.findall(r'%s picked (\w+\d?)'%re.escape(teamA[2]), matchFormat, re.DOTALL|re.IGNORECASE)
            teamB[-2] = re.findall(r'%s picked (\w+\d?)'%re.escape(teamB[2]), matchFormat, re.DOTALL|re.IGNORECASE)
            
            lineups = re.findall(r'lineup standard-box">(.*?)</table', res, re.DOTALL|re.IGNORECASE)
            for lineup in lineups:
                lineup = re.findall(r'href="/\w+/(\d+)/', lineup, re.DOTALL|re.IGNORECASE)
#                print lineup, len(lineup)
                if teamA[0]==lineup[0]:
                    teamA[-1] = lineup[1:6]
#                    print teamA[-1]
                elif teamB[0]==lineup[0]:
                    teamB[-1] = lineup[1:6]
#                    print teamB[-1]
                
#            print teamA, teamB
            playerStats = {}
#            print mapsDict, maps
            matchMapStats = re.findall(r'class="stats-content" id="([a-zA-Z]+)\d.*?">(.+?</table>)\s+</div>', res, re.DOTALL|re.IGNORECASE)#</div>
            ratingType = re.search(r'ratingDesc">(\d)\.', res, re.DOTALL|re.IGNORECASE)
            if ratingType:
                ratingType = int(ratingType.groups()[0])
            else:
                ratingType = 0
                
            statMaps = []
#            print ratingType
#            ratingType = 1
#            print len(matchMapStats)
#            return
            for mp, ps in matchMapStats:
#                print mp
                if mp=='Dust':
                    mp += '2'
                statMaps += [mp]
                playerStats[mp] = [[], []]
                tms = re.findall(r'class="table">(.*?)</table>', ps, re.DOTALL|re.IGNORECASE)
#                print len(re.findall(r'<tr class="">(.*?)</tr>', ps, re.DOTALL|re.IGNORECASE))
#                print len(tms)
#                print ps
                for ind, team in enumerate(tms):
                    plrs = re.findall(r'<tr class="">(.*?)</tr>', team, re.DOTALL|re.IGNORECASE)
                    for player in plrs:
                        if ratingType==2:#rating 2.0
                            playerStatsPerMap = re.findall(r'href="/player/(\d+)/.*?<td.*?>(.+?)<.*?<td.*?>([\+-]?\d+)<.*?<td.*?>(.+?)<.*?<td.*?>(.+?)\%<.*?<td.*?>(.+?)<', player, re.DOTALL|re.IGNORECASE)
                            if playerStatsPerMap:
                                playerLink, playerScore, playerScoreRes, playerADR, playerKAST, playerRating = playerStatsPerMap[0]
                            else:
                                playerStatsPerMap = re.findall(r'href="/player/(\d+)/.*?<td.*?>(.+?)<.*?<td.*?>([\+-]?\d+)<.*?<td.*?>(.+?)<.*?<td.*?>(.+?)<', player, re.DOTALL|re.IGNORECASE)
                                if playerStatsPerMap:
                                    if '%' in playerStatsPerMap[0][-2]:
                                        playerLink, playerScore, playerScoreRes, playerKAST, playerRating = playerStatsPerMap[0]
                                        playerKAST = playerKAST[:-1]
                                        playerADR = '-100'
                                    else:
                                        playerLink, playerScore, playerScoreRes, playerADR, playerRating = playerStatsPerMap[0]
                                        playerKAST = '-100'
                                else:
                                    playerStatsPerMap = re.findall(r'href="/player/(\d+)/.*?<td.*?>(.+?)<.*?<td.*?>([\+-]?\d+)<.*?<td.*?<td class="rating text-center">(.+?)<', player, re.DOTALL|re.IGNORECASE)
                                    if playerStatsPerMap:
                                        playerLink, playerScore, playerScoreRes, playerRating = playerStatsPerMap[0]
                                        playerADR = '-100'
                                        playerKAST = '-100'
                                    else:
                                        with open('dopCheck.txt', 'a') as f:
                                            f.write('%s\n'%str(matchHistory).encode('utf-8'))
                                        continue
                                    
                        elif ratingType==1:#rating 1.0
                            playerStatsPerMap = re.findall(r'href="/player/(\d+)/.*?<td.*?>(.+?)<.*?<td.*?>([\+-]?\d+)<.*?">(.+?)<', player, re.DOTALL|re.IGNORECASE)
                            if playerStatsPerMap:
                                playerLink, playerScore, playerScoreRes, playerRating = playerStatsPerMap[0]
                                playerADR = '-100'
                                playerKAST = '-100'
                            else:#no stats
                                with open('dopCheck.txt', 'a') as f:
                                    f.write('%s\n'%str(matchHistory).encode('utf-8'))
                                continue
                        else:#no stats
                            with open('dopCheck.txt', 'a') as f:
                                f.write('%s\n'%str(matchHistory).encode('utf-8'))
                            continue
                        
                        playerKill, playerDeath =  map(int, playerScore.split('-'))
                        playerScoreRes, playerADR, playerKAST, playerRating = int(playerScoreRes), round(float(playerADR), 3), round(float(playerKAST)/100.0, 3), float(playerRating)
                        playerStats[mp][ind] += [[playerLink, playerKill, playerDeath, playerScoreRes, playerADR, playerKAST, playerRating, ratingType]]
#                    
#                    print playerStats[mp][ind], len(playerStats[mp][ind])
                        

                
#            print playerStats.keys(), maps, len(mapsDict), len(playerStats['Mirage'])
#            for val in maps:#[:len(mapsDict)]
#                if val not in playerStats:#matchMapStats:
#                    mapsDict[val] += [-10, -10]
#                else:
##                    print zip(*playerStats[val][:5])[-2], zip(*playerStats[val][5:])[-2]
#                    mapsDict[val] += [round(np.mean(zip(*playerStats[val][0])[-2]), 2), round(np.mean(zip(*playerStats[val][1])[-2]), 2)]#среднее значение рейтинга игроков команды
                    
            teamRatingAvg = {}
#            print all([i in maps for i in statMaps])
            if not all([i in maps for i in statMaps]):
                for i in statMaps:
                    if len(maps)>=bo:
                        break
                    elif i not in maps:
                        maps += [i]
                
#            mapSet = list(set(mapsDict.keys()+playerStats.keys()))
#            print mapSet
            for val in maps:
                if val not in playerStats:#matchMapStats:
                    teamRatingAvg[val] = [-10, -10]
                else:
                    teamRatingAvg[val] = [round(np.mean(zip(*playerStats[val][0])[-2]), 2), round(np.mean(zip(*playerStats[val][1])[-2]), 2)]#среднее значение рейтинга игроков команды
                
#            print maps, bo, matchHistory['matchlinkId']
#            return 
            cnt = 0
            for mp in maps:#maps[:len(mapsDict)]:
#                print mp
                cnt += 1
                mappicker = 1 if mp in teamA[-2] else -1 if mp in teamB[-2] else 0 if mp in maps else 10#1 - teamA, -1 - teamB, 0 - was left, 10 - no info
                if mappicker==10:
                    if bo==1:
                        mappicker = 0
                    elif bo==3 and cnt==3:
                        mappicker = 0
                    elif bo==5 and not prewin and cnt==5:
                        mappicker = 0
                        
                
                #добавить число раундов на карте - OK
                #добавить сколько раундов было сыграно за каждую из сторон - OK
                updateDic = {'date': str(date), 'map': mp,
                               'mapPicker': mappicker, 'bo': bo, 'matchFormat': matchFormatCat, 'matchlinkId': matchHistory['matchlinkId'],
                               'teamAMapLeftToWin': teamA[3], 'teamBMapLeftToWin': teamB[3]}
                if mp in mapsDict:
                    lclDic = {'totalRounds': mapsDict[mp][0]+mapsDict[mp][1],
                               'teamCTRounds': mapsDict[mp][2]+mapsDict[mp][4], 'teamTRounds': mapsDict[mp][3]+mapsDict[mp][5]}
#                    if mapsDict[mp][-1]:
#                        lclDic.update({'maplinkId': mapsDict[mp][-1]})
                    updateDic.update(lclDic)
                
                flag = mp in playerStats
                for ind, team in enumerate((lambda: playerStats[mp] if flag else [teamA[-1], teamB[-1]])()):
                    if ind==0:
                        if mp in mapsDict:
                            lclDic = {'teamCTWin': mapsDict[mp][2], 'teamTWin': mapsDict[mp][3],
                                          'teamWinRounds': mapsDict[mp][0]}
#                            if mapsDict[mp][-1]:
#                                lclDic.update({'maplinkId': mapsDict[mp][-1]})
                            updateDic.update(lclDic)
                        
                        if flag:
                            tmp = zip(*playerStats[mp][ind])
                            teamAKill = sum(tmp[1])
                            teamADeath = sum(tmp[2])
                        else:
                            teamAKill = -10
                            teamADeath = -10
                        updateDic.update({'teamId': teamA[0], 'teamKill': teamAKill, 'teamDeath': teamADeath})
                    elif ind==1:
                        if mp in mapsDict:
                            lclDic = {'teamCTWin': mapsDict[mp][4], 'teamTWin': mapsDict[mp][5],
                                          'teamWinRounds': mapsDict[mp][1]}
#                            if mapsDict[mp][-1]:
#                                lclDic.update({'maplinkId': mapsDict[mp][-1]})
                            updateDic.update(lclDic)
                            
                        if flag:
                            tmp = zip(*playerStats[mp][ind])
                            teamBKill = sum(tmp[1])
                            teamBDeath = sum(tmp[2])
                        else:
                            teamBKill = -10
                            teamBDeath = -10
                        updateDic.update({'teamId': teamB[0], 'teamKill': teamBKill, 'teamDeath': teamBDeath})
                        
                    for ps in (lambda: playerStats[mp][ind] if flag else team)():#playerStats[mp])
                        if flag:
                            playerLink, playerKill, playerDeath, playerScoreRes, playerADR, playerKAST, playerRating, ratingType = ps
                            updateDic.update({'playerId': playerLink, 
                                             'kill': playerKill, 'death': playerDeath, 'scoreRes': playerScoreRes,
                                             'adr': playerADR, 'kast': playerKAST, 
                                             'mapRating': playerRating, 'ratingType': ratingType})
                        else:
                            playerLink = ps

                        self.db.update('players', {'playerId': playerLink, 'matchlinkId': matchHistory['matchlinkId'], 'map': mp}, updateDic)
#                            db['players'].update_one({'playerId': playerLink, 'matchlinkId': matchHistory['matchlinkId'], 'map': mp}, 
#                                                          {'$set': updateDic}, upsert=True)
#                    print updateDic, '\n'
                            
                if flag:
                    teamALineup = zip(*playerStats[mp][0])[0]
                    teamBLineup = zip(*playerStats[mp][1])[0]
                else:
                    teamALineup = list(set(teamA[-1]))
                    teamBLineup = list(set(teamB[-1]))
                    
#                print 'teamALineup', teamALineup
#                print 'teamBLineup', teamBLineup
#                print mp, mappicker, bo, matchFormatCat, teamA[3], teamB[3]
#                print mapsDict[mp]
                updateDic = {'map': mp, 'matchlinkId': matchHistory['matchlinkId'],
                           'teamAId': teamA[0], 'teamAUrlName': teamA[1],
                           'teamBId': teamB[0], 'teamBUrlName': teamB[1],
                           'mapPicker': mappicker, 'bo': bo, 'matchFormat': matchFormatCat,
                           'teamAMapLeftToWin': teamA[3], 'teamBMapLeftToWin': teamB[3],
                           'matchType': matchHistory['matchType'], 'date': str(date),
                           'teamALineup': teamALineup, 'teamBLineup': teamBLineup,
                           'teamAKill': teamAKill, 'teamBKill': teamBKill,
                           'teamADeath': teamADeath, 'teamBDeath': teamBDeath,
                           'eventLink': eventUrl, 'ratingType': ratingType,
                           'teamAResRating': teamRatingAvg[mp][0], 'teamBResRating': teamRatingAvg[mp][1]}
                
                if mp in mapsDict:
#                    print mp, mapsDict[mp]
                    lclDic = {'teamAResScore': mapsDict[mp][0], 'teamBResScore': mapsDict[mp][1],
                               'teamACTScore': mapsDict[mp][2], 'teamATScore': mapsDict[mp][3],
                               'teamBCTScore': mapsDict[mp][4], 'teamBTScore': mapsDict[mp][5]}
                    if mapsDict[mp][-1]:
                        lclDic.update({'maplinkId': mapsDict[mp][-1]})
                    updateDic.update(lclDic)
                
                
                self.db.update('fullMatchHistory', {'map': mp, 'matchlinkId': matchHistory['matchlinkId']}, updateDic)
#                    db['fullMatchHistory'].update_one({'map': mp, 'matchlinkId': matchHistory['matchlinkId']}, 
#                                                              {'$set': updateDic}, upsert=True)
#                if bo!=1:
                if mp in mapsDict:
                    if mapsDict[mp][0]>mapsDict[mp][1]:
                        teamA[3] -= 1
                    else:
                        teamB[3] -= 1
                elif teamAKill-teamADeath>teamBKill-teamBDeath:
                    teamA[3] -= 1
                elif teamAKill-teamADeath<teamBKill-teamBDeath:
                    teamB[3] -= 1
                elif teamRatingAvg[mp][0]>teamRatingAvg[mp][1]:
                    teamA[3] -= 1
                elif teamRatingAvg[mp][0]<teamRatingAvg[mp][1]:
                    teamB[3] -= 1
                elif teamAKill>teamBKill:
                    teamA[3] -= 1
                elif teamAKill<teamBKill:
                    teamB[3] -= 1
                    
#            print bo, teamB[3], teamA[3], list(set(teamA[-1])), list(set(teamB[-1]))
            if teamB[3]>0 and teamA[3]>0:
                updateDic = {'matchlinkId': matchHistory['matchlinkId'],
                           'teamAId': teamA[0], 'teamAUrlName': teamA[1],
                           'teamBId': teamB[0], 'teamBUrlName': teamB[1],
                           'bo': bo, 'matchFormat': matchFormatCat,
                           'matchType': matchHistory['matchType'], 'date': str(date),
                           'teamALineup': list(set(teamA[-1])), 'teamBLineup': list(set(teamB[-1])),
                           'eventLink': eventUrl}
                
                if matchResult[0]=='tie':
                    updateDic.update({'matchResult': 0})#tie
                elif matchResult[0]=='won':
                    updateDic.update({'matchResult': 1})#teamA won
                else:
                    updateDic.update({'matchResult': -1})#teamA lost
                
                self.db.update('fullMatchHistory', {'map': {'$exists': False}, 'matchlinkId': matchHistory['matchlinkId']}, updateDic)
#                    db['fullMatchHistory'].update_one({'map': {'$exists': False}, 'matchlinkId': matchHistory['matchlinkId']}, 
#                                                              {'$set': updateDic}, upsert=True)
#            print updateDic
#            return eventUrl
#            return ''
            return date
    
        except Exception as e:
            with open('matchHistory.txt', 'a') as f:
                f.write('%s %s\n'%(str(matchHistory), str(e)))
            print matchHistory, e
#            db['matchHistory'].delete_one({'matchlinkId': matchHistory['matchlinkId']})
            return datetime.datetime.today()
        
    def __roundHistoryProcess(self, teamAroundHistory, teamBroundHistory):
        #TODO __roundHistoryProcess
        if teamAroundHistory[0] in ['ct_win', 'stopwatch', 'bomb_defused'] or teamBroundHistory[0] in ['t_win', 'bomb_exploded']:
            # teamA half_0 on CT, teamB half_0 on T sides and teamA half_1 on T side and teamB half_1 on CT side
            teamA_half_0 = 1#ct
            teamB_half_0 = 0#t
            teamA_half_1 = 0#t
            teamB_half_1 = 1#ct
        else:
            # teamA half_0 on T, teamB half_0 on CT sides and teamA half_1 on CT side and teamB half_1 on T side
            teamA_half_0 = 0#t
            teamB_half_0 = 1#ct
            teamA_half_1 = 1#ct
            teamB_half_1 = 0#t
                
        if teamAroundHistory[0]=='emptyHistory':
            teamA_pistol_half_0 = -1
            teamB_pistol_half_0 = 1
        else:
            teamA_pistol_half_0 = 1
            teamB_pistol_half_0 = -1
        
        teamA_half_0_win_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamA_half_0_loose_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamB_half_0_win_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamB_half_0_loose_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamA_cnt = 0
        teamB_cnt = 0
        for _ in teamAroundHistory[:15]:
            if _=='emptyHistory':
                teamB_cnt += 1
                if teamA_cnt:
                    teamA_half_0_win_sequence[min(teamA_cnt, 8) - 1] += 1
                    teamB_half_0_loose_sequence[min(teamA_cnt, 8) - 1] += 1
                    teamA_cnt = 0
            else:
                teamA_cnt += 1
                if teamB_cnt:
                    teamB_half_0_win_sequence[min(teamB_cnt, 8) - 1] += 1
                    teamA_half_0_loose_sequence[min(teamB_cnt, 8) - 1] += 1
                    teamB_cnt = 0
        if teamA_cnt:
            teamA_half_0_win_sequence[min(teamA_cnt, 8) - 1] += 1
            teamB_half_0_loose_sequence[min(teamA_cnt, 8) - 1] += 1
        elif teamB_cnt:
            teamB_half_0_win_sequence[min(teamB_cnt, 8) - 1] += 1
            teamA_half_0_loose_sequence[min(teamB_cnt, 8) - 1] += 1
                    
        if teamAroundHistory[15]=='emptyHistory':
            teamA_pistol_half_1 = -1
            teamB_pistol_half_1 = 1
        else:
            teamA_pistol_half_1 = 1
            teamB_pistol_half_1 = -1   
            
        teamA_half_1_win_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamA_half_1_loose_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamB_half_1_win_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamB_half_1_loose_sequence = [0, 0, 0, 0, 0, 0, 0, 0]
        teamA_cnt = 0
        teamB_cnt = 0
        for _ in teamAroundHistory[15:]:
            if _=='emptyHistory':
                teamB_cnt += 1
                if teamA_cnt:
                    teamA_half_1_win_sequence[min(teamA_cnt, 8) - 1] += 1
                    teamB_half_1_loose_sequence[min(teamA_cnt, 8) - 1] += 1
                    teamA_cnt = 0
            else:
                teamA_cnt += 1
                if teamB_cnt:
                    teamB_half_1_win_sequence[min(teamB_cnt, 8) - 1] += 1
                    teamA_half_1_loose_sequence[min(teamB_cnt, 8) - 1] += 1
                    teamB_cnt = 0
                    
        if teamA_cnt:
            teamA_half_1_win_sequence[min(teamA_cnt, 8) - 1] += 1
            teamB_half_1_loose_sequence[min(teamA_cnt, 8) - 1] += 1
        elif teamB_cnt:
            teamB_half_1_win_sequence[min(teamB_cnt, 8) - 1] += 1
            teamA_half_1_loose_sequence[min(teamB_cnt, 8) - 1] += 1
                        
        teamA_defused, teamA_exploded = sum([_=='bomb_defused' for _ in teamAroundHistory]), sum([_=='bomb_exploded' for _ in teamAroundHistory])
        teamB_defused, teamB_exploded = sum([_=='bomb_defused' for _ in teamBroundHistory]), sum([_=='bomb_exploded' for _ in teamBroundHistory])
        teamArh = str([1 if _!='emptyHistory' else 0 for _ in teamAroundHistory])
        teamBrh = str([1 if _!='emptyHistory' else 0 for _ in teamBroundHistory])
        
        return [teamA_half_0, teamA_pistol_half_0, teamA_half_0_win_sequence, teamA_half_0_loose_sequence, teamA_half_1, teamA_pistol_half_1, teamA_half_1_win_sequence, teamA_half_1_loose_sequence, teamA_defused, teamA_exploded, teamArh], \
                [teamB_half_0, teamB_pistol_half_0, teamB_half_0_win_sequence, teamB_half_0_loose_sequence, teamB_half_1, teamB_pistol_half_1, teamB_half_1_win_sequence, teamB_half_1_loose_sequence, teamB_defused, teamB_exploded, teamBrh]
        
        
    def parseMapStats(self, matchHistory, proxy={}):
        #TODO GET ADVANCED MATCH STATISTICS
    #    global totalTm
    #    pass
        try:
    #        print matchHistory
    #        print base + '/stats/matches/mapstatsid/%s/dsf'%matchHistory['maplinkId']
            res = self.request(base + '/stats/matches/mapstatsid/%s/dsf'%matchHistory['maplinkId'], proxy=proxy)
            if not res:
                with open('repeatMatchMapHistory.txt', 'a') as f:
                    f.write('%s\n'%str(matchHistory))
                return
    #        with open('tttt.txt', 'w') as f:
    #            f.write(res)
            
            matchLinkId = re.search(r'"/matches/(\d+)', res, re.DOTALL|re.IGNORECASE).groups()[0]
    #        print matchLinkId
            #map search
    #        bo = int(re.search(r'Best of\s*(\d)', res, re.DOTALL|re.IGNORECASE).groups()[0])
            mp = re.search(r'>Map</span></div>\s+(\w+\d?)', res, re.DOTALL|re.IGNORECASE).groups()[0]
    #        print mp
            teamAId, teamBId = re.findall(r'/stats/teams/(\d+)/', res, re.DOTALL|re.IGNORECASE)[:2]
    #        tm = time.time()
            full = self.db.find('fullMatchHistory', {'matchlinkId': matchLinkId, 'map': mp}, multi=False)
#            print full
#            full = db['fullMatchHistory'].find_one({'matchlinkId': matchLinkId, 'map': mp})
            if full:
                if teamAId != full['teamAId'] and teamBId != full['teamBId']:
                    exchange = True
    #                print matchHistory, matchLinkId
                else:
                    exchange = False
            else:
                exchange = False
                
            
            roundHistory = re.findall(r'scoreboard/(.*?)\.svg', res, re.DOTALL|re.IGNORECASE)
#            print roundHistory
#            roundHistory = re.findall(r'round-history-half">(.+?)</div>', res, re.DOTALL|re.IGNORECASE)
#            print roundHistory, len(roundHistory)
#            assert False
            if roundHistory:
                roundHistory = [roundHistory[:30], roundHistory[30:]]
                while roundHistory[0][-1]=='emptyHistory' and roundHistory[1][-1]=='emptyHistory':
                    roundHistory[0].pop()
                    roundHistory[1].pop()
#                print roundHistory
                teamAidRes, teamBidRes = re.search(r'round-history-con.*?/logo/(\d+).*?/logo/(\d+)', res, re.DOTALL|re.IGNORECASE).groups()
    #            print teamAidRes, teamBidRes
                if teamAId!=teamAidRes or teamBidRes!=teamBId:
                    print 'id mismatch', teamAId, teamBId, '!=', teamAidRes, teamBidRes
                    with open('hahahahah.txt', 'a') as f:
                        f.write('%s\n'%str(matchHistory))
                    return
#                [teamA_half_0, teamA_pistol_half_0, teamA_half_0_win_sequence, teamA_half_0_loose_sequence, teamA_half_1, teamA_pistol_half_1, teamA_half_1_win_sequence, teamA_half_1_loose_sequence, teamA_defused, teamA_exploded], \
#                [teamB_half_0, teamB_pistol_half_0, teamB_half_0_win_sequence, teamB_half_0_loose_sequence, teamB_half_1, teamB_pistol_half_1, teamB_half_1_win_sequence, teamB_half_1_loose_sequence, teamB_defused, teamB_exploded]
                teamAroundHistory,  teamBroundHistory = self.__roundHistoryProcess(*roundHistory)
#                print teamAroundHistory
#                print teamAroundHistory, '\n', teamBroundHistory
                
#                teamArh = [sum([_=='bomb_defused' for _ in roundHistory[0]]),
#                           sum([_=='bomb_exploded' for _ in roundHistory[0]])]
#                teamBrh = [sum([_=='bomb_defused' for _ in roundHistory[1]]),
#                           sum([_=='bomb_exploded' for _ in roundHistory[1]])]
#                print re.findall('scoreboard/(.*?)\.svg', roundHistory[0], re.DOTALL|re.IGNORECASE), re.findall('scoreboard/(.*?)\.svg', roundHistory[1], re.DOTALL|re.IGNORECASE)
#                print re.findall('scoreboard/(.*?)\.svg', roundHistory[2], re.DOTALL|re.IGNORECASE), re.findall('scoreboard/(.*?)\.svg', roundHistory[3], re.DOTALL|re.IGNORECASE)
#                teamArh = [max(len(re.findall('(bomb_defused)', roundHistory[0], re.DOTALL|re.IGNORECASE)), len(re.findall('(bomb_defused)', roundHistory[1], re.DOTALL|re.IGNORECASE))),
#                           max(len(re.findall('(bomb_exploded)', roundHistory[0], re.DOTALL|re.IGNORECASE)), len(re.findall('(bomb_exploded)', roundHistory[1], re.DOTALL|re.IGNORECASE)))]
#                teamBrh = [max(len(re.findall('(bomb_defused)', roundHistory[2], re.DOTALL|re.IGNORECASE)), len(re.findall('(bomb_defused)', roundHistory[3], re.DOTALL|re.IGNORECASE))),
#                           max(len(re.findall('(bomb_exploded)', roundHistory[2], re.DOTALL|re.IGNORECASE)), len(re.findall('(bomb_exploded)', roundHistory[3], re.DOTALL|re.IGNORECASE)))]
            else:
                print 'no rh', matchHistory['maplinkId']
                teamAroundHistory = [0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 0, 0, '']
                teamBroundHistory = [0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 0, 0, [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], 0, 0, '']
                
    #        print exchange, teamArh, teamBrh
    #        totalTm += time.time()-tm
    #        print teamAId, teamBId
    #        firstKillStats = re.search(r'class="right">(\d+)\s*:\s*(\d+)</div>\s*<div class="bold">First kills</', res, re.DOTALL|re.IGNORECASE).groups()
    #        print firstKillStats
    #        clutches = re.search(r'class="right">(\d+)\s*:\s*(\d+)</div>\s*<div class="bold">Clutches won</', res, re.DOTALL|re.IGNORECASE).groups()
            i = re.search(r'div class="right"><span.*?(\d+).+?(\d+).*?class="(\w+)-.*?(\d+).*?class="(\w+)-.*?(\d+).*?class="(\w+)-.*?(\d+).*?class="(\w+)-.*?(\d+).*?class="right">(\d+)\s*:\s*(\d+)</div>\s*<div class="bold">Clut', res, re.DOTALL|re.IGNORECASE).groups()
            clutches = i[-2:]
            i = i[:-2]
    #        print i
            teamAres, teamBres = [[int(i[0]), int(i[-3]), int(i[3]), int(clutches[0])], [int(i[1]), int(i[5]), int(i[-1]), int(clutches[1])]] if i[2]=='t' else [[int(i[0]), int(i[3]), int(i[-3]), int(clutches[0])], [int(i[1]), int(i[-1]), int(i[5]), int(clutches[1])]]
    #        print teamAres, teamBres#, clutches
    #        print clutches
            playerStatsPerTeam = re.findall(r'<tbody>(.*?)</tbody>', res, re.DOTALL|re.IGNORECASE)
    #        print 'teamId, K(hs), A(F-r2.0), D, KAST, ADR, FK, FD, Rating, rType'
            playerStats = [[], []]
            ratingType = int(re.search(r'ratingDesc">(\d)', res, re.DOTALL|re.IGNORECASE).groups()[0])
    #        print ratingType 
    #        return
    #        tm = time.time()
            transformFunc = lambda x: round(float(x), 3) if '.' in x else -100 if '-' in x else int(x)
            for ind, team in enumerate(playerStatsPerTeam):
                if 'Flash assists' in team:
                    stats = re.findall(r'players/(\d+)/.*?kills">(.+?)<.*?gtSmartphone-only"> \((.+?)\).*?">(.+?)\s*<.*?\.">\((.*?)\).*?">(.+?)<.*?">(.+?)\%?<.*?adr">(.+?)<.*?title="(\d+).*?(\d+).*?">(\d\.\d+)<', team, re.DOTALL|re.IGNORECASE)
                else:
                    stats = re.findall(r'players/(\d+)/.*?kills">(.+?)<.*?gtSmartphone-only"> \((.+?)\).*?">(.+?)<.*?">(.+?)<.*?">(.+?)\%?<.*?adr">(.+?)<.*?title="(\d+).*?(\d+).*?">(\d\.\d+)<', team, re.DOTALL|re.IGNORECASE)
                    
    #            print len(stats)
                for player in stats:
                    if len(player)==10:#with no flash
                        playerStats[ind] += [list(player[:4])+['-1']+list(player[4:])+[ratingType]]
                    else:#with flash
                        playerStats[ind] += [list(player)+[ratingType]]
                    playerStats[ind][-1][1:-1] = map(transformFunc, playerStats[ind][-1][1:-1])
                    playerStats[ind][-1][6] = round(playerStats[ind][-1][6]/100.0, 3)#kast
    #                print playerStats[-1]#playerId, kill, headshort, asist, flash, death, KAST, ADR, firstKill in round, firstDeath in round, rating 1.0
            
    #        print time.time()-tm
    #        print len(stats[0])==10
            if exchange:
                playerStats = playerStats[::-1]
                
                teamAres, teamBres = teamBres, teamAres
#                tmpRes = copy.deepcopy(teamAres)
#                teamAres = copy.deepcopy(teamBres)
#                teamBres = tmpRes
                
                teamAroundHistory, teamBroundHistory = teamBroundHistory, teamAroundHistory
#                tmpRes1 = copy.deepcopy(teamArh)
#                teamArh = copy.deepcopy(teamBrh)
#                teamBrh = tmpRes1
                
                teamAId, teamBId = teamBId, teamAId
#                a = teamAId
#                teamAId = teamBId
#                teamBId = a
                
            teamAStats = zip(*playerStats[0])
            teamBStats = zip(*playerStats[1])
    #        print teamAId, teamBId
    #        print teamAStats
    #        print teamBStats
    #        print teamAStats[0]
    #        return
            
            #Среднее и сумма отражают одну и ту же информацию, так как среднее - сумма/5;
            #Поэтому рассчитывать среднее для параметров с суммой нет сымсла
            teamAK = sum(teamAStats[1])
            teamBK = sum(teamBStats[1])
            teamAhs = sum(teamAStats[2])
            teamBhs = sum(teamBStats[2])
            teamAA = sum(teamAStats[3])
            teamBA = sum(teamBStats[3])
            
            teamAF = sum(teamAStats[4])
            teamBF = sum(teamBStats[4])
            teamAD = sum(teamAStats[5])
            teamBD = sum(teamBStats[5])
            teamAKastAvg = round(np.mean(teamAStats[6]), 2)
            teamBKastAvg = round(np.mean(teamBStats[6]), 2)
            teamAAdrAvg = round(np.mean(teamAStats[7]), 2)
            teamBAdrAvg = round(np.mean(teamBStats[7]), 2)
            teamAFK = sum(teamAStats[8])
            teamBFK = sum(teamBStats[8])
            teamAFD = sum(teamAStats[9])
            teamBFD = sum(teamBStats[9])
            teamAResRating = round(np.mean(teamAStats[10]), 2)
            teamBResRating = round(np.mean(teamBStats[10]), 2)
            
            teamA_half_0, teamA_pistol_half_0, teamA_half_0_win_sequence, teamA_half_0_loose_sequence, teamA_half_1, teamA_pistol_half_1, teamA_half_1_win_sequence, teamA_half_1_loose_sequence, teamA_defused, teamA_exploded, teamA_rh = teamAroundHistory
            teamB_half_0, teamB_pistol_half_0, teamB_half_0_win_sequence, teamB_half_0_loose_sequence, teamB_half_1, teamB_pistol_half_1, teamB_half_1_win_sequence, teamB_half_1_loose_sequence, teamB_defused, teamB_exploded, teamB_rh = teamBroundHistory
    #        return 
    #        print len(playerStats)
    #        if len(playerStats)!=10:
    #            print '%s too low or too much players'%str(matchHistory)
    #            with open('mapHistory.txt', 'a') as f:
    #                f.write('%s too low or too much players \n'%str(matchHistory))
    #            return
            updateDic = {'map': mp, 'matchlinkId': matchLinkId, 'maplinkId': matchHistory['maplinkId']}
            for ind, team in enumerate(playerStats):
                if ind==0:
                    updateDic.update({'teamHs': teamAhs, 'teamA': teamAA, 'teamF': teamAF, 'teamKast': teamAKastAvg, 
                                      'teamAdr': teamAAdrAvg, 'teamResRating': teamAResRating, 'teamFirstK': teamAFK, 'teamFirstD': teamAFD,
                                      'teamDifuse': teamA_defused, 'teamExplode': teamA_exploded, 'team_rh': teamA_rh,
                                      'teamHalf_0': teamA_half_0, 'teamHalf_1': teamA_half_1,
                                      'teamPistol_half_0': teamA_pistol_half_0, 'teamPistol_half_1': teamA_pistol_half_1,
                                      })
                    for _ in range(len(teamA_half_0_win_sequence)):
                        updateDic.update({'teamHalf_0_winStrike_%s'%_: teamA_half_0_win_sequence[_], 
                                          'teamHalf_0_looseStrike_%s'%_: teamA_half_0_loose_sequence[_], 
                                          'teamHalf_1_winStrike_%s'%_: teamA_half_1_win_sequence[_], 
                                          'teamHalf_1_looseStrike_%s'%_: teamA_half_1_loose_sequence[_]})
                        
                else:
                    updateDic.update({'teamHs': teamBhs, 'teamA': teamBA, 'teamF': teamBF, 'teamKast': teamBKastAvg, 
                                      'teamAdr': teamBAdrAvg, 'teamResRating': teamBResRating, 'teamFirstK': teamBFK, 'teamFirstD': teamBFD,
                                      'teamDifuse': teamB_defused, 'teamExplode': teamB_exploded, 'team_rh': teamB_rh,
                                      'teamHalf_0': teamB_half_0, 'teamHalf_1': teamB_half_1,
                                      'teamPistol_half_0': teamB_pistol_half_0, 'teamPistol_half_1': teamB_pistol_half_1,
                                      })
                    for _ in range(len(teamB_half_0_win_sequence)):
                        updateDic.update({'teamHalf_0_winStrike_%s'%_: teamB_half_0_win_sequence[_], 
                                          'teamHalf_0_looseStrike_%s'%_: teamB_half_0_loose_sequence[_], 
                                          'teamHalf_1_winStrike_%s'%_: teamB_half_1_win_sequence[_], 
                                          'teamHalf_1_looseStrike_%s'%_: teamB_half_1_loose_sequence[_]})
                
                for val in playerStats[ind]:
    #                                            {'playerId': playerLink, 
    #                                             'kill': playerKill, 'death': playerDeath, 'scoreRes': playerScoreRes,
    #                                             'adr': playerADR, 'kast': playerKAST, 
    #                                             'mapRating': playerRating, 'ratingType': ratingType}
                    updateDic.update({'playerId': val[0], 'kill': val[1], 'hs': val[2], 'assist': val[3], 'flash':  val[4], 'death': val[5], 
                                     'kast': val[6], 'adr':val[7], 'fk': val[8], 'fd': val[9], 'mapRating': val[10], 'ratingType': val[11], 
                                     })
                    self.db.update('players', {'map': mp, 'matchlinkId': matchLinkId, 'playerId': val[0]}, updateDic)
#                    db['players'].update_one({'map': mp, 'matchlinkId': matchLinkId, 'playerId': val[0]}, {'$set': updateDic}, upsert=True)
    #            print {'playerId': val[0], 'kill': val[1], 'hs': val[2], 'assist': val[3], 'flash':  val[4], 'death': val[5], 
    #                                 'kast': val[6], 'adr':val[7], 'fk': val[8], 'fd': val[9], 'mapRating': val[10], 'ratingType': val[11], 
    #                                 }
#            print updateDic
    #        print sum(teamAStats[1]), sum(teamAStats[2]), sum(teamAStats[3]), sum(teamAStats[4]), sum(teamAStats[5]), round(np.mean(teamAStats[6]), 2), round(np.mean(teamAStats[7]), 2), sum(teamAStats[8]), sum(teamAStats[9]), round(np.mean(teamAStats[10]), 2)
    #        print sum(teamBStats[1]), sum(teamBStats[2]), sum(teamBStats[3]), sum(teamBStats[4]), sum(teamBStats[5]), round(np.mean(teamBStats[6]), 2), round(np.mean(teamBStats[7]), 2), sum(teamBStats[8]), sum(teamBStats[9]), round(np.mean(teamBStats[10]), 2)
            
            updateDic = {'map': mp, 'matchlinkId': matchLinkId, 'maplinkId': matchHistory['maplinkId'], 
                         'teamAKill': teamAK, 'teamBKill': teamBK, 'teamAhs': teamAhs, 'teamBhs': teamBhs, 'teamAa': teamAA, 'teamBa': teamBA,
                         'teamAFlash': teamAF, 'teamBFlash': teamBF, 'teamADeath': teamAD, 'teamBDeath': teamBD, 'teamAKastavg': teamAKastAvg, 'teamBKastavg': teamBKastAvg,
                         'teamAadravg': teamAAdrAvg, 'teamBadravg': teamBAdrAvg, 'teamAResRating': teamAResRating, 'teamBResRating': teamBResRating, 'ratingType': ratingType,
                         'teamALineup': list(teamAStats[0]), 'teamBLineup': list(teamBStats[0]), 'teamAFirstK': teamAFK, 'teamBFirstK': teamBFK,
                         'teamAClutch': teamAres[-1], 'teamBClutch': teamBres[-1], 'teamAId': teamAId, 'teamBId': teamBId,
                         'teamADifuse': teamA_defused, 'teamAExplode': teamA_exploded, 'teamBDifuse': teamB_defused, 'teamBExplode': teamB_exploded, 'teamA_rh': teamA_rh, 'teamB_rh': teamB_rh,
                         'teamAHalf_0': teamA_half_0, 'teamAHalf_1': teamA_half_1, 'teamBHalf_0': teamB_half_0, 'teamBHalf_1': teamB_half_1,
                         'teamAPistol_half_0': teamA_pistol_half_0, 'teamAPistol_half_1': teamA_pistol_half_1, 'teamBPistol_half_0': teamB_pistol_half_0, 'teamBPistol_half_1': teamB_pistol_half_1,
                         }
            for _ in range(len(teamB_half_0_win_sequence)):
                updateDic.update({'teamAHalf_0_winStrike_%s'%_: teamA_half_0_win_sequence[_], 
                                  'teamAHalf_0_looseStrike_%s'%_: teamA_half_0_loose_sequence[_], 
                                  'teamAHalf_1_winStrike_%s'%_: teamA_half_1_win_sequence[_], 
                                  'teamAHalf_1_looseStrike_%s'%_: teamA_half_1_loose_sequence[_],
                                  'teamBHalf_0_winStrike_%s'%_: teamB_half_0_win_sequence[_], 
                                  'teamBHalf_0_looseStrike_%s'%_: teamB_half_0_loose_sequence[_], 
                                  'teamBHalf_1_winStrike_%s'%_: teamB_half_1_win_sequence[_], 
                                  'teamBHalf_1_looseStrike_%s'%_: teamB_half_1_loose_sequence[_]})
#            print updateDic
    #        print teamArh, teamBrh
            res = self.db.update('fullMatchHistory', {'map': mp, 'matchlinkId': matchLinkId}, updateDic)
#            res = db['fullMatchHistory'].update_one({'map': mp, 'matchlinkId': matchLinkId}, {'$set': updateDic}, upsert=True)
    #        
    #        if not res.matched_count:
    #            with open('missingMatchlinkId.txt', 'a') as f:
    #                f.write('%s\n'%matchLinkId)
                
        except Exception as e:
            with open('mapHistory.txt', 'a') as f:
                f.write('%s %s\n'%(str(matchHistory), str(e)))
            print matchHistory, e
    ##        db['matchHistory'].delete_one({'matchlinkId': matchHistory['matchlinkId']})
    ##        return datetime.datetime.today()
    
    def parseMapEconomy(self, matchHistory, proxy={}):
        #TODO parseMapEconomy
        try:
    #        print matchHistory
    #        print base + '/stats/matches/economy/mapstatsid/%s/dfg'%matchHistory['maplinkId']
            res = self.request(base + '/stats/matches/economy/mapstatsid/%s/dsf'%matchHistory['maplinkId'], proxy=proxy)
            if not res:
                with open('repeatMatchMapEconomyHistory.txt', 'a') as f:
                    f.write('%s\n'%str(matchHistory))
                return 0
            
            if 'Economy not available' in res:
                print 'Economy not available for mapstatid: %s'%matchHistory['maplinkId']
                return -1
            
            teamAId, teamBId = re.findall(r'"team-categories".*?logo/(\d+)"', res, re.DOTALL|re.IGNORECASE)[:2]
#            print teamAId, teamBId
            
            full = self.db.find('fullMatchHistory', {'maplinkId':matchHistory['maplinkId']}, multi=False)
            
            if full:
                if teamAId != full['teamAId'] and teamBId != full['teamBId']:
                    exchange = True
    #                print matchHistory, matchLinkId
                else:
                    exchange = False
            else:
                exchange = False

#            print economyStats
#            
#            for ind, val in enumerate(economyStats):
#                economyStats[ind] = map(lambda x: map(int, x), re.findall(r'"Played">(.*?)<.*?"Won"> \((.*?)\)<', economyStats[ind][1], re.DOTALL|re.IGNORECASE))
                
            economyHistory = map(lambda x: [int(x[0]), 0 if x[1]=='Loss' else 1], re.findall(r'value: (\d+).*?(Loss|Win)', res, re.DOTALL|re.IGNORECASE))
            teamAeconomyHistory_half_0 = economyHistory[:15]
            teamBeconomyHistory_half_0 = economyHistory[15:30]
            teamAeconomyHistory_half_1 = economyHistory[30:30+(len(economyHistory)-30)/2]
            teamBeconomyHistory_half_1 = economyHistory[30+(len(economyHistory)-30)/2:]
#            print matchHistory['maplinkId']
#            print teamAeconomyHistory_half_0
#            print teamAeconomyHistory_half_1
#            print teamBeconomyHistory_half_0
#            print teamBeconomyHistory_half_1
#            print teamAeconomyHistory_half_0, teamBeconomyHistory_half_0, teamAeconomyHistory_half_1, teamBeconomyHistory_half_1
            
            economyStatsPerHalf = [  
                                     [[j[0]<=5000, (j[0]<=5000)*j[1], 
                                    j[0]>5000 and j[0]<=10000, (j[0]>5000 and j[0]<=10000)*j[1], 
                                    j[0]>10000 and j[0]<=15000, (j[0]>10000 and j[0]<=15000)*j[1], 
                                    j[0]>15000 and j[0]<=20000, (j[0]>15000 and j[0]<=20000)*j[1], 
                                    j[0]>20000, (j[0]>20000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]<=5000, (j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]<=5000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>5000 and teamBeconomyHistory_half_0[ind+1][0]<=10000, (j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>5000 and teamBeconomyHistory_half_0[ind+1][0]<=10000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>10000 and teamBeconomyHistory_half_0[ind+1][0]<=15000, (j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>10000 and teamBeconomyHistory_half_0[ind+1][0]<=15000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>15000 and teamBeconomyHistory_half_0[ind+1][0]<=20000, (j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>15000 and teamBeconomyHistory_half_0[ind+1][0]<=20000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>20000, (j[0]>20000 and teamBeconomyHistory_half_0[ind+1][0]>20000)*j[1],
                                    ] for ind, j in enumerate(teamAeconomyHistory_half_0[1:])],
                                       
                                     [[j[0]<=5000, (j[0]<=5000)*j[1], 
                                    j[0]>5000 and j[0]<=10000, (j[0]>5000 and j[0]<=10000)*j[1], 
                                    j[0]>10000 and j[0]<=15000, (j[0]>10000 and j[0]<=15000)*j[1], 
                                    j[0]>15000 and j[0]<=20000, (j[0]>15000 and j[0]<=20000)*j[1], 
                                    j[0]>20000, (j[0]>20000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]<=5000, (j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]<=5000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>5000 and teamBeconomyHistory_half_1[ind+1][0]<=10000, (j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>5000 and teamBeconomyHistory_half_1[ind+1][0]<=10000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>10000 and teamBeconomyHistory_half_1[ind+1][0]<=15000, (j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>10000 and teamBeconomyHistory_half_1[ind+1][0]<=15000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>15000 and teamBeconomyHistory_half_1[ind+1][0]<=20000, (j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>15000 and teamBeconomyHistory_half_1[ind+1][0]<=20000)*j[1],
                                    j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>20000, (j[0]>20000 and teamBeconomyHistory_half_1[ind+1][0]>20000)*j[1]
                                    ] for ind, j in enumerate(teamAeconomyHistory_half_1[1:])],
                                       
                                     [[j[0]<=5000, (j[0]<=5000)*j[1], 
                                    j[0]>5000 and j[0]<=10000, (j[0]>5000 and j[0]<=10000)*j[1], 
                                    j[0]>10000 and j[0]<=15000, (j[0]>10000 and j[0]<=15000)*j[1], 
                                    j[0]>15000 and j[0]<=20000, (j[0]>15000 and j[0]<=20000)*j[1], 
                                    j[0]>20000, (j[0]>20000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]<=5000, (j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]<=5000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>5000 and teamAeconomyHistory_half_0[ind+1][0]<=10000, (j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>5000 and teamAeconomyHistory_half_0[ind+1][0]<=10000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>10000 and teamAeconomyHistory_half_0[ind+1][0]<=15000, (j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>10000 and teamAeconomyHistory_half_0[ind+1][0]<=15000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>15000 and teamAeconomyHistory_half_0[ind+1][0]<=20000, (j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>15000 and teamAeconomyHistory_half_0[ind+1][0]<=20000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>20000, (j[0]>20000 and teamAeconomyHistory_half_0[ind+1][0]>20000)*j[1]
                                    ] for ind, j in enumerate(teamBeconomyHistory_half_0[1:])],
                                     
                                     [[j[0]<=5000, (j[0]<=5000)*j[1], 
                                    j[0]>5000 and j[0]<=10000, (j[0]>5000 and j[0]<=10000)*j[1], 
                                    j[0]>10000 and j[0]<=15000, (j[0]>10000 and j[0]<=15000)*j[1], 
                                    j[0]>15000 and j[0]<=20000, (j[0]>15000 and j[0]<=20000)*j[1], 
                                    j[0]>20000, (j[0]>20000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]<=5000, (j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]<=5000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>5000 and teamAeconomyHistory_half_1[ind+1][0]<=10000, (j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>5000 and teamAeconomyHistory_half_1[ind+1][0]<=10000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>10000 and teamAeconomyHistory_half_1[ind+1][0]<=15000, (j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>10000 and teamAeconomyHistory_half_1[ind+1][0]<=15000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>15000 and teamAeconomyHistory_half_1[ind+1][0]<=20000, (j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>15000 and teamAeconomyHistory_half_1[ind+1][0]<=20000)*j[1],
                                    j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>20000, (j[0]>20000 and teamAeconomyHistory_half_1[ind+1][0]>20000)*j[1]
                                    ] for ind, j in enumerate(teamBeconomyHistory_half_1[1:])]
                                ]
#            for i in economyStatsPerHalf:
#                print i
#            print economyStatsPerHalf
            economyStatsPerHalf = [list(np.sum(j, axis=0)) if j else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for j in economyStatsPerHalf]
#            for i in economyStatsPerHalf:
#                print i
#            print economyStatsPerHalf
#            print teamAId, teamBId, economyStatsPerHalf
#            assert False
            
            if exchange:
                
                economyStatsPerHalf = economyStatsPerHalf[2:] + economyStatsPerHalf[:2]
                teamAId, teamBId = teamBId, teamAId
#            print economyStatsPerHalf
                
            updateDic = {'maplinkId': matchHistory['maplinkId']}
            for ind0, j in enumerate(['A', 'B']):
                for ind1, __ in enumerate(['half_0_', 'half_1_', '']):
                    for i in range(5):
                        if not __:
                            updateDic.update({'team%s_%scat%s_played'%(j, __, i): economyStatsPerHalf[ind0*2][i*2]+economyStatsPerHalf[ind0*2+1][i*2], 
                                              'team%s_%scat%s_win'%(j, __, i): economyStatsPerHalf[ind0*2][i*2+1]+economyStatsPerHalf[ind0*2+1][i*2+1]})
                        else:
                            updateDic.update({'team%s_%scat%s_played'%(j, __, i): economyStatsPerHalf[ind0*2+ind1][i*2], 
                                              'team%s_%scat%s_win'%(j, __, i): economyStatsPerHalf[ind0*2+ind1][i*2+1]})
                    for i in range(5):
                        if not __:
                            updateDic.update({'team%s_%scat5VScat%s_played'%(j, __, i): economyStatsPerHalf[ind0*2][10+i*2]+economyStatsPerHalf[ind0*2+1][10+i*2], 
                                              'team%s_%scat5VScat%s_win'%(j, __, i): economyStatsPerHalf[ind0*2][10+i*2+1]+economyStatsPerHalf[ind0*2+1][10+i*2+1]})
                        else:
                            updateDic.update({'team%s_%scat5VScat%s_played'%(j, __, i): economyStatsPerHalf[ind0*2+ind1][10+i*2], 
                                              'team%s_%scat5VScat%s_win'%(j, __, i): economyStatsPerHalf[ind0*2+ind1][10+i*2+1]})
            
            
#            print updateDic
            res = self.db.update('fullMatchHistory', {'maplinkId': matchHistory['maplinkId']}, updateDic)
#            res = db['fullMatchHistory'].update_one({'maplinkId': matchHistory['maplinkId']}, {'$set': updateDic}, upsert=True)
#            
#            if not res.matched_count:
#                with open('missingMatchlinkId.txt', 'a') as f:
#                    f.write('%s\n'%matchLinkId)
                
            return 1
        
        except Exception as e:
            with open('mapEconomyHistory.txt', 'a') as f:
                f.write('%s %s\n'%(str(matchHistory), str(e)))
            print matchHistory, e
            return 0
            
            
    def fullFillMatchistory(self, date=''):
        #TODO fullFillMatchistory
        print '#fullFillMatchistory'
        mindate = datetime.datetime.today()
        with open('repeatMatchHistory.txt', 'w') as f:
            pass
        for i in range(2):
            if i:
                with open('repeatMatchHistory.txt', 'r') as f:
                    alldocs = map(lambda x: ast.literal_eval(x.strip()) if x.strip() else {}, f.readlines())
            elif date:
    #            date = 
                allMatched = set(self.db.find('matchHistory', {'date': {'$gte': date}}, distinct_field='matchlinkId'))
                filledLst = set(self.db.find('fullMatchHistory', {'date': {'$gte': date}}, distinct_field='matchlinkId'))
                findMatch = list(allMatched-filledLst)
                if findMatch:
                    print 'parseMatchInfo', len(findMatch), 'rows'
                    alldocs = list(self.db.find('matchHistory', {'matchlinkId': {'$in' : findMatch}}, {'matchlinkId': 1, 'matchType': 1}))
                else:
                    print 'there is nothing to parse in matchInfo'
                    return str(mindate)
            else:
                alldocs = list(self.db.find('matchHistory', {}, {'matchlinkId': 1, 'matchType': 1}))
#            if date:
#    #            date = 
#                allMatched = set(db['matchHistory'].find({'date': {'$gte': date}}).distinct('matchlinkId'))
#                filledLst = set(db['fullMatchHistory'].find({'date': {'$gte': date}}).distinct('matchlinkId'))
#                findMatch = list(allMatched-filledLst)
#                if findMatch:
#                    print 'parseMatchInfo', len(findMatch), 'rows'
#                    alldocs = list(db['matchHistory'].find({'matchlinkId': {'$in' : findMatch}}, {'matchlinkId': 1, 'matchType': 1}))
#                else:
#                    print 'there is nothing to parse in matchInfo'
#                    return str(mindate)
#            else:
#                alldocs = list(db['matchHistory'].find({}, {'matchlinkId': 1, 'matchType': 1}))
                
            
        #    return
            
            for number, matchHistory in enumerate(alldocs):
        #        print matchHistory
                if not number%25:
                    print '%s/%s'%(number, len(alldocs))
                    
                localdate = self.parseMatchInfo(matchHistory)
#                assert False
                if localdate<mindate:
                    mindate = localdate
        #        break
    
        return str(mindate)
    
        
    
    def fullFillMapStatsHistory(self, date=''):
        #TODO fullFillMapStatsHistory
        print '#fullFillMapStatsHistory'
        mindate = datetime.datetime.today()
        with open('repeatMatchMapHistory.txt', 'w') as f:
            pass
        for i in range(2):
            if i:
                with open('repeatMatchMapHistory.txt', 'r') as f:
                    alldocs = map(lambda x: ast.literal_eval(x.strip()) if x.strip() else {}, f.readlines())
            elif date:
                allMatched = set(self.db.find('matchMapStatsHistory', {'date': {'$gte': date}}, distinct_field='maplinkId'))
    #            print len(allMatched)
                filledLst = set(self.db.find('fullMatchHistory', {'date': {'$gte': date}}, distinct_field='maplinkId'))
        #        allMatched = set([i['maplinkId'] for i in alldocs])
                findMatch = list(allMatched-filledLst)
                if findMatch:
                    print 'parseMapStats', len(findMatch), 'rows'
                    alldocs = list(self.db.find('matchMapStatsHistory', {'maplinkId': {'$in' : findMatch}}, {'maplinkId': 1, 'matchType': 1}))
                else:
                    print 'there is nothing to parse in mapStats'
                    break
    #                return str(mindate)
            else:
                alldocs = list(self.db.find('matchMapStatsHistory', {}, {'maplinkId': 1, 'matchType': 1}))
#            elif date:
#                allMatched = set(db['matchMapStatsHistory'].find({'date': {'$gte': date}}).distinct('maplinkId'))
#    #            print len(allMatched)
#                filledLst = set(db['fullMatchHistory'].find({'date': {'$gte': date}}).distinct('maplinkId'))
#        #        allMatched = set([i['maplinkId'] for i in alldocs])
#                findMatch = list(allMatched-filledLst)
#                if findMatch:
#                    print 'parseMapStats', len(findMatch), 'rows'
#                    alldocs = list(db['matchMapStatsHistory'].find({'maplinkId': {'$in' : findMatch}}, {'maplinkId': 1, 'matchType': 1}))
#                else:
#                    print 'there is nothing to parse in mapStats'
#                    break
#    #                return str(mindate)
#            else:
#                alldocs = list(db['matchMapStatsHistory'].find({}, {'maplinkId': 1, 'matchType': 1}))
                
            
            
            for number, matchHistory in enumerate(alldocs):
        #        if number<29200:
        #            continue
                if not number%25:
                    print '%s/%s'%(number, len(alldocs))
                    
                self.parseMapStats(matchHistory)
#                assert False
                
                
        res = list(self.db.find('fullMatchHistory', {'teamAa': {'$exists': False}, 'map': {"$exists": True}, 'date': {'$gte': str(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=30)).split('.')[0]}, 'teamATScore': {'$gte': 0}, 'maplinkId': {'$exists': True}, 'ratingType': {'$gt': 0}}, 
                                                     {'_id': 0, 'maplinkId': 1}))
#        res = list(db['fullMatchHistory'].find({'teamAa': {'$exists': False}, 'map': {"$exists": True}, 'date': {'$gte': str(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=30)).split('.')[0]}, 'teamATScore': {'$gte': 0}, 'maplinkId': {'$exists': True}, 'ratingType': {'$gt': 0}}, {'_id': 0, 'maplinkId': 1}))    
        print len(res), str(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=30)).split('.')[0]
        for number, raw in enumerate(res):
            if not number%25:
                print '%s/%s'%(number, len(res))
            self.parseMapStats({'maplinkId': raw['maplinkId']})
    
        return str(mindate)
    
    def fullFillEconomyMapHistory(self, date=''):
        #TODO fullFillMapStatsHistory
        print '#fullFillMapStatsHistory'
        mindate = datetime.datetime.today()
        with open('repeatMapEconomyHistory.txt', 'w') as f:
            pass
        for i in range(2):
            if i:
                with open('repeatMapEconomyHistory.txt', 'r') as f:
                    alldocs = map(lambda x: ast.literal_eval(x.strip()) if x.strip() else {}, f.readlines())
            elif date:
                allMatched = set(self.db.find('matchMapStatsHistory', {'date': {'$gte': date}}, distinct_field='maplinkId'))
    #            print len(allMatched)
                filledLst = set(self.db.find('fullMatchHistory', {'date': {'$gte': date}, 'teamA_half_0_cat0_played': {'$exists': False}}, distinct_field='maplinkId'))
        #        allMatched = set([i['maplinkId'] for i in alldocs])
                findMatch = list(allMatched-filledLst)
                if findMatch:
                    print 'parseMapStats', len(findMatch), 'rows'
                    alldocs = list(self.db.find('matchMapStatsHistory', {'maplinkId': {'$in' : findMatch}}, sort=[('date', -1)]))
                else:
                    print 'there is nothing to parse in mapStats'
                    break
    #                return str(mindate)
            else:
                alldocs = list(self.db.find('matchMapStatsHistory', {}, sort=[('date', -1)]))
#            elif date:
#                allMatched = set(db['matchMapStatsHistory'].find({'date': {'$gte': date}}).distinct('maplinkId'))
#    #            print len(allMatched)
#                filledLst = set(db['fullMatchHistory'].find({'date': {'$gte': date}}).distinct('maplinkId'))
#        #        allMatched = set([i['maplinkId'] for i in alldocs])
#                findMatch = list(allMatched-filledLst)
#                if findMatch:
#                    print 'parseMapStats', len(findMatch), 'rows'
#                    alldocs = list(db['matchMapStatsHistory'].find({'maplinkId': {'$in' : findMatch}}, {'maplinkId': 1, 'matchType': 1}))
#                else:
#                    print 'there is nothing to parse in mapStats'
#                    break
#    #                return str(mindate)
#            else:
#                alldocs = list(db['matchMapStatsHistory'].find({}, {'maplinkId': 1, 'matchType': 1}))
                
            
            cnt = 0
            for number, matchHistory in enumerate(alldocs):
        #        if number<29200:
        #            continue
                if not number%25:
                    print '%s/%s'%(number, len(alldocs)), matchHistory['date']
                    
                res = self.parseMapEconomy(matchHistory)
                if res==-1 and i==0:
                    cnt += 1
                    if cnt%10==0:
                        print cnt, matchHistory['date']
                    if cnt>25:
                        break
                else:
                    cnt = 0
#                assert False
                
                
#        res = list(self.db.find('fullMatchHistory', {'teamAa': {'$exists': False}, 'map': {"$exists": True}, 'date': {'$gte': str(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=30)).split('.')[0]}, 'teamATScore': {'$gte': 0}, 'maplinkId': {'$exists': True}, 'ratingType': {'$gt': 0}}, 
#                                                     {'_id': 0, 'maplinkId': 1}))
##        res = list(db['fullMatchHistory'].find({'teamAa': {'$exists': False}, 'map': {"$exists": True}, 'date': {'$gte': str(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=30)).split('.')[0]}, 'teamATScore': {'$gte': 0}, 'maplinkId': {'$exists': True}, 'ratingType': {'$gt': 0}}, {'_id': 0, 'maplinkId': 1}))    
#        print len(res), str(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(days=30)).split('.')[0]
#        for number, raw in enumerate(res):
#            if not number%25:
#                print '%s/%s'%(number, len(res))
#            self.parseMapStats({'maplinkId': raw['maplinkId']})
    
        return str(mindate)
    
        
#obj = HistoryMatchStats()
#for index, maplinkId in enumerate(obj.db.find('fullMatchHistory', {'maplinkId': {'$exists': True}}, distinct_field='maplinkId')):
#    if index%250==0:
#        print index
#    obj.parseMapStats({'maplinkId': maplinkId}, proxy={})

#for index, maplinkId in enumerate(obj.db.find('fullMatchHistory', {'teamA_half_0_cat0_played': {'$exists': True}, 'maplinkId': {'$exists': True}}, distinct_field='maplinkId')):
#    if index%250==0:
#        print index
#    obj.parseMapEconomy({'maplinkId': maplinkId}, proxy={})    

#{ "teamAHalf_0" : {"$exists" : 1}}
#{ "maplinkId" : {"$exists" : 1}}
#parser = Parser(db)
#
#obj.parseMapStats({'maplinkId': '62883'}, proxy={})
#obj.db.close()
#db.close()
