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


mapDic = {'inf': 'Inferno', 'nuke': 'Nuke', 'ovp': 'Overpass', 'd2': 'Dust2', 'trn': 'Train', 'mrg': 'Mirage', 'cch': 'Cache', 'cbl': 'Cobblestone', '-': '-',
                  'Inferno': 'Inferno', 'Nuke': 'Nuke', 'Overpass': 'Overpass', 'Dust2': 'Dust2', 'Train': 'Train', 'Mirage': 'Mirage', 'Cache': 'Cache', 'Cobblestone': 'Cobblestone', 'TBA': 'TBA'}
#TODO PARSER
class UpcomingMatchStats(Parser):
    def __init__(self, db=DataBase()):
        Parser.__init__(self, db)
        self.db = db# mongodb by default
            
    def getHLTVUpcoming(self, matchType=''): #get only date, matchId, matchType, bo
        self.db.delete('upcomingHLTVMatches', {'date': {'$lte': str(datetime.datetime.today()-datetime.timedelta(days=1)).split()[0]}})
#        db['upcomingHLTVMatches'].delete_many({'date': {'$lte': str(datetime.datetime.today()-datetime.timedelta(days=1)).split()[0]}})
        if matchType:#ONLINE
            res = self.request('https://www.hltv.org/matches?eventType=ONLINE')
            matchType = 'Online'
        else:
            res = self.request('https://www.hltv.org/matches?eventType=MAJOR&eventType=INTLLAN&eventType=REGIONALLAN&eventType=LOCALLAN')
            matchType = 'LAN'
            
        matchesPerDay = re.findall(r'class="match-day">(.*?)</a></div>', res, re.DOTALL|re.IGNORECASE)
        for matchPerDay in matchesPerDay:
            date = re.search(r'"standard-headline">(.*?)<', matchPerDay, re.DOTALL|re.IGNORECASE).group(1)
    #        print date,
            matches = re.findall(r'href="/matches/(\d+).*?data-unix="\d+">(.*?)<.*?map-text">(.*?)<', matchPerDay, re.DOTALL|re.IGNORECASE)
    #        print len(matches)
            for match in matches:
                #match - [matchId, matchTime, mapName if bo1 else bo#]
                bo = int(match[-1][-1]) if 'bo' in match[-1] else 1
#                print match[:2], bo
                self.db.update('upcomingHLTVMatches', {'matchId': match[0]}, {'matchId': match[0], 'date': str(datetime.datetime.strptime('%s %s'%(date, match[1]), '%Y-%m-%d %H:%M')), 'matchType': matchType, 'bo': bo})
#                db['upcomingHLTVMatches'].update_one({'matchId': match[0]}, {'$set': {'matchId': match[0], 'date': str(datetime.datetime.strptime('%s %s'%(date, match[1]), '%Y-%m-%d %H:%M')), 'matchType': matchType, 'bo': bo}}, upsert=True)
       
        
    
    def getHLTVLive(self):
        res = self.request('https://www.hltv.org/matches')
        liveMatches = re.findall(r'live-match"(.*?)</a></div>', res, re.DOTALL|re.IGNORECASE)
        matches = []
        for liveMatch in liveMatches:
            #liveMatch - [matchId, bo, maps, teamAId, teamAName, teamBId, teamBName]
            matchId, bo, maps, teamAId, teamAName, teamBId, teamBName = re.search(r'href="/matches/(\d+).*?bestof">Best of (\d+)<.*?"map(.*?)</tr.*?logo/(\d+).*?title="(.*?)".*?logo/(\d+).*?title="(.*?)"', liveMatch, re.DOTALL|re.IGNORECASE).groups()
            maps = re.findall(r'">(.*?)</', maps, re.IGNORECASE|re.DOTALL)
            maps = [mapDic.get(i, 'TBA') for i in maps if i!='Maps']
            matches.append([matchId, bo, maps, teamAId, teamAName, teamBId, teamBName])
        
        return matches
            
    def parseLiveMatch(self, matchId, teamALogo, teamBLogo):
        # return playing Maps and mapPicker dict
        while True:
            res = self.request('https://www.hltv.org/matches/%s/asd'%matchId)
            if not res:
                print 'https://www.hltv.org/matches/%s/asd request error'%matchId
                time.sleep(5)
                continue
            break
        
        maps = re.findall(r'mapname">(.*?)<', res, re.DOTALL|re.IGNORECASE)
        mapBox = re.search(r'Maps(.*?)<div class="mapholder">', res, re.DOTALL|re.IGNORECASE)
        if mapBox:
            mapBox = mapBox.groups()[0]
        else:
            mapBox = ''
    #            
    #        print mapBox
        matchFormat = mapBox.strip().replace('\n', ' ')
    #        print matchFormatCat
    
        Apicked = re.findall(r'%s picked (\w+\d?)'%re.escape(teamALogo), matchFormat, re.DOTALL|re.IGNORECASE)
        Bpicked = re.findall(r'%s picked (\w+\d?)'%re.escape(teamBLogo), matchFormat, re.DOTALL|re.IGNORECASE)
        
        if all(['TBA'!=mp and 'Default'!=mp and '-'!=mp and mp for mp in maps]):
            self.db.update('upcomingHLTVMatches', {'matchId': matchId}, {'matchId': matchId, 'maps': str(maps)})
#            db['upcomingHLTVMatches'].update_one({'matchId': matchId}, {'$set': {'matchId': matchId, 'maps': str(maps)}})
            return [], {}
        return maps, {mp: 1 if mp in Apicked else -1 if mp in Bpicked else 0 for mp in maps}
        
    def parseSpecMatch(self, matchId, mode=0):
        while True:
            res = self.request('https://www.hltv.org/matches/%s/asd'%matchId)
            if not res:
                print '%s request error'%('https://www.hltv.org/matches/%s/asd'%matchId)
                time.sleep(3)
                continue
            break
        
        matchInfo = re.search(r'/team/(.*?)/.*?teamName">(.*?)</.*?'+\
                              'class="time".*?>(.*?)</.*?class="date".*?>(.*?)</.*?'+\
                              'events/(\d+).*?title="(.*?)".*?/team/(.*?)/.*?teamName">(.*?)</', res, re.DOTALL|re.IGNORECASE).groups()
        teamAId, teamALogo, matchTime, matchDate, eventId, eventTitle, teamBId, teamBLogo = matchInfo
        
        if 'logo'==teamAId or 'logo'==teamBId:
            print 'oops'
            return {}
            
        
        mapBox = re.search(r'Maps(.*?)<div class="mapholder">', res, re.DOTALL|re.IGNORECASE)
        if mapBox:
            mapBox = mapBox.groups()[0]
        else:
            mapBox = ''

        matchFormat = mapBox.strip().replace('\n', ' ')
        matchFormatCat = catDefine(matchFormat)
        
        Apicked = re.findall(r'%s picked (\w+\d?)'%re.escape(teamALogo), matchFormat, re.DOTALL|re.IGNORECASE)
        Bpicked = re.findall(r'%s picked (\w+\d?)'%re.escape(teamBLogo), matchFormat, re.DOTALL|re.IGNORECASE)
    
        maps = re.findall(r'mapname">(.*?)<', res, re.DOTALL|re.IGNORECASE)
        mapPicker = {mp: 1 if mp in Apicked else -1 if mp in Bpicked else 0 for mp in maps}
        updateDic = {'matchId': matchId, 'teamAId': teamAId, 'teamALogo': teamALogo, 'eventId': eventId, 'eventTitle': eventTitle, 'teamBId': teamBId, 'teamBLogo': teamBLogo, 'matchFormat': matchFormatCat, 'maps': str(maps), 'mapPicker': str(mapPicker)}
        if mode:
            bo = len(maps)
            if '2' in matchDate:
    #            print matchDate, re.search(r'(\d+).*?of\s+?(\w+)\s+(\d+)', matchDate, re.IGNORECASE|re.DOTALL).groups()
                matchDate = '-'.join(re.search(r'(\d+).*?of\s+?(\w+)\s+(\d+)', matchDate, re.IGNORECASE|re.DOTALL).groups())
    #            print matchDate
                if ':' in matchTime:
                    matchDate = datetime.datetime.strptime('%s %s'%(matchDate, matchTime), '%d-%B-%Y %H:%M')
                else:
                    matchDate = datetime.datetime.strptime(matchDate, '%d-%B-%Y')
                updateDic.update({'date': str(matchDate).split('.')[0], 'bo': bo})
            else:
                updateDic.update({'bo': bo})

        self.db.update('upcomingHLTVMatches', {'matchId': matchId}, updateDic)
        return self.db.find('upcomingHLTVMatches', {'matchId': matchId}, multi=False)
#        db['upcomingHLTVMatches'].update_one({'matchId': matchId}, {'$set': updateDic}, upsert=True)
#        return db['upcomingHLTVMatches'].find_one({'matchId': matchId})
    
    def getPreMatchInfo(self):
        # parse upcoming matches for closest 30 minutes matches
        for match in self.db.find('upcomingHLTVMatches', {'$and': [{'date': {'$gte': str(datetime.datetime.today()).split()[0]}}, {'date': {'$lt': str(datetime.datetime.today()+datetime.timedelta(seconds=60*30)).split()[0]}}] }, sort=[("date", 1)]):#, {'date': {'$lt': str(datetime.datetime.today()+datetime.timedelta(seconds=60*30)).split()[0]}}
#        for match in db['upcomingHLTVMatches'].find({'$and': [{'date': {'$gte': str(datetime.datetime.today()).split()[0]}}, {'date': {'$lt': str(datetime.datetime.today()+datetime.timedelta(seconds=60*30)).split()[0]}}] }).sort([('date', 1)]):#, {'date': {'$lt': str(datetime.datetime.today()+datetime.timedelta(seconds=60*30)).split()[0]}}
            res = self.request('https://www.hltv.org/matches/%s/asd'%match['matchId'])
            if not res:
                print '%s request error'%('https://www.hltv.org/matches/%s/asd'%match['matchId'])
                continue
            print 'https://www.hltv.org/matches/%s/asd'%match['matchId']
    #        teams = re.search(r'"/team/(\d+).*?alt="(.*?)".*?'+\
    #                              '/team/(\d+).*?alt="(.*?)".*?mapname">(.*?)<', res, re.DOTALL|re.IGNORECASE).groups()
            matchInfo = re.search(r'/team/(.*?)/.*?teamName">(.*?)</.*?'+\
                                  'events/(\d+).*?title="(.*?)".*?/team/(.*?)/.*?teamName">(.*?)</', res, re.DOTALL|re.IGNORECASE).groups()
            teamAId, teamALogo, eventId, eventTitle, teamBId, teamBLogo = matchInfo
            if 'logo'==teamAId or 'logo'==teamBId: # one of teams is TBD
                continue
    #        maps = list(matchInfo[6:])
    #        print matchInfo
            
            mapBox = re.search(r'Maps(.*?)<div class="mapholder">', res, re.DOTALL|re.IGNORECASE)
            if mapBox:
                mapBox = mapBox.groups()[0]
            else:
                mapBox = ''
    #            
    #        print mapBox
            matchFormat = mapBox.strip().replace('\n', ' ')
            matchFormatCat = catDefine(matchFormat)
    #        print matchFormatCat
    
            maps = re.findall(r'mapname">(.*?)<', res, re.DOTALL|re.IGNORECASE)
            self.db.update('upcomingHLTVMatches', {'matchId': match['matchId']}, {'teamAId': teamAId, 'teamALogo': teamALogo,
                                                                                   'eventId': eventId, 'eventTitle': eventTitle, 
                                                                                   'teamBId': teamBId, 'teamBLogo': teamBLogo,
                                                                                   'matchFormat': matchFormatCat, 'maps': str(maps)
                                                                                   })
#            db['upcomingHLTVMatches'].update_one({'matchId': match['matchId']}, {'$set': {'teamAId': teamAId, 'teamALogo': teamALogo, 'eventId': eventId, 'eventTitle': eventTitle, 'teamBId': teamBId, 'teamBLogo': teamBLogo, 'matchFormat': matchFormatCat, 'maps': str(maps)}}, upsert=True)
        
    
        
#obj = UpcomingMatchStats()
#obj.db.close()
#db.close()
