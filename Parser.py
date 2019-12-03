# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 08:33:19 2017

@author: Андрей
"""
import requests, time, re, copy
#import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
#import hltvUpcoming
import ast
#import mainHelper
#import test_main
#import ratings

from selenium import webdriver

from ast import literal_eval
import os
#import events
from sklearn.externals import joblib
from DataBase import DataBase 
import datetime

#from calendar import month_name
#months = {val.lower(): key+1 for key, val in enumerate(month_name[1:])}
#print months

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']

todayDate = datetime.datetime.today()
base = 'https://www.hltv.org'

#db = DataBase()

#def HLTVStatsHistoryThread():
##    getTeamsRating()
##    getMatchistory()
###    return
##    mindate = fullFillMatchistory()
###    return
###    mindate = '2018-04-19 19:50:00'
##    print 'mindate', mindate
##
##    evntLinks = eventScan(mindate)
##    fromEventsToFullMatchHistory(evntLinks)
##    makeMapStats(mindate)
##    makeRankHistory(mindate)
#    mindate = '2018-05-21 20:05:00'
#    lcldate = posRatingTeamRating('2018-05-01 20:05:00')
##    print '#lcldate', lcldate
#    predictTeamRating()
#    lcldate = posRatingTeamRating('2018-05-01 20:05:00')
#    fillMatchesRating(lcldate)
#    fillMatchesMapStats(mindate)
#
##    evntLinks = eventScan()
##    fromEventsToFullMatchHistory(evntLinks)
##    makeMapStats()
##    makeRankHistory()
##    predictTeamRating()
##    fillMatchesRating()
##    fillMatchesMapStats()
    

    
        
def getBetsCSGOKefs():
    driver = webdriver.PhantomJS('C://Python27/phantomjs/bin/phantomjs.exe')
    #нужно логиниться чтобы просмотреть историю коэфициентов
    driver.get('https://betscsgo.net/history/3/')
     
    html = driver.page_source
    #for i in re.findall(r' timerange(.*?)sys-active', html, re.IGNORECASE|re.DOTALL):
    #    i = i.split('script')[0]
    #    for j in re.findall(r'sys-t1name">(.*?)<.*?sys-stat-koef-1">x(\d+\.?\d*)<.*?sys-bo">(.*?)<.*?sys-stat-koef-2">x(\d+\.?\d*)<.*?sys-t2name">(.*?)<', i, re.IGNORECASE|re.DOTALL):
    #        print j
    driver.close()
    driver.quit()


#TODO PARSER
class Parser():
    def __init__(self, db=DataBase()):
        self.db = db# mongodb by default
        self.cj = {}
        self.proxy = {}
        self.session=''
        
    def request(self, url, dat='', head='', retUrl=False, proxy={}):
            if head =='':#use mobile headers
                head = {
                    'User-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 6P Build/XXXXX; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/47.0.2526.68 Mobile Safari/537.36',
                 } #
            try:
                #with LOCK:
                if dat=='':
                    a = requests.request('GET', url, headers=head, cookies=self.cj, allow_redirects=True, proxies = self.proxy, timeout= 10 )
                else:
                    a = requests.request('POST', url, data=dat, headers=head, cookies=self.cj, allow_redirects=True, proxies = self.proxy, timeout= 10)
                if retUrl:
                    return a.text, a.url
                return a.text
            except Exception,e :
                print 'R:%s'%e
                time.sleep(3)
            if retUrl:
                return '', ''
            return ''

    
                    
    def __getHLTVKefs(self, matchlink):
        #2321990/vexed-vs-radix-esl-uk-premiership-spring-2018-finals
    #    res = csgo.request('https://www.hltv.org/matches/2321990/vexed-vs-radix-esl-uk-premiership-spring-2018-finals')
    #    print base+matchlink
        res = self.request(base + matchlink)
        bets = re.findall(r'geoprovider_(.*?)</tr', res, re.IGNORECASE|re.DOTALL)
    #    print len(bets)
        bts = {}
        for bookmaker in bets:
            rs = re.findall(r'(.*?)".*?>(\d+\.?\d+?)<.*?>(\d+\.?\d+?)<', bookmaker, re.IGNORECASE|re.DOTALL)
            if rs:
                rs = rs[0]
                bts[rs[0]] = [float(rs[1]), float(rs[2])]
                
        return bts
        
    def HLTVkefsThread(self):
        # PARSE HLTV KEFS 30 MINUTES BEFORE THE MATCH
        while True:
            res = self.request('https://www.hltv.org/matches')
            daysMatches = re.findall(r'an class="standard-headline">(.*?)<(.*?)match-day"', res, re.IGNORECASE|re.DOTALL)
            matches = []
            for dt, cnt in daysMatches:
                
                dtmatches = re.findall(r'href="(.*?)".*?data-time-format.*?>(.*?)<.*?/(\d+)".*?/(\d+)".*?map-text">(.*?)<', cnt, re.IGNORECASE|re.DOTALL)
                for mtch in dtmatches:
                    print str(datetime.datetime.strptime('%s %s'%(dt, mtch[1]), '%Y-%m-%d %H:%M')), mtch[0], mtch[2], mtch[3], mtch[4]
                    matches = [datetime.datetime.strptime('%s %s'%(dt, mtch[1]), '%Y-%m-%d %H:%M'), mtch[0], mtch[2], mtch[3], mtch[4]]#+datetime.timedelta(hours=1)
                    if time.time()-time.mktime(matches[0].timetuple())>900:
                        dic = self.__getHLTVKefs(matches[1])
                        print 'update', matches
                        print 'kefs', dic
                        if dic:
                            self.db.update('matchKefs', {'matchLink': matches[1]}, {'teamAId': int(matches[2]), 'teamBId': int(matches[3]), 
                                                                       'date': str(matches[0]), 'matchLink': matches[1], 'HLTVkefs': str(dic)})
    #                        db['matchKefs'].update_one({'matchLink': matches[1]}, 
    #                                                              {'$set': 
    #                                                                  {'teamAId': int(matches[2]), 'teamBId': int(matches[3]), 
    #                                                                   'date': str(matches[0]), 'matchLink': matches[1], 'HLTVkefs': str(dic)}}, 
    #                                                    upsert=True)
                    else:
                        break
                    
                else:
                    print 'next day'
                    continue
                
                break
            
            if matches:
    #            print time.mktime((matches[0]+datetime.timedelta(hours=1)).timetuple())-time.time()
                print 'sleep %s min'%(min(1800, time.mktime((matches[0]+datetime.timedelta(hours=1)).timetuple())-time.time())/60)
    #            break
                time.sleep(min(time.mktime((matches[0]+datetime.timedelta(hours=1)).timetuple())-time.time(), 1800))
            else:
                print 'sleep %s min'%(1800/60)
                time.sleep(1800)
    
    
#parser = Parser(db)
#
#db.close()
