# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 08:33:19 2017

@author: Андрей
"""
import requests
import time
import re
import copy
#import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import pycountry as pyc
import unicodedata
import datetime
import ast
import threading

from calendar import month_name
from sklearn.externals import joblib
from selenium import webdriver

from Parser import Parser
from DataBase import DataBase

firPlacePrediction = joblib.load('firPlacePredictionModel')
cntryDecryptor = {'EU': u'Europe',
                  'LatAm': u'Latin America',
                  'LA': u'Latin America',
                  'SA': u'South America',
                  'MN': u'Mongolia',
                  'NA': u'North America',
                  'ANZ': u'NZ Australia', 
                  'Europe': u'EU',
                  'North America': u'NA',
#                      'South America': u'SA'
                  }
wraper = lambda x: map(unicode, filter(bool, x.split(' ')))
rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
country = [[], []]
for cntr in pyc.countries:
    country[0] += [cntr.name]
    country[1] += [cntr.alpha2]
    
country[1] += ['UK']
country[0] += ['Russia', 'Asia-Pacific', 'Taiwan', 'South Korea', 'Benelux', 'Korea', 'Vietnam', 'Macau', 'Iran', 'Britain', 'Portuguese', 'Oceanic', 'Swedish', 'Danish', 
       'Pacific Islands', 'Pacific', 'Chinese', 'Adriatica', 'Iberia', 'Nordic', 'Baltics', 'Scandinavia', 'Middle East', 'Iberian', 'Balkan']
    
todayDate = datetime.datetime.today()
base = 'https://www.hltv.org'

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']
returnList = []
    
#months = {val.lower(): key+1 for key, val in enumerate(month_name[1:])}
#print months

def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in xrange(1, 1 + len(s1)):
       for y in xrange(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
   return s1[x_longest - longest: x_longest]

def plotGraph(args):
    plt.plot(*args)
    plt.show()

def listParter(lst, n_parts):
    lstLength = len(lst)
    partLength = lstLength/float(min(lstLength, n_parts))
    last = 0
    parts = []
    
    while last<lstLength:
        parts.append(lst[int(last):int(last+partLength)])
        last += partLength
        
    return parts



def roman_to_int(s):
    int_val = 0
    try:
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
    except:
        return s
    return int_val

def lowerCaseCheck(x):
    for i in x:
        if i.islower():
            return True
    return False

def getFirstUpperLetters(x):
    returnList = ''
    for i in re.split(r'[-:\.\s]', x):
        if not i:
            continue
        if re.search('\d', i):
            rs = re.search('^([A-Z]*\d+)$', i)
            if rs:
#                print i
                returnList += rs.groups()[0]
                continue
        if i[0].isupper():
#            print i
            if lowerCaseCheck(i):
                returnList += i[0]
            else:
                returnList += i
    return returnList

def getUpperLettersAndNumbers(x):
    returnList = ''
    for i in x:
        if re.search('[A-Z0-9]', i):
            returnList += i
    return returnList
                    

def getUpperCase(x):
    returnList = ''
    for i in x:
        if i.isupper() or re.match(r'\d', i):
            returnList += i
#        if re.match(r'\d', i[0]):
#            returnList += re.search(r'(\d+)', i).groups()[0]
            
    return returnList
         
def alternativePricePreprocess(x):
    if re.findall(r'(\d+ \w+ [sS][pl]ots?)', x, re.DOTALL|re.IGNORECASE|re.UNICODE):
        return x.replace(re.findall(r'(\d+ \w+ [sS][pl]ots?)', x, re.DOTALL|re.IGNORECASE|re.UNICODE)[0], re.findall(r'\d+ (\w+) [sS][pl]ots?', x, re.DOTALL|re.IGNORECASE|re.UNICODE)[0])
#    print '=>',
    return re.sub('\d*\s*[sS][pl]ots*\s?((?:at|in)\s*(the\s+)?)?', '', x, re.IGNORECASE|re.UNICODE).strip()

def countryExtractor(x):
    returnList = ['', '', '', '', False]#mainTitle, Region, Qual, some shit, isFemale
#    x = x.decode('utf-8').encode('utf-8')
#==============================================================================
#     REGION
#==============================================================================
#    print x
    res = re.search(r'((?:(?:(?:Middle|Central|East|West|North|South|Post|Latin|Northern|Southeast|Eastern|Southern)?(?:\s+|-|/))*(?:(?:Europea?n?|America[sn]?|CIS|Asian?|Oceanian?|African?)+(?!\w)\s*\&?\s*)+(?!\w)\s*\&?\s*)+)', x, re.IGNORECASE)
    if res:
        returnList[1] = res.groups()[0]
#        print res.groups()
        if returnList[1][-3:] == ' & ':
            returnList[1] = returnList[1][:-3]
        if returnList[1][-1] == '&':
            returnList[1] = returnList[1][:-1]
            
        x = '%s %s'%(x[:x.find(returnList[1])].strip(), x[x.find(returnList[1])+len(returnList[1]):].strip())
        
    rs = re.findall(r'((?<![a-zA-Z0-9\.:\'-])[A-Z]+)(?:$|[^a-zA-Z0-9\.\!:-])', x)
    for lclx in rs:
        if type(roman_to_int(lclx))==int:
#            print x, re.search(r'((?<![a-zA-Z0-9\.:\'-])%s)(?:$|[^a-zA-Z0-9\.\!:-])'%str(lclx), x).groups()
            x = re.sub(r'((?<![a-zA-Z0-9\.:\'-])%s)(?:$|[^a-zA-Z0-9\.\!:-])'%str(lclx), '%s '%str(roman_to_int(lclx)), x)
#            print x
        elif len(lclx)==2:
            if lclx in 'PG, TV, SP, HP, OZ, GO, EB, GG, SL, DH, SS, BD, GC, DL, EL':
                continue
            if 'The UK' in x:
                x = re.sub('The UK', '', x)
                
            x = re.sub(r'((?<![a-zA-Z0-9\.:\'-])%s)(?:$|[^a-zA-Z0-9\.\!:-])'%lclx, '', x)
            returnList[1] += ' %s'%lclx
        elif len(lclx)>2:
            if lclx in ['SEA', 'GCC', 'ANZ', 'CIS', 'EMEA', 'APAC', 'MENA']:
                x = re.sub(r'((?<![a-zA-Z0-9\.:\'-])%s)(?:$|[^a-zA-Z0-9\.\!:-])'%lclx, '', x)
                returnList[1] += ' %s'%lclx
              
#    x = unicodedata.normalize('NFD', unicode(x, 'utf-8')).encode('ascii', 'ignore')
    for c in country[0]:
#        while True:
#            try:
                
                if c in x:#re.search(r'(%s(?!\w))'%c, x, re.IGNORECASE):#.encode('utf-8'):
                    clen = len(c)
                    if '%s '%c in x or x[-clen:]==c or '%s,'%c in x:
                        x = re.sub(c, '', x)
                        returnList[1] += ' %s'%c
#                break
#            except:
#                x = unicodedata.normalize('NFD', unicode(x, 'utf-8')).encode('ascii', 'ignore')
            
    cccc = re.search(r' (S \d+)', x)
    if cccc:
        x = x.replace(cccc.groups()[0], cccc.groups()[0].replace(' ', ''))
        
    for br in ['Europe', 'America', 'CIS', 'Asia', 'Oceania', 'Africa']:
        if br.lower() in x.lower():
#            print xlower
            res = re.search(r'((?:North|East|South|Southeast)*?\s*?(?:Asia|America|Oceania)+)', x, re.IGNORECASE).groups()[0]
            x = '%s %s'%(x[:x.find(res)].strip(), x[x.find(res)+len(res):].strip())
            returnList[1] += ' %s'%res.strip()
            
    x = re.sub('\s+[(?:&|vs|\+)\\/,]*?\s+',  ' ', x).strip()
    returnList[1] = returnList[1].strip()
    return returnList[1]

    
#eventTitleFillDb()


#eventTitleFillDb()
#upLet()

class Event(Parser):
    def __init__(self, db=DataBase()):
        Parser.__init__(self, db)
        self.db = db# mongodb by default
        
    def getEventPrizeDistribution(self, eventId):
        #TODO getEventPrizeDistribution
    #    print eventId
        res = self.request(base + '/events/%s/asd'%eventId, proxy={})
    #    if url:
    #        eventDesc = url.split('/')[-1]
    #    else:
    #        eventDesc = ''
        relatedEvents = re.findall(r'related-event".*?href="/events/(.*?)".*?title="(.*?)"', res, re.IGNORECASE|re.DOTALL)
        eventName = re.search(r'"eventname">(.*?)<', res, re.IGNORECASE|re.DOTALL).groups()[0]
    #    dt1, dt2, prizePool, teamAtnd, location
        teamsAtnd = re.findall(r'class="col">.*?/team/(\d+)/', res, re.IGNORECASE|re.DOTALL)
    #    print re.findall(r'eventdate"><(.*?)</td.*?\"prizepool text-ellipsis">(.*?)<.*?teamsNumber">.*?(\d*).*?<.*?title="(.*?)">.*?>(.*?)<', res, re.IGNORECASE|re.DOTALL)
        dt, prizePool, teamAtnd, flagLoc, location = re.search(r'eventdate"><(.*?)</td.*?\"prizepool text-ellipsis">(.*?)<.*?teamsNumber">.*?(\d*).*?<.*?title="(.*?)">.*?>(.*?)<', res, re.IGNORECASE|re.DOTALL).groups()
    #    print countryExtractor(location)
    #    dt, prizePool, teamAtnd, flagLoc, location = re.search(r'eventdate"><(.*?)</td.*?<td>(.*?)<.*?<td>.*?(\d*).*?<.*?title="(.*?)">(.+?)<', res, re.IGNORECASE|re.DOTALL).groups()
        location = max(countryExtractor(flagLoc), countryExtractor(location))
    #    print dt
        if dt.count('data-unix')==2:
            dt1m, dt1d, dt1y, dt2m, dt2d, dt2y = re.search(r'">(\w+)\s*(\d+)\w*\s*(\d*)</.*?">(\w+)\s*(\d+)\w*\s*(\d*)</', dt, re.IGNORECASE).groups()
            dt2 = datetime.datetime.strptime('%s %s %s'%(dt2m, dt2d, dt2y), r'%b %d %Y')
            if dt1y:
                dt1 = str(datetime.datetime.strptime('%s %s %s'%(dt1m, dt1d, dt1y), r'%b %d %Y')).split(' ')[0]
            else:
                dt1 = str(datetime.datetime.strptime('%s %s %s'%(dt1m, dt1d, dt2y), r'%b %d %Y')).split(' ')[0]
            dt2 = str(dt2).split(' ')[0]
        elif dt.count('data-unix')==1:
            dt1m, dt1d, dt1y = re.search(r'">(\w+)\s*(\d+)\w*\s*(\d*)</', dt, re.IGNORECASE).groups()
            dt1 = str(datetime.datetime.strptime('%s %s %s'%(dt1m, dt1d, dt1y), r'%b %d %Y')).split(' ')[0]
            dt2 = dt1
        else:
            dt1, dt2 = '', ''
            
    #    print 'dt', dt1, dt2
        if '$' in prizePool:
            prizeUSD = int(re.search(r'\$((?:\d+,?)+)', prizePool).groups()[0].replace(',', ''))
        else:
            prizeUSD = 0
    #        print prizeUSD
        
        if teamsAtnd:
            teamAtnd = len(teamsAtnd)
        elif teamAtnd:
            teamAtnd = int(teamAtnd)
        else:
            teamAtnd = 0
    #    print teamAtnd
        
        try:
            prizeDistr = re.search(r'class="placements">(.+?)class="section-header"', res, re.IGNORECASE|re.DOTALL).groups()[0]
            prizeDistr = re.findall(r'/team/(\d+)/.*?<div>(.*?)<.*?prizeMoney">(.*?)<.*?prizeMoney">(.*?)<', prizeDistr, re.IGNORECASE|re.DOTALL)
            return prizeDistr, relatedEvents, eventName, dt1, dt2, prizePool, prizeUSD, teamAtnd, location
        except:
    #        print 'EXCEPT', eventId
            return [], relatedEvents, eventName, dt1, dt2, prizePool, prizeUSD, teamAtnd, location
    
        
    #def eventScanUPON():
    #    res = csgo.request(base+'/events#tab-ALL')
    ##    print map(lambda x: map(lambda y: y.strip(), x), re.findall(r'(/events/\d+/.*?)"\s*class="a-reset ongoing-event.*?"text-ellipsis">.*?</div>.*?<.*?>(.*?)<', res, re.DOTALL|re.IGNORECASE))
    #    with open('eventsUPON.txt', 'w') as f:
    ##        print re.findall(r'(/events/\d+/.*?)".*?col-value small-col">(.*?)<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', res, re.IGNORECASE|re.DOTALL)
    #        map(lambda x: f.write('%s\n'%(';'.join(x))), re.findall(r'(/events/\d+/.*?)".*?col-value small-col">(.*?)<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', res, re.IGNORECASE|re.DOTALL))
    #        map(lambda x: f.write('%s\n'%(';'.join(map(lambda y: y.strip(), x)))), re.findall(r'(/events/\d+/.*?)"\s*class="a-reset ongoing-event.*?"text-ellipsis">.*?</div>.*?<.*?>(.*?)<', res, re.DOTALL|re.IGNORECASE))
            
    
    #def eventScanV1():
    #    ids = 0
    #    while True:#ids!=1700:
    #        res = csgo.request(base+'/events/archive?offset=%s'%ids)
    ##        print re.findall(r'(/events/\d+/.*?)".*?text-ellipsis">(.*?)<.*?col-value small-col">(.*?)<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', res, re.IGNORECASE|re.DOTALL)
    ##        return
    #        for i in re.findall(r'standard-headline">(.*?)<(.*?)class="spacer">', res, re.DOTALL|re.IGNORECASE):
    ##            i[0] = i[0].strip()
    #            if 'August' in i[0].strip() and '2015' in i[0].strip():
    #                return
    #            print i[0].strip()
    #            with open('events.txt', 'a') as f:
    #                map(lambda x: f.write('%s\n'%(';'.join(x)+';%s'%i[0].strip())), re.findall(r'(/events/\d+/.*?)".*?col-value small-col">(.*?)<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', i[1], re.IGNORECASE|re.DOTALL))
    #        
    #        ids += 50
    #        print ids
    ##        if re.search(r'august\s*2015', res, re.DOTALL|re.IGNORECASE):
    ##            break
    ##        return
    
    def eventScan(self, date=''):#seems to be unused
        #TODO GET EVENT_LINK, CURENT_PRIZPOOL, ALTERNATIVE_PRIZE
        if not date:
            res = self.db.find('fullMatchHistory')
    #        res = db['fullMatchHistory'].find()
        else:
            res = self.db.find('fullMatchHistory', {'date': {'$gte': date}})
    #        res = db['fullMatchHistory'].find({'date': {'$gte': date}})
        events = set([i['eventLink'] for i in res])
        events = list(events - set(self.db.find('events', distinct_field='eventLink')))
    #    events = list(events - set(db['events'].distinct('eventLink')))
            
        lenEvents = len(events)
        print '#eventScan scan %s events'%lenEvents
    #    db['events'].delete_many({})
        evntLinks = []
    #    flag = False
    #    print db['fullMatchHistory'].find_one({'eventLink': '2874/streamme-gauntlet-cis-vs-eu-6" title="Stream.me Gauntlet: CIS vs EU '})
        for ind, event in enumerate(events):
    #        if not ind%20:
    #            print ind, lenEvents
                
            if self.db.find('events', {'eventLink': event}, multi=False):
    #        if db['events'].find_one({'eventLink': event}):            
                continue
            
            req = self.request(base + '/events/'+ event)
    #        req = csgo.request('https://www.hltv.org/events/3591/iem-sydney-2018-asian-open-qualifier-1')
    #        https://www.hltv.org/events/3591/iem-sydney-2018-asian-open-qualifier-1
            prizePool = re.findall(r'class="eventdate".*?<td>(.*?)<', req, re.DOTALL|re.IGNORECASE)[0]
            
            prizePoolUsd = 0.0
            if '$' in prizePool:
                prizePoolUsd = float(re.findall(r'\$([\d,]+)', prizePool)[0].replace(',',''))
                
            if 'spot' in prizePool or 'slot' in prizePool:
                try:
                    slots = int(re.findall(r'(\d+)[^\d]+s[pl]ot', prizePool)[0])
                except:
                    slots = 0
            else:
                slots = 0
    #        flag = True
            evntLinks += [event]
            self.db.update('events', {'eventLink': event}, {'eventLink': event, 'prizePool': prizePoolUsd, 'alternativePrize': slots})
    #        db['events'].update_one({'eventLink': event}, {'$set': {'eventLink': event, 'prizePool': prizePoolUsd, 'alternativePrize': slots}}, upsert = True)
            
        return evntLinks
           
    def scanEventMain(self, eventId):                 
        #TODO asd
        try:
            prizeDistr, relatedEvents, eventName, dt1, dt2, prizePoolEvent, prizeUSD, teamAtnd, location = self.getEventPrizeDistribution(eventId)
        except:
            print 'continue', eventId
            return False
        
#        print prizeDistr, relatedEvents, eventName, dt1, dt2, prizePoolEvent, prizeUSD, teamAtnd, location
        if relatedEvents:
            relatedEvents = map(lambda x: x[0].split('/')+[x[1]], relatedEvents)
        else:
            relatedEvents = ''
            
#        print eventId
        dic = {'prizeDistr': str(prizeDistr), 
              'eventId': eventId, 'prizeUSD': prizeUSD, 'prizePoolEvent': prizePoolEvent,
              'dt1': dt1, 'dt2': dt2, 'teamAtnd': teamAtnd, 'location': location,
              'eventTitle': eventName, 'relatedEvents': str(relatedEvents),
              }
        
              
        try:
            self.db.update('events', {'eventId': eventId}, dic)
#            db['events'].update_one({'eventId': eventId}, {'$set': dic}, upsert=True)
        except Exception as e:
            print e, 'oooops cant update', eventId
            
    #        print 'pass'
        for tmp in relatedEvents:
            if self.db.find('events', {'eventId': tmp[0]}, multi=False):
#            if db['events'].find_one({'eventId': tmp[0]}):
                continue
    #            print tmp
            try:
                prizeDistr, localRelatedEvents, eventName, dt1, dt2, prizePoolEvent, prizeUSD, teamAtnd, location = self.getEventPrizeDistribution(tmp[0])
            except:
                print 'continue1', tmp
                continue
            
            if localRelatedEvents:
                localRelatedEvents = map(lambda x: x[0].split('/')+[x[1]], localRelatedEvents)
            else:
                localRelatedEvents = ''
                
            dic = {'prizeDistr': str(prizeDistr), 
                  'eventId': tmp[0], 'prizeUSD': prizeUSD, 'prizePoolEvent': prizePoolEvent,
                  'dt1': dt1, 'dt2': dt2, 'teamAtnd': teamAtnd, 'location': location,
    #              'eventDesc': eventDesc,
                  'eventTitle': eventName, 'relatedEvents': str(localRelatedEvents)}
            if not prizeDistr:
                prizeDistr = ''
                
            try:
                self.db.update('events', {'eventId': tmp[0]}, dic)
#                db['events'].update_one({'eventId': tmp[0]}, {'$set': dic}, upsert=True)
            except Exception as e:
                print e, 'oooops cant update', tmp[0]
                
        return True

    def upLet(self):
#        print '#UPLET'
        res = self.db.find('events', {'eventTitleBase0': {'$exists': True}, 'firUpLetBase0': {'$exists': False}})
#        res = self.db['events'].find({'eventTitleBase0': {'$exists': True}, 'firUpLetBase0': {'$exists': False}})
        print res.count()
        cnt = 0
        for i in res:
            if cnt and cnt%25==0:
                print cnt
                
            cnt += 1
            dc = {}
        #    if re.search(r'^World Championships? \d+', i['eventTitle']):#if ['World', 'Championships'] == i['eventTitleBase0'].split(' ')[:2]:
        #        i['eventTitle'] = re.sub(r'^World Championships?', 'TWC', i['eventTitle'])
        #        firUpLetBase0 = getFirstUpperLetters(i['eventTitle'])
        #        upLetAndNumBase0 = getUpperLettersAndNumbers(i['eventTitle'])
        #        dc.update({'firUpLetBase0': firUpLetBase0, 'upLetAndNumBase0': upLetAndNumBase0})
            if i['eventTitleBase0']:
                firUpLetBase0 = getFirstUpperLetters(i['eventTitleBase0'])
                upLetAndNumBase0 = getUpperLettersAndNumbers(i['eventTitleBase0'])
                dc.update({'firUpLetBase0': firUpLetBase0, 'upLetAndNumBase0': upLetAndNumBase0})
            if i.get('eventTitleBase1', ''):
                firUpLetBase1 = getFirstUpperLetters(i['eventTitleBase1'])
                upLetAndNumBase1 = getUpperLettersAndNumbers(i['eventTitleBase1'])
                dc.update({'firUpLetBase1': firUpLetBase1, 'upLetAndNumBase1': upLetAndNumBase1})
                
            if dc:
                self.db.update('events', {'eventId': i['eventId']}, dc, upsert=False)
    #            db['events'].update_one({'eventId': i['eventId']}, {'$set': dc})
        
    def getUpOnEvents(self):
        res = self.request('https://www.hltv.org/events')
        print '#ongoing events'
        for event in re.findall(r'/events/(\d+/.*?)\s*class="a-reset\s*ongoing-event"(.*?)</tr', res, re.DOTALL|re.IGNORECASE):
            tryToFind = self.db.find('events', {'eventId': event[0].split('/')[0]}, multi=False)
#            tryToFind = db['events'].find_one({'eventId': event[0].split('/')[0]})
            if tryToFind:
                continue
            print event[0]
            if 'lan-marker' in event[1]:
                eventType = 'LAN'
            else:
                eventType = 'Online'
            self.db.update('events', {'eventId': event[0].split('/')[0]}, {'eventId': event[0].split('/')[0], 'eventDesc': event[0].split('/')[1], 'eventType': eventType})
#            db['events'].update_one({'eventId': event[0].split('/')[0]}, {'$set': {'eventId': event[0].split('/')[0], 'eventDec': event[0].split('/')[1], 'eventType': eventType}}, upsert=True)
        print '#upgoing events'
        for i in re.findall(r'standard-headline">(.*?)<(.*?)class="spacer">', res, re.DOTALL|re.IGNORECASE):
            print i[0].strip()
            res = re.findall(r'/events/(\d+/.*?)".*?col-value.*?</.*?col-value(.*?)</.*?<.*?>(.*?)</.*?">(.*?)</', i[1], re.IGNORECASE|re.DOTALL)
            print len(res)
            for event in res:
                eventId, eventDesc = event[0].split('/')
    #            if db['events'].find_one({'eventId': eventId}):
    #                continue
    #            print event
                if 'title' in event[1]:
                    try:
                        teamAtnd = int(event[2])
                    except:
                        teamAtnd = -1
    #                prizePoolEvent = re.search(r'>(.+)', event[1], re.DOTALL|re.IGNORECASE).group(1)
                    eventType = 'Intl. LAN'
                else:
    #                prizePoolEvent = event[2]
                    try:
                        teamAtnd = int(re.search(r'>(.+)', event[1], re.DOTALL|re.IGNORECASE).group(1))
                    except:
                        teamAtnd = -1
                    eventType = event[3]
                    
    #            print eventId, eventDec, teamAtnd, prizePoolEvent, eventType
    #            db['events'].update_one({'eventId': eventId}, {'$set': {'eventId': eventId, 'eventDec': eventDec, 'teamAtnd': teamAtnd, 'prizePoolEvent': prizePoolEvent, 'eventType': eventType}}, upsert=True)
                self.db.update('events', {'eventId': eventId}, {'eventId': eventId, 'eventDesc': eventDesc, 'teamAtnd': teamAtnd, 'eventType': eventType})
#                db['events'].update_one({'eventId': eventId}, {'$set': {'eventId': eventId, 'eventDec': eventDec, 'teamAtnd': teamAtnd, 'eventType': eventType}}, upsert=True)
            
    #getUpOnEvents()
    
    def getPastEvents(self, stopDate=datetime.datetime.strptime('August 2015', '%B %Y')):
        print '#past events'
        ids = 0
        while True:
            res = self.request('https://www.hltv.org/events/archive?offset=%s'%ids)
            for i in re.findall(r'standard-headline">(.*?)<(.*?)class="spacer">', res, re.DOTALL|re.IGNORECASE):
    #            if 'August' in i[0].strip() and '2015' in i[0].strip():
    #                return
                print i[0].strip()
    #            print re.findall(r'/events/(\d+)/.*?col-value small-col">(\d+).*?<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', i[1], re.IGNORECASE|re.DOTALL)
                for event in re.findall(r'/events/(\d+/.*?)".*?col-value small-col">(\d+).*?<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', i[1], re.IGNORECASE|re.DOTALL):
    #                print event
    #                db['events'].update_one({'eventId': event[0].split('/')[0]}, {'$set': {'eventId': event[0].split('/')[0], 'eventDec': event[0].split('/')[1], 'teamAtnd': int(event[1]), 'prizePoolEvent': event[2], 'eventType': event[3]}}, upsert=True)
                    self.db.update('events', {'eventId': event[0].split('/')[0]}, {'eventId': event[0].split('/')[0], 'eventDesc': event[0].split('/')[1], 'teamAtnd': int(event[1]), 'eventType': event[3]})
#                    db['events'].update_one({'eventId': event[0].split('/')[0]}, {'$set': {'eventId': event[0].split('/')[0], 'eventDec': event[0].split('/')[1], 'teamAtnd': int(event[1]), 'eventType': event[3]}}, upsert=True)
                
                if datetime.datetime.strptime(i[0].strip(), '%B %Y')<stopDate:
                    return 
    #            with open('events.txt', 'a') as f:
    #                map(lambda x: f.write('%s\n'%(';'.join(x)+';%s'%i[0].strip())), re.findall(r'(/events/\d+/.*?)".*?col-value small-col">(.*?)<.*?col-value small-col.*?>(.*?)<.*?col-value small-col.*?>(.*?)<', i[1], re.IGNORECASE|re.DOTALL))
            ids += 50
            print ids
    
    #getPastEvents()
    
    def getUncheckedPrizeDistrEvents(self):
        print '#getUncheckedPrizeDistrEvents'
        res = self.db.find('events', {'$or': [{'dt2': {'$exists': False}}, {'$and': [{'dt1': {'$gte': str(datetime.datetime.today()).split()[0]}}]}], 'eventId': {'$gte': '3000'}})#, {'prizeDistr': {'$exists': False}}, {'eventId': {'$gte': '2700'}}
#        res = db['events'].find({'$or': [{'dt2': {'$exists': False}}, {'$and': [{'dt1': {'$gte': str(datetime.datetime.today()).split()[0]}}]}], 'eventId': {'$gte': '3000'}})#, {'prizeDistr': {'$exists': False}}, {'eventId': {'$gte': '2700'}}
        #, {'dt1': {'$lte': str(datetime.datetime.today()+datetime.timedelta(days=30)).split()[0]}}
        print res.count()
        cnt = 0
        for dic in res:
    #        print dic
            if cnt%25==0:
                print cnt
            try:
#                print dic['eventId']
                self.scanEventMain(dic['eventId'])
            except:
                print dic
                pass
            cnt += 1
    
    def main(self, preprocesor):
        print '#EVENT START'
        lastEvent = list(self.db.find('events', {'dt2': {'$lte': str(datetime.datetime.today()).split()[0]}}, sort=[("dt2", -1)]))[0]
#        lastEvent = db['events'].find_one({'dt2': {'$lte': str(datetime.datetime.today()).split()[0]}}, sort=[("dt2", pymongo.DESCENDING)])
        lastDate = datetime.datetime.strptime(lastEvent['dt2'], '%Y-%m-%d')
#        lastDate = datetime.datetime.strptime('2018-03-01', '%Y-%m-%d')
        print '#LAST EVENT DATE', lastDate, lastEvent['eventId']
        self.getUpOnEvents()
        self.getPastEvents(lastDate)
        self.getUncheckedPrizeDistrEvents()#заполнение даты туринра, призовой фонд, количество команд, связанные турниры
#        assert False
        preprocesor.eventTitleFillDb()# добавление в БД инфы по Base, Region, Type, Rest, Female
        self.upLet()# преобразование eventTitle и извлечение firUpLetBase0 и upLetAndNumBase0
        print '#altPrizeLink'
        preprocesor.eventLinkedIn()#altPrizeLink
        preprocesor.firPlaceCalculation()#заполнение первого места для каждого ивента
        preprocesor.maxPrizeCalculation()#заполнение первого места для каждого ивента
        print '#EVENT END'    

def scaner(iters, func, proxy={}):
    #TODO scaner
    for number, val in enumerate(iters):
        if not number%25:
            print '%s/%s'%(number, len(iters))
            
#        print proxy
        func(val, proxy=proxy)            
        
    

def multithreadScan(getterFunc, args0, scanFunc, args1, n_thread=1):
    with open('proxy.txt', 'r') as f:
        allProxies = map(lambda x: x.strip(), f.readlines())
        n_threads = len(allProxies) + 1
        
#    print min(n_threads, n_thread)
    parts = listParter(getterFunc(*args0), min(n_threads, n_thread))
    print 'parts = ', len(parts)
    
    thrds = []
    
#    scaner(parts[0][-100:], scanFunc, {})


    thrds.append(threading.Thread(target=scaner, args=[parts[0], scanFunc, {}]))
    thrds[-1].start()
#    thrds[-1].join()
    
    for ind, i in enumerate(parts[1:]):
        #scaner(iters, func, proxy={}):
        proxy = {"http": allProxies[ind], 'https': allProxies[ind]}
        thrds.append(threading.Thread(target=scaner, args=[i, scanFunc, proxy]))
        thrds[-1].start()
        time.sleep(2)

    for i in range(min(n_threads, n_thread)):
        thrds[i].join()
        

def eventScanFile(filename):
    #TODO eventScanFile
    obj = Event()
    with open(filename, 'r') as f:
        res = map(lambda x: x.strip().split(';'), f.readlines())
        for ind, i in enumerate(res):
#            if ind<875:
#                continue
            if not ind%25:
                print ind, '!!!!!!!!!!!!!'
            i[0] = i[0].split('/')[2:]
    #        print i[0]
            obj.scanEventMain(i)
            
            


#2322080 - bo5
#Tyllo : MVP PK : 2318394
#ago : windigo : 2318190: bo5
#2318171/alientech-vs-hexagone-moche-tpgo with default



#{ eventId : "4080" }

#eventLinkedIn()

#for eventRaw in db['events'].find({'eventId': {'$gte': '3000'}}):
#    if eventRaw.get('prizePoolEvent', '') and eventRaw.get('prizePoolEvent', '').lower()!='tba' and '$' not in eventRaw.get('prizePoolEvent', ''):
#        print eventRaw.get('prizePoolEvent', '')


    
#maxPrizeCalculation()        

 

#for i in db['events'].find({'dt1': {'$exists': True}}):
#    if i['dt1']>i['dt2']:
#        print i['eventId'], i['eventTitle'], i['dt1'], i['dt2']
#        dt = datetime.datetime.strptime(i['dt2'], '%Y-%m-%d')
#        if dt.month+1>12:
#            dt = datetime.date(dt.year+1, 1, dt.day)
#        else:
#            dt = datetime.date(dt.year, dt.month+1, dt.day)
#        db['events'].update_one({'eventId': i['eventId']}, {'$set': {'dt2': str(dt)}})

        
        

# Если в качестве alternativePrize выступает Legendary - значит пацаны выйграли мажор либо попали в плейофф и отберутся на след мажор, и никакого спота на турнир нет        

    
    
#tm = time.time()
#main()
#print time.time() - tm
        
#getUncheckedPrizeDistrEvents()
#keys = [']  
#db
    
#print getEventPrizeDistribution(4009)#prizeDistr, relatedEvents, eventName, dt1, dt2, prizePool, prizeUSD, teamAtnd, location
#client.close()