# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 08:33:19 2017

@author: Андрей
"""
import pymongo
import datetime
import copy
import datetime as dt
import pandas as pd
import numpy as np
import re
import unicodedata
import ast
import numpy as np
import Levenshtein as lv
import pycountry as pyc
from sklearn.externals import joblib


from Parser import Parser
from DataBase import DataBase
from Events import *


firPlacePrediction = joblib.load('firPlacePredictionModel')
cntryDecryptor = {'EU': u'Europe',
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

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']
#client.close()

#parser = Parser()
def eventParter(x):
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

#==============================================================================
# QUAL
#==============================================================================
    res = re.search(r'((?:Regional|Final|Main|Closed|Open|pre-)?\s*(?:Qualifier|Qualification)\s*\#?\d*)', x, re.IGNORECASE)
    if res:
        returnList[2] = res.groups()[0].strip()
        x = '%s %s'%(x[:x.find(returnList[2])].strip(), x[x.find(returnList[2])+len(returnList[2]):].strip())
        
    res = re.search(r'((?:finals?|relegation)\s*?)', x, re.IGNORECASE)
    if res:
        if returnList[2]:
            pass
#            print 'arrrrrrrrrrrr', x
        returnList[2] = res.groups()[0].strip()
        x = '%s %s'%(x[:x.find(returnList[2])].strip(), x[x.find(returnList[2])+len(returnList[2]):].strip())

#==============================================================================
#     SOME SHIT
#==============================================================================
    res = re.search(r'((?:last chance|BYOC|Wild Card)\s*?)', x, re.IGNORECASE)
    if res:
        returnList[3] = res.groups()[0].strip()
        x = '%s %s'%(x[:x.find(returnList[3])].strip(), x[x.find(returnList[3])+len(returnList[3]):].strip())
    
#==============================================================================
#     FEMALE        
#==============================================================================
    xlower = x.lower()
    if 'female' in xlower:
        returnList[4] = True
        x = '%s %s'%(x[:xlower.find('female')].strip(), x[xlower.find('female')+len('female'):].strip())
        
#==============================================================================
#     REST IS THE BASIS
#==============================================================================
    returnList[0] = x.strip()
    return returnList
    
    
    
def eventPreProccessor(x, lan=True):
    x = x.replace('StarLadder','SS').\
        replace('DreamHack', 'DH').\
        replace('Season', 'S').\
        replace('Regional', '').\
        replace('Offline', '').\
        replace('FFYI', ' ').\
        replace('&amp;', '').\
        replace(' and ', ' & ').\
        replace('&', '').\
        replace('/', ' ').\
        replace('#', '').\
        replace('Qual.', 'Qual').\
        replace(' Online', '').\
        replace(' Inferno', '').\
        replace('Asia-Pacific', 'APAC').\
        replace('OCE', 'Oceania')
        
    try:
        x = x.replace(u'ö', 'o')
    except:
        x = unicodedata.normalize('NFD', unicode(x, 'utf-8')).encode('ascii', 'ignore')
        
#        replace(' i-League', '').\
    for var in re.findall(r'(?:^|\s)(S \d+)', x):
        x = x.replace(var, var.replace(' ', ''))
    x = re.sub('Qual(?:$|[\.\s])', 'Qualifier', x, re.IGNORECASE)
    x = re.sub(' Championship(?!s)', '', x, re.IGNORECASE)
    x = re.sub(' i[-\s]League', '', x, flags=re.IGNORECASE)
    x = re.sub('promotion(?:\s|$)', 'Relegation', x, flags=re.IGNORECASE)
    x = re.sub('playoffs?(?:\s|$)', '', x, flags=re.IGNORECASE)
    x = x.replace('Qualifiers', 'Qualifier')
    x = re.sub(r'\scs:go' , '' ,x ,flags=re.IGNORECASE)
    x = re.sub(r'(?:^|\s)cup(?:$|\s)' , ' ' ,x, flags=re.IGNORECASE)
    x = re.sub(r'^SL', 'SS', x, flags=re.IGNORECASE)
    if lan:
        x = re.sub(r'(?:^|[^\w\d-])(lan)(?:\s|$)', ' ', x, flags=re.IGNORECASE)
    return x.strip()
        
    
def eventTitleProccessor(eventTitle):
#    eventTitle = event['eventTitle']
    eventTitle = eventTitle.replace(' at ', ' - ')
    if ['World', 'Championships'] == eventTitle.split(' ')[:2]:
        eventTitle = eventTitle.replace('World Championships', 'TWC')
        
    eventTitle = eventPreProccessor(eventTitle)
    eventSpliter = eventTitle.split(' - ')
    
    parts = [[] ,[]]
    if len(eventSpliter)==3:# in event[2]:
        eventSpliter[0] += ' %s'%eventSpliter.pop()#всегда qual есть в eventSpliter[2]
    if len(eventSpliter) == 2:
        eventSpliter[1]
        parts[1] = eventParter(eventSpliter[1])
        if not parts[1][0]:
            eventSpliter[0] += ' %s'%eventSpliter.pop()
            parts[1] = []
            
    parts[0] = eventParter(eventSpliter[0])
#    if 'Mal' in parts[0][0]:
#        print parts[0][0]
    return parts, eventTitle
    
class EventPreprocessor():
    def __init__(self, db=DataBase()):
        self.db = db
            
    
    def eventTitleFillDb(self):
        res = self.db.find('events', {'$and': [{'eventTitle': {'$exists': 1}}, {'eventTitle': {'$ne': ''}}, {'eventTitleBase0': {'$exists': False}}]})
#        res = db['events'].find({'$and': [{'eventTitle': {'$exists': 1}}, {'eventTitle': {'$ne': ''}}, {'eventTitleBase0': {'$exists': False}}]})
        print res.count()
        cnt = 0
        for i in res:
            if cnt and cnt%25==0:
                print cnt
                
            cnt += 1
            parts = eventTitleProccessor(i['eventTitle'])[0]
            for pt in range(len(parts)):
                if parts[pt]:
                    self.db.update('events', {'eventId': i['eventId']}, {'eventTitleBase%s'%pt: parts[pt][0].strip(),
                                                                                      'eventTitleRegion%s'%pt: parts[pt][1].strip(),
                                                                                      'eventTitleType%s'%pt: parts[pt][2].strip(),
                                                                                      'eventTitleRest%s'%pt: parts[pt][3].strip(),
                                                                                      'eventTitleFemale%s'%pt: parts[pt][4],
                                                                                      }, upsert=False)
#                    db['events'].update_one({'eventId': i['eventId']}, {'$set': {'eventTitleBase%s'%pt: parts[pt][0].strip(),
#                                                                                      'eventTitleRegion%s'%pt: parts[pt][1].strip(),
#                                                                                      'eventTitleType%s'%pt: parts[pt][2].strip(),
#                                                                                      'eventTitleRest%s'%pt: parts[pt][3].strip(),
#                                                                                      'eventTitleFemale%s'%pt: parts[pt][4],
#                                                                                      }})
    
        
    
    def eventLinkedIn(self):
        #TODO eventLinkedIn
    #    'dt1': {'$lte': str(datetime.datetime.today()).split()[0]}, 
    #    print str(datetime.datetime.today()-datetime.timedelta(days=150)).split()[0]
    #    return 
        res = self.db.find('events', {'dt1': {'$gte': str(datetime.datetime.today()-datetime.timedelta(days=150)).split()[0]}}, sort=[('dt2', -1)])#'altPrizeLink': {'$exists': False}, 'eventId': {'$gte': '3000'}
#        res = db['events'].find({'dt1': {'$gte': str(datetime.datetime.today()-datetime.timedelta(days=150)).split()[0]}}, sort=[('dt2', -1)])#'altPrizeLink': {'$exists': False}, 'eventId': {'$gte': '3000'}
        print res.count()
        for eventRaw in res:
            if not eventRaw.get('prizeDistr', ''):
                continue
    #        print eventRaw['eventId']
            eventRaw['prizeDistr'] = ast.literal_eval(eventRaw['prizeDistr'])
            if not eventRaw['prizeDistr'] and 'altPrizeLink' in eventRaw:
                continue
                
            if not eventRaw['prizeDistr']:
                if re.findall(r'\$((?:\d+,?)+)', eventRaw['prizePoolEvent']):
    #                print eventRaw['prizePoolEvent'], re.search(r'\$((?:\d+,?)+)', eventRaw['prizePoolEvent']).group(0)
                    eventRaw['prizeDistr'] = [[eventRaw['prizePoolEvent'].replace(re.search(r'\$((?:\d+,?)+)', eventRaw['prizePoolEvent']).group(0), '').strip()]]
                    if not eventRaw['prizeDistr'][0][0]:
                        continue
    #                print eventRaw['prizeDistr']
    #                return
                else:
    #                print eventRaw['prizePoolEvent']
    #                return
                    eventRaw['prizeDistr'] = [[eventRaw['prizePoolEvent']]]
    #        continue
            for event in eventRaw['prizeDistr']:
                if not event[-1] or event[-1].lower()=='tba':
                    continue
                spot = event[-1]#Spot at WCA 2016 World Finals
                    
    #            parts, event[2] = eventTitleProccessor(eventRaw['eventTitle'])
                parts = eventTitleProccessor(eventRaw['eventTitle'])[0]
                if parts[0][1]:
                    str1MainEv = ' '.join([tmp.replace(tmp, cntryDecryptor.get(tmp)) if tmp in cntryDecryptor else tmp for tmp in parts[0][1].split(' ')]).strip()
                    for tmp in ['North America']:#, 'South America']:
                        if tmp in str1MainEv and tmp in parts[0][1] :
                            str1MainEv = str1MainEv.replace(tmp, cntryDecryptor[tmp])
                
    #            if re.search(r'(?:^|[^\w\d-])(lan)(?:\s|$)', event[3], re.IGNORECASE):
    #                print event[3], '=>=>', event[2]
                prizeReserve = spot
                spot = eventPreProccessor(spot, False)
                prize = spot
                prize = prize.replace('\'', '').\
                                replace('?', '').\
                                replace('@', '').\
                                replace('Grand', '').\
                                replace('Invite', '').\
                                replace('ESEA Premier', 'ESEA MDL').\
                                replace('ESEA-P', 'ESEA MDL').\
                                replace('EPL', 'ESL Pro League').\
                                replace('DHV', 'DH Open Valencia').\
                                replace('EBL', 'Esports Balkan League').\
                                replace('Global Challenge', 'MDL Global Challenge').\
                                replace('EL Major', 'ELEAGUE Major').\
                                replace('Qualiifer', 'Qualifier').\
                                replace('to next event', '').strip()
                                
                if 'main tournament' in prize.lower() or 'main event' in prize.lower():
                    if parts[1]:
                        prize = re.sub('main (?:tournament|event)', parts[1][0], prize, flags=re.IGNORECASE)
                    else:
                        prize = re.sub('main (?:tournament|event)', parts[0][0], prize, flags=re.IGNORECASE)
                    
    #            print prize              
                                
    #            print prize      
                if 'legends' in prize.lower() or '$' in prize or 'isqualified' in prize or 'Withdrew' in prize or 'Community' in prize or 'Trip to' in prize or prize in ['Hardware', '204', '407', 'Relegated'] or 'all-star' in spot.lower() or 'contract' in prize or 'gear' in prize:
                    continue
                
    #            print prizeReserve, eventRaw['eventTitle']
                if re.findall(r's[lp]ot', prize, re.IGNORECASE):
                    prize = alternativePricePreprocess(prize)
                    prizeReserve = alternativePricePreprocess(prizeReserve)
                if event[-1].count(' at')>=2 or ' at' in event[-1] and not re.findall(r's[lp]ot', event[-1], re.IGNORECASE):
    #                Italy Finals at Milan Games
    #                Grand final at Gamescom
                    prize = re.sub(r' at .+', '', prize, flags = re.IGNORECASE)
                    prize = prize.strip()
                    
                prize = re.sub(r'(?:^|[^\w\d-])(lan)(?:\s|$)', ' ', prize, flags=re.IGNORECASE)
    
                    
                if re.findall(r'[^\w\d\s:-]', prize, re.IGNORECASE):
                    prize = re.split('(?:\s*\+\s*|,) ', prize)
                else:
                    prize = [prize]
                    
                altPrizeLink = []
                for pz in prize:
                    pz = pz.strip()
    #                if eventRaw['eventId']=='4065':
    #                    print eventRaw['eventId'], pz, prize, re.findall(r'[^\w\d\s:-]', prize[0], re.IGNORECASE)
                    if pz:
                        pass
                        altPrizeLink += [[self.altPrizeLinkedIn(spot, pz, parts, eventRaw), pz]]
                        
    #            if eventRaw['eventId']=='4065':
    #            print altPrizeLink
    #                return 
                if altPrizeLink:
                    self.db.update('events', {'eventId': eventRaw['eventId']}, {'altPrizeLink': altPrizeLink}, upsert=False)
#                    db['events'].update_one({'eventId': eventRaw['eventId']}, {'$set': {'altPrizeLink': altPrizeLink}})
                            
    def altPrizeLinkedIn(self, event, pz, parts, eventRaw):
        findDic = {}
    #    if parts[0][1]:
        
        str1MainEv = ' '.join([tmp.replace(tmp, cntryDecryptor.get(tmp)) if tmp in cntryDecryptor else tmp for tmp in parts[0][1].split(' ')]).strip()
        for tmp in ['North America']:#, 'South America']:
            if tmp in str1MainEv and tmp in parts[0][1] :
                str1MainEv = str1MainEv.replace(tmp, cntryDecryptor[tmp])
                
        pzParts = eventParter(pz)
        pzPartsLower = map(lambda x: x.lower(), pzParts[:4])
    #                        tmplst += [pzParts[1]]
    #                        tmplst1 += [pzParts[2]]
        pzPartsLower.append(pzParts[-1])
    #    print pzPartsLower
        
        if 'qual' in pzPartsLower[2]:
    #        print '123123123123'
            if 'close' in pzPartsLower[2]:
                pass#find event with close qual
                code2 = 0
    #                            findDic.update({'$and': [ {'$or': [{'eventTitleType0': {'$not': re.compile('open', re.IGNORECASE) }}, {'eventTitleType0': re.compile('clos', re.IGNORECASE) }]}, {'eventTitleType0': re.compile('qual', re.IGNORECASE) } ] })
            elif 'open' in pzPartsLower[2]:
                pass#find event with open qual
                code2 = 1
    #                            findDic.update({'$and': [ {'$or': [{'eventTitleType0': {'$not': re.compile('clos', re.IGNORECASE) }}, {'eventTitleType0': re.compile('open', re.IGNORECASE) }]}, {'eventTitleType0': re.compile('qual', re.IGNORECASE) } ] })
            else:
                pass#find eventl with Regional|Final|Main|pre- or with no add info
                code2 = 2
            findDic.update({'eventTitleType0': re.compile('qual', re.IGNORECASE) })
                
        elif 'final' in pzPartsLower[2]:
            pass#find eventl with final
            code2 = 3
            findDic.update({'$and': [ {'$and': [{'eventTitleType0': {'$not': re.compile('qual', re.IGNORECASE) }}, {'eventTitleType0': {'$not': re.compile('relegation', re.IGNORECASE) }}]} ]})
    
        elif 'relegation' in pzPartsLower[2]:
            pass#find eventl with relegation
            code2 = 4
            findDic.update({'$and': [ {'$and': [{'eventTitleType0': {'$not': re.compile('qual', re.IGNORECASE) }}, {'eventTitleType0': re.compile('relegation', re.IGNORECASE) } ]} ]})
        else:
            code2 = -1
    
        if re.search(r'(?:open\s+qual\w+\s*#?\d*|qual\w+\s+#?\d+)', parts[0][2], re.IGNORECASE):
            code2 = 1
        
    #                        if pzParts[3]:#the rest of title
    #                            findDic.update({'eventTitleRest0': pzParts[3]})
    #                        elif parts[0][3]:
    #                            findDic.update({'$or': [{'eventTitleRest0': parts[0][3]}, {'eventTitleRest0': ''}]})#eventTitleRest['eventTitleRest0'] = 
        if not pzParts[4]:
            findDic.update({'eventTitleFemale0': parts[0][-1]})
        
    #                    print db['events'].find(findDic).count()
    #                    break
        
        if 'dt2' in eventRaw:
            if eventRaw['dt2']:
                if '$and' not in findDic:
                    findDic.update({'$and': []})
                highLimit = str(datetime.datetime.strptime(eventRaw['dt2'], '%Y-%m-%d') + datetime.timedelta(days=360)).split(' ')[0]
                findDic['$and'] += [{'dt2': {'$gte': eventRaw['dt2']}}, {'dt1': {'$lte': highLimit}}]
    
        if not pzPartsLower[0]:#если нет основной части
    
            lclCntry = countryExtractor(eventRaw.get('location', ''))
            lclCntry1 = ' '.join([tmp.replace(tmp, cntryDecryptor.get(tmp)) if tmp in cntryDecryptor else tmp for tmp in lclCntry.split(' ')]).strip()
            for tmp in ['North America']:#, 'South America']:
                if tmp in lclCntry1 and tmp in lclCntry:
                    lclCntry1 = lclCntry1.replace(tmp, cntryDecryptor[tmp])
            
            str1 = ' '.join([tmp.replace(tmp, cntryDecryptor.get(tmp)) if tmp in cntryDecryptor else tmp for tmp in pzParts[1].split(' ')]).strip()
            for tmp in ['North America']:#, 'South America']:
                if tmp in str1 and tmp in pzParts[1]:
                    str1 = str1.replace(tmp, cntryDecryptor[tmp])
                    
            
            if parts[0][0]:
                prizeFirUpLet = getFirstUpperLetters(parts[0][0])
                prizeUpLetAndNum = getUpperLettersAndNumbers(parts[0][0])
            else:
                prizeFirUpLet = ''
                prizeUpLetAndNum = ''
                
                
            firstWordMainEv = parts[0][0].split(' ')[0]
            mainEvNumbers = re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', unicode(parts[0][0]))
                
            if prizeFirUpLet:
                if '$and' not in findDic:
                    findDic.update({'$and': []})
                findDic['$and'] += [{'$or': [   
                                                  {'eventTitleBase0': re.compile(r'(?:^|\s)%s(?:$|\s)'%firstWordMainEv)}, 
                                                  {'eventTitleBase0': ''},
                                                  {'eventTitleBase0': re.compile(r'(^\d*$)')},
                                                  {'firUpLetBase0': re.compile(prizeFirUpLet)},
                                                  {'upLetAndNumBase0': re.compile(prizeUpLetAndNum)} 
                                            ]}]
            else:
                findDic['$and'] += [{'$or': [   
                                                  {'eventTitleBase0': re.compile(r'(?:^|\s)%s(?:$|\s)'%firstWordMainEv)}, 
                                                  {'eventTitleBase0': re.compile(r'(^\d*$)')},
                                                  {'eventTitleBase0': ''}
                                    ]}]
            if pzParts[1]:
                regionDic = {'$or': [{'eventTitleRegion0': re.compile(pzParts[1])}, 
    #                                                      {'eventTitleRegion0': re.compile(lclCntry1)}, {'eventTitleRegion0': re.compile(lclCntry)}, 
                                      {'location': re.compile(pzParts[1])}, {'location': re.compile(str1)}, 
                                      {'eventTitleRegion0': re.compile(str1)}]}
                findDic['$and'] += [regionDic]
            
            res = self.db.find('events', findDic)
#            res = db['events'].find(findDic)
            rescnt = res.count()
            if rescnt==0:
    #            print '0-0 cnt ;', event[3], '=>', event[2],';', event[0], pzParts
                return ''
            
            else:# rescnt > 1:
                lst = []
    #            print event[2]
                for ind in range(2):
                    for raw in res:
    #                                            if event[0] == '3012':
    #                                                print raw['eventTitle'],  raw['dt1'], raw['dt2'], raw['eventTitleType0'], pzParts, '-%s-'%raw['eventTitleRest0'], event[0]==raw['eventId'], raw['eventTitleRest0'] and not parts[0][3] and not pzParts[3], ('qual' in pzParts[2].lower() or code2==1), re.search(r'(?:open\s+qual\w+\s*#?\d*|qual\w+\s+#?\d+)', raw['eventTitleType0'], re.IGNORECASE), 'pre' in raw['eventTitleType0'] and 'pre' in parts[0][2], not pzParts[2] and 'qual' in raw['eventTitleType0'].lower()
                        if (eventRaw['eventId']==raw['eventId'] or 
                            raw['eventTitleRest0'] and not parts[0][3] and not pzParts[3] or 
                            ('qual' in pzParts[2].lower() or code2==1) and re.search(r'(?:open\s+qual\w+\s*#?\d*|qual\w+\s+#?\d+)', raw['eventTitleType0'], re.IGNORECASE) or
                            'pre' in raw['eventTitleType0'] and 'pre' in parts[0][2] or
                            not pzParts[2] and 'qual' in raw['eventTitleType0'].lower()):
                            continue
                        
    #                                            if event[0] == '3264':
    #                                                print raw['eventTitle'],  raw['dt1'], raw['dt2'], raw['eventTitleType0'], raw['eventTitleRegion0'], parts[0][1], str1MainEv, max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(parts[0][1])), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1MainEv)))
                        if not pzParts[1] and raw['eventTitleRegion0'] and ind==0:# and not any([xx in ['EU', 'Europe', 'America', 'NA', 'APAC', 'Asia'] for xx in raw['eventTitleRegion0'].split(' ')]):
                            if max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(parts[0][1])), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1MainEv)))!=1:
                                if lclCntry and max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry)), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry1)))!=1:
                                    continue
                                
                        lst += [[raw['eventId'], raw['eventTitle'], raw['eventTitleBase0']]]
                        
                        if parts[0][0] and 'firUpLetBase0' in raw: 
    #                                            print [raw['upLetAndNumBase0'], prizeUpLetAndNum], lv.ratio(raw['upLetAndNumBase0'], prizeUpLetAndNum)
                            lst[-1] += [[0.5*lv.setratio(wraper(raw['eventTitleBase0']), wraper(parts[0][0])),
    #                                                                    lv.setratio(wraper(raw['eventTitle']), wraper(pzParts[0])),
                                        0.5*lv.ratio(raw['firUpLetBase0'], unicode(prizeFirUpLet)),
                                        0.5*lv.ratio(raw['upLetAndNumBase0'], unicode(prizeUpLetAndNum))]
                                        ]
                        elif not parts[0][0] and not raw.get('eventTitleBase0', ''):
                            lst[-1] += [[1.5]
                                        ]
                        else:
                            lst.pop()
                            continue
                        lst[-1][-1] = sum(lst[-1][-1])
                            
                        if pzParts[2]:
                            if raw['eventTitleType0']:
                                lst[-1] += [lv.setratio(wraper(raw['eventTitleType0']), wraper(pzParts[2]))]
                            elif 'inal' in pzParts[2]:
                                lst[-1] += [0.75]
                        
                            
                        if pzParts[1]:
                                         
                            lst[-1] += [1*max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(pzParts[1])), 
                                         lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1)),
                                         lv.setratio(wraper(raw['location']), wraper(pzParts[1])),
                                         lv.setratio(wraper(raw['location']), wraper(str1)))
                                    ]
                        else:
                            lst[-1] += [0.25*max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(parts[0][1])), 
                                         lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry)),
                                         lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry1)),
                                         lv.setratio(wraper(raw['location']), wraper(parts[0][1])),
                                         lv.setratio(wraper(raw['location']), wraper(lclCntry)),
                                         lv.setratio(wraper(raw['location']), wraper(lclCntry1)))
                                    ]
    
                        if pzParts[3]:
                            lst[-1] += [lv.setratio(wraper(raw['eventTitleRest0']), wraper(pzParts[3]))]
                            
                        
                        if prizeFirUpLet and 'firUpLetBase0' in raw: 
                            lst[-1] += [[lv.setratio(wraper(raw['eventTitleBase0']), wraper(parts[0][0])),
    #                                                           lv.setratio(wraper(raw['eventTitle']), wraper(pzParts[0])),
                                            lv.ratio(raw['firUpLetBase0'], unicode(prizeFirUpLet)),
                                            lv.ratio(raw['upLetAndNumBase0'], unicode(prizeUpLetAndNum))], raw['dt1'], raw['dt2']
                                        ]
                        else:
                            lst[-1] += [[], '', '']
                            
                    if lst:
                        break
    
                if not lst:
    #                print 'NO LST', event, '=>', eventRaw['eventTitle'],';', eventRaw['eventId'],  pzParts#, firstWord, altPrizeNumbers, parts[0]
                    return ''
    
                tmp = max(lst, key=lambda x: np.mean(x[3:-3]))
                tmpMean = np.mean(tmp[3:-3])
                maxEls = [x for i, x in enumerate(lst) if np.mean(x[3:-3]) == tmpMean]
                
                if 'relatedEvents' in eventRaw:# and pzParts[1]:
                    if eventRaw['relatedEvents']:
                        if type(eventRaw['relatedEvents'])!=list:
                            eventRaw['relatedEvents'] = ast.literal_eval(eventRaw['relatedEvents'])
                        if len(eventRaw['relatedEvents'])>=1:
                            if tmp[1] not in zip(*eventRaw['relatedEvents'])[2] and eventRaw['dt2']:
                                relEvRaw = [[x['eventId'], x['eventTitle'], x['eventTitleBase0'], x.get('dt1', ''), x.get('dt2', '')] for x in self.db.find('events', {'eventId': {'$in': zip(*eventRaw['relatedEvents'])[0]}})\
                                                 if (lambda xx: xx>=eventRaw['dt2'] if xx else False)(x.get('dt1', '')) ]
#                                relEvRaw = [[x['eventId'], x['eventTitle'], x['eventTitleBase0'], x.get('dt1', ''), x.get('dt2', '')] for x in db['events'].find({'eventId': {'$in': zip(*eventRaw['relatedEvents'])[0]}})\
#                                                 if (lambda xx: xx>=eventRaw['dt2'] if xx else False)(x.get('dt1', '')) ]
                                if relEvRaw:
                                    best = min(relEvRaw, key=lambda x: x[4])
                                    if best[4]<tmp[-2]:# and lv.setratio(wraper(raw['eventTitleBase0']), wraper(pzParts[0])):
                                        if parts[0][0]:
                                            tmp[:2] = max([(tmp[:2], lv.setratio(wraper(tmp[2]), wraper(parts[0][0])) ), (best[:2], lv.setratio(wraper(best[2]), wraper(parts[0][0])) )], key=lambda x: x[1])[0]
                                        elif not best[2]:
                                            tmp[:2] = best[:2]
                
                
                return tmp[0]
    
                    
        else:# если есть основная часть
    #                                continue
            if pzParts[0]:
                prizeFirUpLet = getFirstUpperLetters(pzParts[0])
                prizeUpLetAndNum = getUpperLettersAndNumbers(pzParts[0])
            else:
                prizeFirUpLet = ''
                prizeUpLetAndNum = ''
                    
            lclCntry = countryExtractor(eventRaw.get('location', ''))
            lclCntry1 = ' '.join([tmp.replace(tmp, cntryDecryptor.get(tmp)) if tmp in cntryDecryptor else tmp for tmp in lclCntry.split(' ')]).strip()
            for tmp in ['North America']:#, 'South America']:
                if tmp in lclCntry1 and tmp in lclCntry:
                    lclCntry1 = lclCntry1.replace(tmp, cntryDecryptor[tmp])
            
            str1 = ' '.join([tmp.replace(tmp, cntryDecryptor.get(tmp)) if tmp in cntryDecryptor else tmp for tmp in pzParts[1].split(' ')]).strip()
            for tmp in ['North America']:#, 'South America']:
                if tmp in str1 and tmp in pzParts[1]:
                    str1 = str1.replace(tmp, cntryDecryptor[tmp])
                
            
            firstWord = pzParts[0].split(' ')[0]
            altPrizeNumbers = re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', unicode(pzParts[0]))
                
            firstWordMainEv = parts[0][0].split(' ')[0]
            mainEvNumbers = re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', unicode(parts[0][0]))
                
            if prizeFirUpLet:
                if '$and' not in findDic:
                    findDic.update({'$and': []})
                findDic['$and'] += [{'$or': [   
                                                  {'eventTitleBase0': re.compile(r'(?:^|\s)%s(?:$|\s)'%firstWord)}, 
                                                  {'firUpLetBase0': re.compile(prizeFirUpLet)},
                                                  {'upLetAndNumBase0': re.compile(prizeUpLetAndNum)} 
                                            ]}]
            else:
                findDic.update({'eventTitleBase0': re.compile(r'(?:^|\s)%s(?:$|\s)'%firstWord)})
                
            
                    
            res = self.db.find('events', findDic)
#            res = db['events'].find(findDic)
            rescnt = res.count()
                
            if rescnt == 0:
                pass
                if pzParts[1]:
                    pass
    #                print '1-0 cnt ;',  event, '=>', eventRaw['eventTitle'],';', eventRaw['eventId'], pzParts, firstWord, altPrizeNumbers
                else:
                    pass
    #                print '1-1 cnt ;',  event, '=>', eventRaw['eventTitle'],';', eventRaw['eventId'], pzParts, firstWord, altPrizeNumbers
    #            print 
                return ''
                    
            else:#if rescnt>0:
                lst = []
                for ind in range(2):
                    for raw in res:
    #                                            if event[0] == '3264':
    #                                                print raw['eventTitle'],  raw['dt1'], raw['dt2'], raw['eventTitleType0'], pzParts, '-%s-'%raw['eventTitleRest0'], event[0]==raw['eventId'], raw['eventTitleRest0'] and not parts[0][3] and not pzParts[3], ('qual' in pzParts[2].lower() or code2==1), re.search(r'(?:open\s+qual\w+\s*#?\d*|qual\w+\s+#?\d+)', raw['eventTitleType0'], re.IGNORECASE), 'pre' in raw['eventTitleType0'] and 'pre' in parts[0][2], not pzParts[2] and 'qual' in raw['eventTitleType0'].lower()
                        if (eventRaw['eventId']==raw['eventId'] or 
                            raw['eventTitleRest0'] and not parts[0][3] and not pzParts[3] or 
                            ('qual' in pzParts[2].lower() or code2==1) and re.search(r'(?:open\s+qual\w+\s*#?\d*|qual\w+\s+#?\d+)', raw['eventTitleType0'], re.IGNORECASE) or
                            'pre' in raw['eventTitleType0'] and 'pre' in parts[0][2] or
                            not pzParts[2] and 'qual' in raw['eventTitleType0'].lower()):
                            continue
                        
    #                                            if event[0] == '3264':
    #                                                print raw['eventTitle'],  raw['dt1'], raw['dt2'], raw['eventTitleType0'], raw['eventTitleRegion0'], parts[0][1], str1MainEv, max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(parts[0][1])), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1MainEv)))
                        if not pzParts[1] and raw['eventTitleRegion0'] and ind==0:# and not any([xx in ['EU', 'Europe', 'America', 'NA', 'APAC', 'Asia'] for xx in raw['eventTitleRegion0'].split(' ')]):
                            if max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(parts[0][1])), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1MainEv)))!=1:
                                if lclCntry and max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry)), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry1)))!=1:
                                    continue
                                
                        lst += [[raw['eventId'], raw['eventTitle'], raw['eventTitleBase0']]]
                        
                        if prizeFirUpLet and 'firUpLetBase0' in raw: 
    #                                            print [raw['upLetAndNumBase0'], prizeUpLetAndNum], lv.ratio(raw['upLetAndNumBase0'], prizeUpLetAndNum)
                            lst[-1] += [[2*lv.setratio(wraper(raw['eventTitleBase0']), wraper(pzParts[0])),
    #                                                                    lv.setratio(wraper(raw['eventTitle']), wraper(pzParts[0])),
                                        lv.ratio(raw['firUpLetBase0'], unicode(prizeFirUpLet)),
                                        lv.ratio(raw['upLetAndNumBase0'], unicode(prizeUpLetAndNum))]
                                        ]
                            if pzParts[1]:
                                if 'firUpLetBase0' in eventRaw:
                                    lst[-1][-1] += [0.75*lv.setratio(wraper(raw['eventTitleBase0']), wraper(eventRaw['eventTitleBase0'])),
                                                    0.5*lv.ratio(raw['firUpLetBase0'], unicode(eventRaw['firUpLetBase0'])),
                                                    0.5*lv.ratio(raw['upLetAndNumBase0'], unicode(eventRaw['upLetAndNumBase0']))]
                                else:
                                    lst[-1][-1] += [0.75*lv.setratio(wraper(raw['eventTitleBase0']), wraper(eventRaw['eventTitleBase0']))]
                        else:
                            lst[-1] += [[2*lv.setratio(wraper(raw['eventTitleBase0']), wraper(pzParts[0]))]]
                            if pzParts[1]:
                                if 'firUpLetBase0' in eventRaw:
                                    lst[-1][-1] += [0.75*lv.setratio(wraper(raw['eventTitleBase0']), wraper(eventRaw['eventTitleBase0'])),
                                                    0.5*lv.ratio(raw['firUpLetBase0'], unicode(eventRaw['firUpLetBase0'])),
                                                    0.5*lv.ratio(raw['upLetAndNumBase0'], unicode(eventRaw['upLetAndNumBase0']))]
                                else:
                                    lst[-1][-1] += [0.75*lv.setratio(wraper(raw['eventTitleBase0']), wraper(eventRaw['eventTitleBase0']))]
                        lst[-1][-1] = sum(lst[-1][-1])
                            
                        if pzParts[2]:
                            if raw['eventTitleType0']:
                                lst[-1] += [lv.setratio(wraper(raw['eventTitleType0']), wraper(pzParts[2]))]
                            elif 'inal' in pzParts[2]:
                                lst[-1] += [0.75]
                        
                            
                        if pzParts[1]:
    #                                            if flag:#pzParts[1] in cntryDecryptor:
    #                                            print 'ere', wraper(raw['eventTitleRegion0']), wraper(pzParts[1]), wraper(raw['eventTitleRegion0']), wraper(cntryDecryptor[pzParts[1]])
                            lst[-1] += [0.5*max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(pzParts[1])), 
                                         lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1)),
                                         lv.setratio(wraper(raw['location']), wraper(pzParts[1])),
                                         lv.setratio(wraper(raw['location']), wraper(str1)))
                                    ]
    #                                            else:
    #    #                                            print wraper(raw['eventTitleRegion0']), wraper(pzParts[1]), wraper(raw['eventTitleType0']), wraper(pzParts[2])
    #                                                lst[-1] += [0.5*lv.setratio(wraper(raw['eventTitleRegion0']), wraper(pzParts[1]))]
                        else:
                            if parts[0][1]:#str1MainEv
                                if max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(parts[0][1])), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(str1MainEv)))==1:
                                    
                                    lst[-1] += [0.5]
                                elif not raw['eventTitleRegion0']:
                                    lst[-1] += [0.5]
                                else:
                                    lst[-1] += [0]
                            elif lclCntry:
                                
                                if lclCntry:
                                    if max(lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry)), lv.setratio(wraper(raw['eventTitleRegion0']), wraper(lclCntry1)))==1:
                                        lst[-1] += [0.5]
                                    else:
                                        lst[-1] += [0]
                                else:
                                    lst[-1] += [0]
    #                                            else:
    #                                                lst[-1] += [0.25]
    
                        if pzParts[3]:
                            lst[-1] += [lv.setratio(wraper(raw['eventTitleRest0']), wraper(pzParts[3]))]
                            
                        if parts[1]:
                            if 'eventTitleBase1' in raw:
                                lst[-1] += [5*lv.setratio(wraper(raw['eventTitleBase1']), wraper(parts[1][0]))]
    #                                            else:
                            lst[-1][1] = max( lst[-1][1], lv.setratio( wraper(raw['eventTitle']), wraper(parts[1][0]) ) )
                            
                        if not altPrizeNumbers:
                            if mainEvNumbers:
                                if firstWord==firstWordMainEv:
                                    lst[-1] += [2.5*lv.setratio(re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', raw['eventTitleBase0']), mainEvNumbers)]
                        else:
    #                                            print re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', raw['eventTitleBase0']), altPrizeNumbers
                            lst[-1] += [2.5*lv.setratio(re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', raw['eventTitleBase0']), altPrizeNumbers)]
                           
                        
                        if prizeFirUpLet and 'firUpLetBase0' in raw: 
    #                                            print [raw['upLetAndNumBase0'], prizeUpLetAndNum], lv.ratio(raw['upLetAndNumBase0'], prizeUpLetAndNum)
                            lst[-1] += [[lv.setratio(wraper(raw['eventTitleBase0']), wraper(pzParts[0])),
    #                                                           lv.setratio(wraper(raw['eventTitle']), wraper(pzParts[0])),
                                            lv.ratio(raw['firUpLetBase0'], unicode(prizeFirUpLet)),
                                            lv.ratio(raw['upLetAndNumBase0'], unicode(prizeUpLetAndNum))], raw['dt1'], raw['dt2']
                                        ]
                        else:
                            lst[-1] += [[], '', '']
                            
                    if lst:
                        break
    
                if not lst:
    #                print 'NO LST', event, '=>', eventRaw['eventTitle'],';', eventRaw['eventId'], pzParts, firstWord, altPrizeNumbers, parts[0]
                    return ''
    
                tmp = max(lst, key=lambda x: np.mean(x[3:-3]))
                tmpMean = np.mean(tmp[3:-3])
                maxEls = [x for i, x in enumerate(lst) if np.mean(x[3:-3]) == tmpMean]
                
                    
                if len(maxEls)>1:
                    newMaxEls = sorted([[x, re.findall(r'(?:^|\s)((?:[^\s]*?\d+[^\s]*?)+)(?:\s|$)', x[1])]  for i, x in enumerate(maxEls)], key=lambda x: x[1])
                    if newMaxEls[0][1]==newMaxEls[1][1] or not newMaxEls[0][1]:
                        if tmp[-2]:
                            newMaxEls = sorted([[x, np.mean(x[-3])]  for i, x in enumerate(maxEls)], key=lambda x: x[1], reverse=True)
                            
                        if newMaxEls[0][1]==newMaxEls[1][1] or not newMaxEls[0][1]:
                            tmp = min(maxEls, key=lambda x: x[-1])
                            tmpMean = np.mean(tmp[3:-3])
                        else:
                            tmp = newMaxEls[0][0]
                            tmpMean = np.mean(tmp[3:-3])
                    else:
                        tmp = newMaxEls[0][0]
                        tmpMean = np.mean(tmp[3:-3])
                
                if 'relatedEvents' in eventRaw:# and pzParts[1]:
                    if eventRaw['relatedEvents']:
                        if type(eventRaw['relatedEvents'])!=list:
                            eventRaw['relatedEvents'] = ast.literal_eval(eventRaw['relatedEvents'])
                        if len(eventRaw['relatedEvents'])>=1:
                            if tmp[1] not in zip(*eventRaw['relatedEvents'])[2] and eventRaw['dt2']:
                                pass
                                relEvRaw = [[x['eventId'], x['eventTitle'], x['eventTitleBase0'], x.get('dt1', ''), x.get('dt2', '')] for x in self.db.find('events', {'eventId': {'$in': zip(*eventRaw['relatedEvents'])[0]}})\
#                                relEvRaw = [[x['eventId'], x['eventTitle'], x['eventTitleBase0'], x.get('dt1', ''), x.get('dt2', '')] for x in db['events'].find({'eventId': {'$in': zip(*eventRaw['relatedEvents'])[0]}})\
                                                 if sum([lv.setratio(wraper(x['eventTitleBase0']), wraper(pzParts[0])),
                                                        lv.ratio(x['firUpLetBase0'], unicode(prizeFirUpLet)),
                                                        lv.ratio(x['upLetAndNumBase0'], unicode(prizeUpLetAndNum))])>1 and 
                                                    (lambda xx: xx>=eventRaw['dt2'] if xx else False)(x.get('dt1', '')) ]
                                if relEvRaw:
                                    best = min(relEvRaw, key=lambda x: x[4])
                                    if best[4]<tmp[-2]:# and lv.setratio(wraper(raw['eventTitleBase0']), wraper(pzParts[0])):
                                        tmp[:2] = max([(tmp[:2], lv.setratio(wraper(tmp[1]), wraper(pzParts[0])) ), (best[:2], lv.setratio(wraper(best[2]), wraper(pzParts[0])) )], key=lambda x: x[1])[0]
                                        
                return tmp[0]
    
    
    
    def firPlaceCalculation(self):
        for i in self.db.find('events', {"eventId" : {"$exists": True}, 'firPlace': {'$exists': False}}):#{'$and': [{'prizeDistr': {'$exists': 1}}, {'prizeDistr': {'$ne': ''}}]}):
#        for i in db['events'].find({"eventId" : {"$exists": True}, 'firPlace': {'$exists': False}}):#{'$and': [{'prizeDistr': {'$exists': 1}}, {'prizeDistr': {'$ne': ''}}]}):
            if ast.literal_eval(i.get('prizeDistr', '[]')):
                i['prizeDistr'] = ast.literal_eval(i['prizeDistr'])
                cashDistr = map(lambda x: int(x[2][1:].replace(',', '')) if '$' in x[2] else int(x[2].replace(',', '') if x[2] else 0), i['prizeDistr'])
                if not cashDistr:
                    self.db.update('events', {'eventId': i['eventId']}, {'firPlace': 0}, upsert=False)
#                    db['events'].update_one({'eventId': i['eventId']}, {'$set': {'firPlace': 0}})
            #        print cashDistr, i['eventId']
                else:
                    self.db.update('events', {'eventId': i['eventId']}, {'firPlace': max(cashDistr)}, upsert=False)
#                    db['events'].update_one({'eventId': i['eventId']}, {'$set': {'firPlace': max(cashDistr)}})
            #        print i['eventId'], max(cashDistr)
            else:
        #        ['teamQnt', 'prizeUSD']
                firPlace = 0
                if i.get('teamAtnd', 0) and i.get('prizeUSD', 0):
                    proc = firPlacePrediction.predict([i.get('teamAtnd'), i.get('prizeUSD')])[0]
                    firPlace = int(round(proc*i.get('prizeUSD')))
                self.db.update('events', {'eventId': i['eventId']}, {'firPlace': firPlace}, upsert=False)
#                db['events'].update_one({'eventId': i['eventId']}, {'$set': {'firPlace': firPlace}})
    
    #firPlaceCalculation()
    #        i['prizeDistr'] = map(lambda x: int(x[2][1:].replace(',', '')) if '$' in x[2] else int(x[2].replace(',', '') if x[2] else 0), i['prizeDistr'])
    #print set([3,2,1])==set([1,2,3])
    
    def maxPrizeCalculation(self):
        for i in self.db.find('events', {'prizeUSD': {'$exists': True}, 'firPlace': {'$exists': True}, 'dt2': {'$gte': str(datetime.datetime.today()).split()[0]}}):#'altPrizeLink': {'$exists': True}}):
#        for i in db['events'].find({'prizeUSD': {'$exists': True}, 'firPlace': {'$exists': True}, 'dt2': {'$gte': str(datetime.datetime.today()).split()[0]}}):#'altPrizeLink': {'$exists': True}}):
    #        if 'firPlace' not in i:
    #            continue
            maxFirPlace = i['firPlace']
            maxPrizePool = i['prizeUSD']
            if i.get('altPrizeLink', ''):
            #    evIdLst = []
                evIdLst = list(zip(*i['altPrizeLink'])[0])
            #    print evIdLst, i['eventId']
                while evIdLst:
                    if not evIdLst[0]:
                        evIdLst.pop(0)
                        continue
                    raw = self.db.find('events', {'eventId': evIdLst[0]}, multi=False)
#                    raw = db['events'].find_one({'eventId': evIdLst[0]})
            #        if not raw:
            #            print evIdLst[0]
                    maxFirPlace = max(maxFirPlace, raw['firPlace'])
                    maxPrizePool = max(maxPrizePool, raw['prizeUSD'])
                    if raw.get('altPrizeLink', ''):
                        evIdLst += list(set(list(zip(*raw['altPrizeLink'])[0])) - set(evIdLst))
                    evIdLst.pop(0)
                self.db.update('events', {'eventId': i['eventId']}, {'firPlace': maxFirPlace, 'maxPrizeUSD': maxPrizePool}, upsert=False)
#                db['events'].update_one({'eventId': i['eventId']}, {'$set': {'firPlace': maxFirPlace, 'maxPrizeUSD': maxPrizePool}})
            else:
                self.db.update('events', {'eventId': i['eventId']}, {'firPlace': maxFirPlace, 'maxPrizeUSD': maxPrizePool}, upsert=False)
#                db['events'].update_one({'eventId': i['eventId']}, {'$set': {'maxPrizeUSD': maxPrizePool}})


#parser.close()
#db = DataBase()
#db.close()