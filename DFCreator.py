# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 08:33:19 2017

@author: Андрей
"""
import ast
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from DataBase import DataBase
#months = {val.lower(): key+1 for key, val in enumerate(month_name[1:])}
#print months

todayDate = datetime.datetime.today()
base = 'https://www.hltv.org'

class DFCreator():
    def __init__(self, db=DataBase()):
        self.db = db
    
    def createHLTVRatingDF(self):
        #TODO createHLTVRatingDF
        clmns = ['teamId', 'date', 'rating', 'position']
        
        df = pd.DataFrame(columns=clmns)
        
        with open('csgoHLTVRatings.csv', 'w') as f:
            df.to_csv(f, sep=';', index=False)
        
        lst = []
        for ind, doc in enumerate(list(self.db.find('teamsRating'))):
            if ind%100==0 and ind/100>1:
                print ind
                df = pd.DataFrame(lst, columns=clmns)
                with open('csgoHLTVRatings.csv', 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=False)
                    
                lst = []
                
#            print doc
#            if not 'posRating' in doc:
#                print doc
            lst += [[doc['teamId'], doc['date'], doc['teamRating'], doc.get('position', -1)]]
                
            
        df = pd.DataFrame(lst, columns=clmns)
        with open('csgoHLTVRatings.csv', 'a') as f:
            df.to_csv(f, sep=';', index=False, header=False)
    
    def createPrizeDistrDF(self):
        #TODO createPrizeDistrDF
        clmns = ['eventId', 'date', 'teamId', 'win', 'prizePool', 'firPlace', 'maxPrizeUSD']
        
        df = pd.DataFrame(columns=clmns)
        
        with open('csgoPrizeDistr.csv', 'w') as f:
            df.to_csv(f, sep=';', index=False)
        
        visited = []
        lst = []
        for ind, doc in enumerate(list(self.db.find('events'))):
            if ind%100==0 and ind/100>1:
                print ind
                df = pd.DataFrame(lst, columns=clmns)
                with open('csgoPrizeDistr.csv', 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=False)
                    
                lst = []
                
            if doc.get('prizeDistr', '') not in ['', '[]'] and doc.get('eventId', ''):
                if doc['eventId'] in visited:
                    continue
                else:
                    visited += [doc['eventId']]
                    
                doc['prizeDistr'] = ast.literal_eval(doc['prizeDistr'])
                doc['prizeDistr'] = map(list, doc['prizeDistr'])
                doc['prizeDistr'] = map(lambda x: x + [int(x[2][1:].replace(',', ''))] if '$' in x[2] else x + [int(x[2].replace(',', ''))] if x[2] else x + [0], doc['prizeDistr'])
                
    #            fr = 0
                for j, pz in enumerate(doc['prizeDistr']):
                    if pz[-1]:
    #                    if fr and pz[-1]>fr:
    #                        print doc['eventId']
    #                        if j==1:
    #                            lst[-1][3] = pz[-1]
    #                            lst += [[doc['eventId'], doc['dt2'], pz[0], doc['prizeDistr'][j-1][-1], doc.get('prizeUSD', -1), doc.get('firPlace', -1)]]
    #                    fr = pz[-1]
                        lst += [[doc['eventId'], doc['dt2'], pz[0], pz[-1], doc.get('prizeUSD', -1), doc.get('firPlace', -1), doc.get('maxPrizeUSD', -1)]]
                
            
        df = pd.DataFrame(lst, columns=clmns)
        with open('csgoPrizeDistr.csv', 'a') as f:
            df.to_csv(f, sep=';', index=False, header=False)
            
    def createMainDF(self):
        #TODO createMainDF
        clmns = ['matchlinkId', 'teamAId', 'teamBId', 'map', 'mapPicker', 'date', 
                 'teamAScore', 'teamBScore', 'teamACTScore', 'teamATScore', 'teamBCTScore', 'teamBTScore',
                 'teamAKill', 'teamAhs', 'teamADeath', 'teamAAss', 'teamAFlash', 'teamAAdrAvg', 'teamAKastAvg', 'teamAFirstK', 'teamAClutch', 'teamADifuse', 'teamAExplode',
                 'teamBKill', 'teamBhs', 'teamBDeath', 'teamBAss', 'teamBFlash', 'teamBAdrAvg', 'teamBKastAvg', 'teamBFirstK', 'teamBClutch', 'teamBDifuse', 'teamBExplode',
                 'teamAHalf_0', 'teamAHalf_1', 
                 'teamBHalf_0', 'teamBHalf_1', 
                 'teamAPistol_half_0', 'teamAPistol_half_1',
                 'teamBPistol_half_0', 'teamBPistol_half_1',
                 'teamAMapLeftToWin', 'teamBMapLeftToWin',
                 'bo', 'matchFormat', 'ratingType', 
                 'teamAMatchRating', 'teamBMatchRating',# 'teamARating', 'teamBRating', 'teamAposRat', 'teamBposRat',# 'teamAMapStat', 'teamBMapStat',
                 'matchType', 'eventId']
        
        for _ in range(8):
            clmns += ['teamAHalf_0_winStrike_%s'%_, 
                      'teamAHalf_0_looseStrike_%s'%_, 
                      'teamAHalf_1_winStrike_%s'%_, 
                      'teamAHalf_1_looseStrike_%s'%_,
                      'teamBHalf_0_winStrike_%s'%_, 
                      'teamBHalf_0_looseStrike_%s'%_, 
                      'teamBHalf_1_winStrike_%s'%_, 
                      'teamBHalf_1_looseStrike_%s'%_]
            
        for _ in ['A', 'B']:
            for __ in ['half_0_', 'half_1_', '']:
                for i in range(5):
                    clmns += ['team%s_%scat%s_played'%(_, __, i), 'team%s_%scat%s_win'%(_, __, i), 'team%s_%scat5VScat%s_played'%(_, __, i), 'team%s_%scat5VScat%s_win'%(_, __, i)]
                        
#        for _ in ['A', 'B']:
#            for __ in ['half_0_', 'half_1_', '']:
#                for ___ in range(5):
#                    clmns += ['team%s_%scat%s_played'%(_, __, ___), 'team%s_%scat%s_win'%(_, __, ___)]
                    
        print clmns, len(clmns)
#        assert False
                 
        
        df = pd.DataFrame(columns=clmns)
        
        with open('csgoMain.csv', 'w') as f:
            df.to_csv(f, sep=';', index=False)
            
        visited = []
        lst = []
        cnt = 0
        for ind, doc in enumerate(list(self.db.find('fullMatchHistory'))):
            if ind%100==0 and ind/100>1:
                print ind
                df = pd.DataFrame(lst, columns=clmns)
                with open('csgoMain.csv', 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=False)
                    
                lst = []
                
                
            if not doc.get('map', '') or not doc.get('date', ''):# or not doc.get('teamARating', ''):#   or not doc.get('mapPicker', '')
                continue
            
    #        if doc.get('matchlinkId')=='2321813':
    #            print doc
    ##        continue
            if (doc['matchlinkId'], doc['map']) in visited:
                continue
            else:
                visited += [(doc['matchlinkId'], doc['map'])]
                
            cnt += 1
            lst += [[doc.get('matchlinkId'), doc.get('teamAId'), doc.get('teamBId'), doc.get('map'), doc.get('mapPicker'),
                      doc.get('date'), doc.get('teamAResScore', -10), doc.get('teamBResScore', -10), doc.get('teamACTScore', -10), doc.get('teamATScore', -10),
                       doc.get('teamBCTScore', -10), doc.get('teamBTScore', -10), doc.get('teamAKill', -10), doc.get('teamAhs', -10), doc.get('teamADeath', -10),
                        doc.get('teamAa', -10), doc.get('teamAFlash', -10), doc.get('teamAadravg', -100), doc.get('teamAKastavg', -1), doc.get('teamAFirstK', -500),
                         doc.get('teamAClutch', -1), doc.get('teamADifuse', -1), doc.get('teamAExplode', -1),
                          doc.get('teamBKill', -10), doc.get('teamBhs', -10), doc.get('teamBDeath', -10), doc.get('teamBa', -10), doc.get('teamBFlash', -10),
                           doc.get('teamBadravg', -100), doc.get('teamBKastavg', -1), doc.get('teamBFirstK', -500), doc.get('teamBClutch', -1), doc.get('teamBDifuse', -1),
                            doc.get('teamBExplode', -1), 
                            doc.get('teamAHalf_0', -1), doc.get('teamAHalf_1', -1), 
                             doc.get('teamBHalf_0', -1), doc.get('teamBHalf_1', -1), 
                             doc.get('teamAPistol_half_0', 0), doc.get('teamAPistol_half_1', 0),
                             doc.get('teamBPistol_half_0', 0), doc.get('teamBPistol_half_1', 0),
                             doc.get('teamAMapLeftToWin', -1), doc.get('teamBMapLeftToWin', -1), doc.get('bo', -1), doc.get('matchFormat', -10), doc.get('ratingType', -1), 
                             doc.get('teamAResRating', -10), doc.get('teamBResRating', -10),# doc.get('teamARating', ''), doc.get('teamBRating', ''),
                              #doc.get('teamAposRat', ''), doc.get('teamBposRat', ''), 
                              doc.get('matchType'), doc.get('eventLink')]]
            
            for _ in range(8):
                lst[-1] += [doc.get('teamAHalf_0_winStrike_%s'%_, 0),
                          doc.get('teamAHalf_0_looseStrike_%s'%_, 0),
                          doc.get('teamAHalf_1_winStrike_%s'%_, 0),
                          doc.get('teamAHalf_1_looseStrike_%s'%_,0),
                          doc.get('teamBHalf_0_winStrike_%s'%_, 0),
                          doc.get('teamBHalf_0_looseStrike_%s'%_, 0),
                          doc.get('teamBHalf_1_winStrike_%s'%_, 0),
                          doc.get('teamBHalf_1_looseStrike_%s'%_, 0)]
                
            for _ in ['A', 'B']:
                for __ in ['half_0_', 'half_1_', '']:
                    for i in range(5):
                        lst[-1] += [doc.get('team%s_%scat%s_played'%(_, __, i), 0), doc.get('team%s_%scat%s_win'%(_, __, i), 0),
                                    doc.get('team%s_%scat5VScat%s_played'%(_, __, i), 0), doc.get('team%s_%scat5VScat%s_win'%(_, __, i), 0)]
            
        print cnt
        df = pd.DataFrame(lst, columns=clmns)
        with open('csgoMain.csv', 'a') as f:
            df.to_csv(f, sep=';', index=False, header=False)
    
    
    def createEventDF(self):
        #TODO createEventDF
        clmns = ['eventId', 'firPlace', 'eventType', 'eventTitleFemale0', 'teamQnt', 'prizeUSD', 'maxPrizeUSD', 'endDate']#еще дифьюзы и тд.
        
        df = pd.DataFrame(columns=clmns)
        
        with open('csgoEvent.csv', 'w') as f:
            df.to_csv(f, sep=';', index=False)
            
        visited = []
        lst = []
        for ind, doc in enumerate(list(self.db.find('events'))):
            if ind%100==0 and ind/100>1:
                print ind
                df = pd.DataFrame(lst, columns=clmns)
                with open('csgoEvent.csv', 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=False)
                    
                lst = []
                
            if not doc.get('eventId', ''):# or not doc.get('mapPicker', '') or not doc.get('date', '') or not doc.get('teamARating', ''):
                continue
            
            if doc['eventId'] in visited:
                continue
            else:
                visited += [doc['eventId']]
                
    #        if doc.get('prizeDistr', '') not in ['', '[]']:
    #            doc['prizeDistr'] = ast.literal_eval(doc['prizeDistr'])
    #            if len(doc['prizeDistr'])>doc.get('teamAtnd') and doc.get('teamAtnd'):
    #                print doc['eventId'], len(doc['prizeDistr']), doc.get('teamAtnd')
    #            else:
    #                print doc['eventId'], len(doc['prizeDistr']), doc.get('teamAtnd')
                
            lst += [[doc.get('eventId'), doc.get('firPlace', 0), doc.get('eventType', 'asd'), doc.get('eventTitleFemale0'), doc.get('teamAtnd'),
                      doc.get('prizeUSD'), doc.get('maxPrizeUSD', 0), doc.get('dt2')]]
            
        df = pd.DataFrame(lst, columns=clmns)
        with open('csgoEvent.csv', 'a') as f:
            df.to_csv(f, sep=';', index=False, header=False)
            
    def createKefDF(self):
        #TODO createKefDF
        clmns = ['matchLinkId', 'teamAId', 'teamBId', 'kefs']#, 'teamQnt', 'prizeUSD', 'maxPrizeUSD', 'endDate']#еще дифьюзы и тд.
                
        df = pd.DataFrame(columns=clmns)
        
        with open('csgoKefs.csv', 'w') as f:
            df.to_csv(f, sep=';', index=False)
            
        lst = []
        for ind, doc in enumerate(list(self.db.find('matchKefs'))):
            if ind%100==0 and ind/100>1:
                print ind
                df = pd.DataFrame(lst, columns=clmns)
                with open('csgoKefs.csv', 'a') as f:
                    df.to_csv(f, sep=';', index=False, header=False)
                    
                lst = []
                
            kefs = doc.get('HLTVkefs')
            kefs = str(ast.literal_eval(kefs).values())
            lst += [[doc.get('matchLink').split('/')[2], doc.get('teamAId'), doc.get('teamBId'), kefs]]
            
        if lst:
            df = pd.DataFrame(lst, columns=clmns)
            with open('csgoKefs.csv', 'a') as f:
                df.to_csv(f, sep=';', index=False, header=False)
                
#    def createKefDF1(self):
#        #TODO createKefDF1
#        bookmekkers = set()
#        for ind, doc in enumerate(list(self.db.find('matchKefs'))):
#            kefs = doc.get('HLTVkefs')
#            kefs = set(ast.literal_eval(kefs).keys())
#    #        prelen = len(kefs)
#            kefs = set([i.replace('betting_provider', '').strip() for i in kefs])
#    #        if len(kefs)!=prelen or 'xbet' in kefs and '1xbet' in kefs:
#    #            print kefs
#    #            print set(ast.literal_eval(doc.get('HLTVkefs')).keys())
#            bookmekkers = bookmekkers.union(kefs)
#    #        print kefs
#    #    print bookmekkers
#    #    return 
#            
#    #    clmns = ['matchLinkId', 'teamAId', 'teamBId', 'kefs']#, 'teamQnt', 'prizeUSD', 'maxPrizeUSD', 'endDate']#еще дифьюзы и тд.
#        clmns = ['matchLinkId', 'teamAId', 'teamBId'] + list(bookmekkers)#, 'teamQnt', 'prizeUSD', 'maxPrizeUSD', 'endDate']#еще дифьюзы и тд.
#        print clmns
#    #    return
#        df = pd.DataFrame(columns=clmns)
#        
#        with open('csgoKefs1.csv', 'w') as f:
#            df.to_csv(f, sep=';', index=False)
#            
#        lst = []
#        for ind, doc in enumerate(list(self.db.find('matchKefs'))):
#            if ind%100==0 and ind/100>1:
#                print ind
#                df = pd.DataFrame(lst, columns=clmns)
#                with open('csgoKefs1.csv', 'a') as f:
#                    df.to_csv(f, sep=';', index=False, header=False)
#                    
#                lst = []
#                
#            kefs = ast.literal_eval(doc.get('HLTVkefs'))
#            
#            lst += [[doc.get('matchLink').split('/')[2], doc.get('teamAId'), doc.get('teamBId')]]
#            for book in bookmekkers:
#                if kefs.get(book, ''):
#                    lst[-1].append(str(kefs.get(book, '')))
#                    continue
#                elif kefs.get('%s betting_provider'%book, ''):
#                    lst[-1].append(str(kefs.get('%s betting_provider'%book, '')))
#                    continue
#                lst[-1].append('')
#                
#    #        kefs = set(ast.literal_eval(kefs).keys())
#    ##        prelen = len(kefs)
#    #        kefs = set([i.replace('betting_provider', '').strip() for i in kefs])
#    #        lst += [[doc.get('matchLink').split('/')[2], doc.get('teamAId'), doc.get('teamBId'), kefs]]
#            
#        if lst:
#            df = pd.DataFrame(lst, columns=clmns)
#            with open('csgoKefs1.csv', 'a') as f:
#                df.to_csv(f, sep=';', index=False, header=False)

#createKefDF()
#createKefDF1()
#createPrizeDistrDF()
#createMainDF()
#createEventDF()

#db['fullMatchHistory'].update_many({}, {'$rename': {'teamBDDeath': 'teamBDeath'}})
#createMainDF()
#db['fullMatchHistory'].update_many({'teamAposRat': {'$exists': False}}, {'$set': {'teamAposRat': -1}})
#db['fullMatchHistory'].update_many({'teamBposRat': {'$exists': False}}, {'$set': {'teamBposRat': -1}})
#print db['fullMatchHistory'].find({'teamARating': {'$exists': False}}).count()
#print db['fullMatchHistory'].find({'teamBRating': {'$exists': False}}).count()
#print db['fullMatchHistory'].find({'teamAposRat': {'$exists': False}}).count()
#print db['fullMatchHistory'].find({'teamBposRat': {'$exists': False}}).count()

db = DataBase()
#db.delete('teamsRating', {'teamUrlName': {'$exists': False}})
#for ind, _ in enumerate(list(db.find('teamsRating', {'position': {'$lte': 30}}))):
#    if ind%250==0:
#        print ind
#    if db.find('teamsRating', {'date': _['date'], 'position': _['position']}).count()>1:
#        print _

obj = DFCreator(db)
#obj.createHLTVRatingDF()
#obj.createMainDF()
##obj.createPrizeDistrDF()
##obj.createEventDF()
##obj.createKefDF()
##obj.createKefDF1()
db.close()
