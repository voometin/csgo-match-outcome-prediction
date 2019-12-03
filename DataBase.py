# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 08:33:19 2017

@author: Андрей
"""
import pymongo

#client = pymongo.MongoClient(port=33333)
#db = client['csgo']
#client.close()

class DataBase():
    def __init__(self, client=pymongo.MongoClient(port=33333), dbname='csgo'):
        if isinstance(client, pymongo.MongoClient):
            self.client = client
            self.db = self.client[dbname]
        else:
            raise 'unknow client or DB'
            
    def close(self):
        if isinstance(self.client, pymongo.MongoClient):
            self.client.close()
            print 'client has been closed successfuly'
        else:
            raise 'unknow client or DB'
            
    def find(self, collection, condition={}, filtering={}, multi=True, sort='', distinct_field=''):
        if isinstance(self.client, pymongo.MongoClient):
            if distinct_field:
                if condition:
                    return self.db[collection].find(condition).distinct(distinct_field)
                return self.db[collection].distinct(distinct_field)
            elif multi:
                if filtering:
                    if sort:
                        return self.db[collection].find(condition, filtering).sort(sort)
                    return self.db[collection].find(condition, filtering)
                else:
                    if sort:
                        return self.db[collection].find(condition).sort(sort)
                    return self.db[collection].find(condition)
            elif sort:
                if filtering:
                    return self.db[collection].find_one(condition, filtering).sort(sort)
                return self.db[collection].find_one(condition).sort(sort)
            if filtering:
                return self.db[collection].find_one(condition, filtering)
            return self.db[collection].find_one(condition)
        else:
            raise 'unknow client or DB'
            
    def update(self, collection, condition, update_vals, upsert=True, multi=False):
        if isinstance(self.client, pymongo.MongoClient):
            if multi:
                return self.db[collection].update_many(condition, {'$set': update_vals}, upsert=upsert)
            else:
                return self.db[collection].update_one(condition, {'$set': update_vals}, upsert=upsert)
        else:
            raise 'unknow client or DB'
            
    def insert(self, collection, insert_vals):
        if isinstance(self.client, pymongo.MongoClient):
            return self.db[collection].insert_one(insert_vals)
        else:
            raise 'unknow client or DB'
            
    def delete(self, collection, condition={}, multi=True):
        if isinstance(self.client, pymongo.MongoClient):
            if multi:
                return self.db[collection].delete_many(condition)
            else:
                return self.db[collection].delete_one(condition)
        else:
            raise 'unknow client or DB'

    def getMatchHistoryMatchesToScan(self, updateAll=False):
        #TODO getMatchHistoryMatchesToScan
        alldocs = list(self.db.find('matchHistory', {}, {'matchlinkId': 1, 'matchType': 1}))
        if not updateAll:
            filledLst = set(self.db.find('fullMatchHistory', distinct_field='matchlinkId'))
            allMatched = set([i['matchlinkId'] for i in alldocs])
            findMatch = list(allMatched-filledLst)
            print len(findMatch)
            alldocs = list(self.db.find('matchHistory', {'matchlinkId': {'$in' : findMatch}}, {'matchlinkId': 1, 'matchType': 1}))
            
        return alldocs
    
    def getMapHistoryMatchesToScan(self, updateAll=False):
        #TODO getMapHistoryMatchesToScan
        alldocs = list(self.db.find('matchMapStatsHistory', {}, {'maplinkId': 1, 'matchType': 1}))
        if not updateAll:
            filledLst = set(self.db.find('fullMatchHistory', distinct_field='maplinkId'))
            allMatched = set([i['maplinkId'] for i in alldocs])
            findMatch = list(allMatched-filledLst)
            print len(findMatch)
    #        print findMatch[0], type(findMatch[0])
            alldocs = list(self.db.find('matchMapStatsHistory', {'maplinkId': {'$in' : findMatch}}, {'maplinkId': 1, 'matchType': 1}))
    #        print len(alldocs)
            
        return alldocs
    
#db = DataBase()
#db.close()