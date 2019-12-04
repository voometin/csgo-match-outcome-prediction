# csgo-match-outcome-prediction

This repo contains multiple parsing modules that extract features from HLTV (ratings, past match stats, info about events, ...). In ./data/ folder there is multiple ....csv files that were gotten from DFCreator and during preprocessing in csgoMain.ipynb. During execution of csgoMain.ipynb were calculated multiple features only for HLTV and elo0 rating system variation. That's why other folders in ./rating_systems/ are empty. In order to get features based on other rating systems you would do it yourself (you would start csgoMain notebook stating from the cell containing pd.read_csv('./data/preprocessedMAIN.csv', sep=';')). Due to file size upload restriction on github, I made a public dataset on Kaggle - https://www.kaggle.com/peacemaket/csgo-outcome-prediction

Parsing modules:
	DFCreator - module for creating different ....csv files for further analysing, preprocessing and building the models for predicting the outcome of the game.
	DataBase - the wrapper module for different databases with a single interface (insert, find, delete). I used only MongoDB.
	Parser - module for generating GET/POST requests to any website with some parameters (cookies, headers, proxy, ...) and extracting HLTV matches coeficients from multiple bookmakers.
	Event - module for parsing and preprocessing any info about events from HLTV (dates, title, number of participents, prizePool, ...).
	HistoryMatchStats - module for extracting and adding past matches and maps statistics from HLTV to database (MongoDB in my case).
	UpcomingMatchStats - module for extracting and adding upcoming and live matches and maps any information from HLTV to database.
	Preprocessor - module for event preprocessing mainly (link qualifing tournament with main tournament. E.G. link 'CS:GO Asia Championships 2019 China Qualifier' with first place prize 'CAC 2019' to 'CS:GO Asia Championships 2019')
	Ratings - module for extracting and adding HLTV rating from HLTV to database.

Different rating systems:
	./CSGO_ratings_prediction/EloTestNewVersion - implementation of Elo rating system
	./CSGO_ratings_prediction/GlickoNewVersion - implementation of Glicko rating system
	./CSGO_ratings_prediction/Glicko2NewVersion - implementation of Glicko2 rating system
	./CSGO_ratings_prediction/TrueSkill123NewVersion - implementation of TrueSkill rating system

Main modules for processing, feature engineering, model prediction and evaluation:
	ipynbHelper - module containing helper and preprocessing functions for csgoMain.ipynb noteboook.
	csgoMain.ipynb - the main preprocessing and feature engineering notebook. 
	py27Stack.ipynb - notebook containing models building and evaluation the model based on bookmakers coeficients and different betting strategies.
