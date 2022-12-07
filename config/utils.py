import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdflib import Variable
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
from sklearn import linear_model
# from scipy.stats.mstats import winsorize
from sklearn import preprocessing
import configparser
import os
from os import path
pd.options.mode.chained_assignment = None
from datetime import datetime


class EventStudy():
    def __init__(self, eventDate):

        risk_free = {
            '2020-03-09': 0.0054,#0.54%,
            '2020-03-16': 0.0073,#0.73%,
            '2021-07-19': 0.0119,#1.19%,
            '2021-11-26': 0.0163 #1.63%
            }

        self.industry_dict = {
            'Energy':50, #201
            'Construction_Materials':51, #170
            'Industrial':52,
            'Consumer_ProdServ':53,
            'Food_consumerGoods':54,
            'Financial':55, #526
            'Biotech':56, #530
            'Technology':57, #460
            'Utilities':59, #71
            'Real_Estate':60, #199
            'Education':63 #19
        }

        self.narrow_industry_dict = {
            'Automobile_Parts' : 531010.0, #40
            'Freight_Logistics': 524050.0, #50
            'Passenger_Transportation': 524060.0, #14
            'Hotels_Entertainment': 533010.0, #89
            'Media': 533020.0, #56
            'Food_Bev_Tabacco': [541010.0, 541020.0, 543010.0], #99
            'IT_hard' : [571010.0, 571020.0, 571040.0, 571050.0, 571060.0], 
            'IT_soft' : [572010.0, 573010.0, 574010.0],
            'bio_narrao' : 562020.0,

        }

        self.eventDate = eventDate
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.normalize = None

        self.window = [(0,0),(-1,1),(-3,3),(-5,5),(0,1),(0,3),(0,5)]
        self.pillar = ['EnvironmentalPillar','SocialPillar','GovernancePillar']
        self.rk = risk_free[self.eventDate]
        
        self.combined = pd.read_csv(self.config['fPath']['combined']).dropna()
        self.returnDF = pd.read_csv(self.config['fPath']['returnDF']).dropna()
        self.abReturn = self.AR(eventDate = self.eventDate, stock_list = list(set((self.combined)['ticker'])), data = self.returnDF)

        def BoardIndustry(x):
            return int(x[0:2])
        self.combined['industry'] = self.combined['industry'].astype(str)
        self.combined['boardIndustry'] = self.combined['industry'].apply(BoardIndustry)
        

    def get_returnDF(self):
        """
        Calculating Return for each companies
        Return return_df(type: DataFrame)
        Will be used at calculating Abnormal Return

        ** Recalculate if Risk free rate are changed
        
        """
        # read daily adjusted price csv and SP500 csv
        adjPrice = pd.read_csv(self.config['fPath']['adjPrice'])
        SP500 = pd.read_csv(self.config['fPath']['SP500'])
        SP500.index = SP500['Date']
        SP500 = SP500.drop(['Date'],axis=1)

        # Reset index as date and drop
        adjPrice.index = adjPrice['Date']
        adjPrice.drop(['Date'],axis=1, inplace = True)
        # Prevent the NAN problem
        adjPrice = adjPrice.rename(columns={'NAN': 'NANNN'})
        # add SP500 to the adjPrice
        adjPrice['^GSPC'] = SP500['^GSPC']

        # Calculate Abnormal Return
        return_df = adjPrice.pct_change().dropna()
        return_df['R_f'] = self.rk #0.0089 # risk free rate 0.0192

        # calculate market return
        return_df['Mkt_RF'] = return_df['^GSPC'] - return_df['R_f'] 
        # set the company name back to NAN
        return_df = return_df.rename(columns={'NANNN': 'NAN'})
        return_df.reset_index(inplace=True, drop=False)
        return return_df

    def AR(self, eventDate, stock_list, data, t1=7, t2=7):
        """
        Given eventday and stocklist,
        Calculating Abnormal Return for every stock
        Return abreturn(type: DataFrame)

        Fama French Version.
        """
        
        date_index = pd.DataFrame(data['Date'])
        abnreturn = {} 
        ff_data = pd.read_csv('/Users/kiyas/Desktop/ESG/journal/config/F-F_data.csv')
        ff_data = ff_data.drop(columns= ['Date'], axis = 1)
        data = pd.concat([data, ff_data], axis=1)

        for stock in stock_list:
            eventindex = int(date_index[date_index['Date']==eventDate].index.values)
            
            event_df = data.loc[eventindex-t1: eventindex+t2, ['Date', stock , 'R_f', 'Mkt_RF', 'SMB', 'HML']]
            estimation_df = data.loc[eventindex-200 : eventindex-t1-1, ['Date', stock , 'R_f', 'Mkt_RF', 'SMB', 'HML']]

            formula = stock + " - R_f ~ Mkt_RF + SMB + HML"
            beta_Mkt = sm.OLS.from_formula(formula, data=estimation_df).fit().params["Mkt_RF"]
            beta_SMB = sm.OLS.from_formula(formula, data=estimation_df).fit().params["SMB"]
            beta_HML = sm.OLS.from_formula(formula, data=estimation_df).fit().params["HML"]
            alpha = sm.OLS.from_formula(formula, data=estimation_df).fit().params["Intercept"]

            expectedreturn_eventwindow = ((event_df[['Mkt_RF']].values * beta_Mkt) + (event_df[['SMB']].values * beta_SMB) + (event_df[['HML']].values * beta_HML) + alpha)

            abnormal_return = event_df[stock].values - list(expectedreturn_eventwindow.flatten())
            abnreturn[stock] = abnormal_return

        
        abnreturn = pd.DataFrame(abnreturn)
        abnreturn.index = abnreturn.index-t1
        return abnreturn

    def CAR(self, abReturn,t1,t2):
        """
        Caculating CAR based on abnormal return
        t1 and t2 is the event window
        return res(type: DataFrame)
        """
        car = ((abReturn.iloc[7-t1:7+t2+1,:].cumsum()).iloc[-1:,:])
        car = (car.drop(car.columns[[0]],axis=1)).T
        car = car.rename(columns={car.columns.values[0]: 'y'})
        return car

    def generalES(self, normalize = False, exclude = False, subp = True):
        """
        Event Study Multivariate OLS Regression
        Via sm.OLS LinearModel
        21 Setting
        if exclude is True: exclude financial and energy industries
        two types of log save
        (1) res : two csv
        (2) log : model.summary2()
        """

        summary = pd.DataFrame(index = range(21), columns = range(3))
        if exclude:
            comb_df = self.combined.loc[~((self.combined['boardIndustry'] == 50) |
                                                (self.combined['boardIndustry'] == 55))]
            ticker = list(set(comb_df['ticker']))
            pathFile = self.config['fPath']['savePath'] + '/exclude'
        else:
            comb_df = self.combined
            ticker = list(set(self.combined['ticker']))
            pathFile = self.config['fPath']['savePath'] + '/general'


        for idwindow, w in enumerate(self.window):
            CAR = self.CAR(self.abReturn, w[0],w[1])
            y = pd.DataFrame(CAR[CAR.index.isin(ticker)],columns=['y'])
            tic = CAR[CAR.index.isin(ticker)].index
            comb_df = comb_df[comb_df['ticker'].isin(tic)]

            if subp:
                for ind, p in enumerate(self.pillar):
                    pShort = self.set_pShort(ind)
                    X = comb_df[['ticker', p, 'log_makvalt','BMRatio','ROA','asset_growth']]
                    X = pd.DataFrame(X)
                    X.index = X['ticker']
                    X = X.drop(columns=['ticker'], axis=1)

                    X = pd.concat([y,X],axis=1)
                    LinearModel = sm.OLS.from_formula('y ~ {subPillar} + log_makvalt + BMRatio + ROA + asset_growth'.format(subPillar = p), X)
                    res = LinearModel.fit()
                    
                    # write to summary table
                    summary = self.write_to_summary(summary, p, idwindow, ind, res)
                    # save the result
                    self.save_result(pathFile, pShort, res, w, 'X')
            
            # general
            else:
                X = comb_df[['ticker', 'ESG', 'log_makvalt','BMRatio','ROA','asset_growth']]
                X = pd.DataFrame(X)
                X.index = X['ticker']
                X = X.drop(columns=['ticker'], axis=1)
                X = pd.concat([y,X],axis=1)

                LinearModel = sm.OLS.from_formula('y ~ ESG + log_makvalt + BMRatio + ROA + asset_growth', X)
                res = LinearModel.fit()
                self.save_ESG(pathFile, res, w, 'X')
        return summary
            
    def industryES_big(self, industry, normalize = False, exclude = False, pillar = True):
        """
        Create dummies for industry

        """

        if exclude:
            comb_df = self.combined.loc[~((self.combined['boardIndustry'] == 50) |
                                                (self.combined['boardIndustry'] == 55))]
            pathFile = self.config['fPath']['savePath'] + '/exclu_' + industry

        else:
            comb_df = self.combined
            pathFile = self.config['fPath']['savePath'] + '/' + industry


        # three summary tables
        summary = pd.DataFrame(index = range(21), columns = range(3))
        dummy = pd.DataFrame(index = range(21), columns = range(3))
        intersum = pd.DataFrame(index = range(21), columns = range(3))
        
        industry_dummy = industry+'_dummy' # industry dummy column name
        tick = comb_df[comb_df['boardIndustry'] == self.industry_dict[industry]].index
        comb_df[industry_dummy] = 0
        comb_df.loc[tick, industry_dummy] = 1
        ticker = list(set(comb_df['ticker']))

        comb_df['InteractionE'] = comb_df['EnvironmentalPillar'] * comb_df[industry_dummy]
        comb_df['InteractionS'] = comb_df['SocialPillar'] * comb_df[industry_dummy]
        comb_df['InteractionG'] = comb_df['GovernancePillar'] * comb_df[industry_dummy]
        comb_df['InteractionESG'] = comb_df['ESG'] * comb_df[industry_dummy]


        for idset, i in enumerate(self.window):
            CAR = self.CAR(self.abReturn, i[0],i[1])
            y = pd.DataFrame(CAR[CAR.index.isin(ticker)],columns=['y'])

            tic = CAR[CAR.index.isin(ticker)].index
            comb_df = comb_df[comb_df['ticker'].isin(tic)]

            if pillar:
                for ind, p in enumerate(self.pillar):
                    
                    pShort = self.set_pShort(ind)
                    inter = self.set_interaction(ind)

                    X = comb_df[['ticker', p, industry_dummy, inter, 'log_makvalt','BMRatio','ROA','asset_growth','HHI']]
                    X = pd.DataFrame(X)
                    X.index = X['ticker']
                    X = X.drop(columns=['ticker'], axis=1)

                    X = pd.concat([y,X],axis=1)
                    LinearModel = sm.OLS.from_formula('y ~ {subPillar} + {industry} + {interaction} + \
                                    log_makvalt + BMRatio + ROA + asset_growth + HHI'.format(subPillar = p, industry = industry_dummy, interaction = inter), X)
                    res = LinearModel.fit()
                    
                    # write to summary
                    summary = self.write_to_summary(summary, p, idset, ind, res)
                    dummy = self.write_to_summary(dummy, industry_dummy, idset, ind, res)
                    interaction = self.write_to_summary(intersum, inter, idset, ind, res)

                    self.save_result(pathFile, pShort, res, i, len(tick))

                
            else:
                X = comb_df[['ticker', "ESG", industry_dummy, 'InteractionESG', 'log_makvalt','BMRatio','ROA','asset_growth', 'HHI']]
                X = pd.DataFrame(X)
                X.index = X['ticker']
                X = X.drop(columns=['ticker'], axis=1)

                X = pd.concat([y,X],axis=1)
                LinearModel = sm.OLS.from_formula('y ~ ESG + {industry} + InteractionESG + \
                                log_makvalt + BMRatio + ROA + asset_growth' + 'HHI'.format(industry = industry_dummy), X)
                res = LinearModel.fit()
                self.save_ESG(pathFile, res, i, len(tick))
                summary, dummy, interaction = None, None, None
        return summary, dummy, interaction
    
    def controlES_inter(self, variable, exclude = False, extremeGroup = False, pillar = True):

        """
        Experiments for control variables: ['log_makvalt','BMRatio','ROA','asset_growth']
        extremeGroup: cut to 3 parts and take the extreme
        """

        # summary tables for coefficients, dummy variables and interaction
        summary = pd.DataFrame(index = range(21), columns = range(3))
        dummy = pd.DataFrame(index = range(21), columns = range(3))
        intersum = pd.DataFrame(index = range(21), columns = range(3))

        # excluding financial and energy industries
        if exclude:
            comb_df = self.combined.loc[~((self.combined['boardIndustry'] == 50) |
                                                (self.combined['boardIndustry'] == 55))]
            pathFile = self.config['fPath']['savePath'] + '/exclu_{con}'.format(con = variable)
        else:
            comb_df = self.combined
            pathFile = self.config['fPath']['savePath'] + '/{con}'.format(con = variable)

        # managing the extremeGroup setting
        control_dummy = variable + '_dummy'
        if extremeGroup:
            threeDivide = np.array_split(comb_df[variable].sort_values(ascending = True), 3)
            extremeSmall = list(threeDivide[0].index)
            extremeLarge = list(threeDivide[2].index)
            comb_df[control_dummy] = 0
            comb_df.loc[comb_df.index.isin(extremeLarge), control_dummy] = 1
            comb_df = comb_df[comb_df.index.isin(extremeLarge + extremeSmall)]
            ticker = list(set(comb_df['ticker']))
        else:
            median = comb_df[variable].median()
            comb_df[control_dummy] = 0
            comb_df.loc[comb_df[variable] > median, control_dummy] = 1
            ticker = list(set(comb_df['ticker']))

        comb_df['InteractionE'] = comb_df['EnvironmentalPillar'] * comb_df[control_dummy]
        comb_df['InteractionS'] = comb_df['SocialPillar'] * comb_df[control_dummy]
        comb_df['InteractionG'] = comb_df['GovernancePillar'] * comb_df[control_dummy]
        comb_df['InteractionESG'] = comb_df['ESG'] * comb_df[control_dummy]

        for idset, i in enumerate(self.window):
            CAR = self.CAR(self.abReturn, i[0],i[1])
            y = pd.DataFrame(CAR[CAR.index.isin(ticker)],columns=['y'])

            tic = CAR[CAR.index.isin(ticker)].index
            comb_df = comb_df[comb_df['ticker'].isin(tic)]
            if pillar:
                for ind, p in enumerate(self.pillar):
                    
                    pShort = self.set_pShort(ind)
                    inter = self.set_interaction(ind)

                    X = comb_df[['ticker', p, control_dummy, inter, 'log_makvalt','BMRatio','ROA','asset_growth', 'HHI']]
                    X = pd.DataFrame(X)
                    X.index = X['ticker']
                    X = X.drop(columns=['ticker'], axis=1)

                    X = pd.concat([y,X],axis=1)
                    LinearModel = sm.OLS.from_formula('y ~ {subPillar} + {control} + {interaction} + \
                                    log_makvalt + BMRatio + ROA + asset_growth + HHI'.format(subPillar = p, control = control_dummy, interaction = inter), X)
                    res = LinearModel.fit()
                    
                    # write to summary
                    summary = self.write_to_summary(summary, p, idset, ind, res)
                    dummy = self.write_to_summary(dummy, control_dummy, idset, ind, res)
                    interaction = self.write_to_summary(intersum, inter, idset, ind, res)

                    self.save_result(pathFile, pShort, res, i, 'X')
        
            else:
                X = comb_df[['ticker', "ESG", control_dummy, 'InteractionESG', 'log_makvalt','BMRatio','ROA','asset_growth', 'HHI']]
                X = pd.DataFrame(X)
                X.index = X['ticker']
                X = X.drop(columns=['ticker'], axis=1)

                X = pd.concat([y,X],axis=1)
                LinearModel = sm.OLS.from_formula('y ~ ESG + {control} + InteractionESG + \
                                log_makvalt + BMRatio + ROA + asset_growth + HHI'.format(control = control_dummy), X)
                res = LinearModel.fit()
                self.save_ESG(pathFile, res, i, 'X')
                summary, dummy, interaction = None, None, None
        return summary, dummy, interaction

    def get_combined(self):
        return self.combined

    def get_AR(self):
        return self.abReturn
    
    def get_return(self):
        return self.returnDF

    def write_to_summary(self, df, obj, idset, ind, res):
        # write to summary
        
        df.loc[[3*idset+ind],[0]] = res.params[obj]
        df.loc[[3*idset+ind],[1]] = res.tvalues[obj]
        
        # *p < .1; **p < .05; ***p < .01
        if res.pvalues[obj] < 0.1:
            df.loc[[3*idset+ind],[2]] = "*"+str(res.pvalues[obj])
            if res.pvalues[obj] < 0.05:
                df.loc[[3*idset+ind],[2]] = "**"+str(res.pvalues[obj])
                if res.pvalues[obj] < 0.01:
                    df.loc[[3*idset+ind],[2]] = "***"+str(res.pvalues[obj])
        else:
            df.loc[[3*idset+ind],[2]] = str(res.pvalues[obj])

        return df

    def save_result(self, pathFile, pShort, res, i, industry_num):
        # check if directory exists
        if not path.exists(pathFile):
            os.makedirs(pathFile)

        # name the file according to their date, window, subpillar
        f = open(pathFile+'/{date}_({w1}{w2})_{subPillar}.txt'.format(date=self.eventDate, w1 = i[0], w2=i[1], subPillar = pShort), 'w')
        f.write(str(res.summary2()))
        f.close()
        
        val = pd.DataFrame({0: res.params, 1: res.tvalues , 2: res.pvalues})
        table = pd.DataFrame(res.summary2().tables[0])
        conf = pd.DataFrame([['adjustedR2', res.rsquared_adj], ['#obs', int(table[1][3])], [ 'industry_num', industry_num]]).T
        val = pd.concat([val, conf], axis = 0)
        val = val.rename(columns={0: 'coef', 1: 'T_val', 2: 'P_val'})

        val.to_csv(pathFile+'/{date}_({w1}{w2})_{subPillar}.csv'.format(date=self.eventDate, w1 = i[0], w2=i[1], subPillar = pShort))

    def set_pShort(self, ind):
        if ind==0:
            return 'E'
        elif ind==1:
            return 'S'
        else:
            return 'G'
    
    def set_interaction(self, ind):
        if ind==0:
            return 'InteractionE'
        elif ind==1:
            return 'InteractionS'
        else:
            return 'InteractionG'

    def save_ESG(self, pathFile, res, i, industry_num):
        pathLog = pathFile + '/log'
        pathRes = pathFile + '/res'
        if not path.exists(pathFile):
            os.makedirs(pathFile)
        if not path.exists(pathLog):
            os.makedirs(pathLog)
            os.makedirs(pathRes)
        
        f = open(pathLog+'/{date}_({w1}{w2})_ESG.txt'.format(date=self.eventDate, w1 = i[0], w2=i[1]), 'w')
        f.write(str(res.summary2()))
        f.close()

        val = pd.DataFrame({'Coef': res.params, 'tValue': res.tvalues, 'pValue': res.pvalues})
        val.to_csv(pathRes+'/{date}_({w1}{w2})_ESG_value.csv'.format(date=self.eventDate, w1 = i[0], w2=i[1]))
        table = pd.DataFrame(res.summary2().tables[0])
        conf = pd.DataFrame([res.rsquared_adj, int(table[1][3]), self.eventDate, industry_num],
                                index = ['adjustedR2','#obs','eventDay', 'industry_num'], columns=['conf'])
        conf.to_csv(pathRes+'/{date}_({w1}{w2})_ESG_config.csv'.format(date=self.eventDate, w1 = i[0], w2=i[1]))

def eventDate():
    #重大事件日期
    EventsList = ['OB Mid1','OB Mid2','Delta','Omicorn']
    date = ['2020-03-09','2020-03-16','2021-07-19','2021-11-26']
    date_df = pd.DataFrame(np.zeros((4,2)), columns=['date','eventlist'])
    date_df['date'] = date
    date_df['eventlist'] = EventsList
    return date_df
    

