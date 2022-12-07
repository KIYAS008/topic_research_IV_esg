"""
Summarize the experiments that are statistically siginificant to an excel sheet.

"""

import pandas as pd
import xlsxwriter as xl
template = pd.read_csv('/ESG/1020_journal/1204_summary/summary_template.csv')
title = 'Unnamed: 0'
basePath = '/ESG/1020_journal/1204_summary'
windows = ['00', '05', '-33', '01']
industries = ['Materials', 'Industrials', 'Healthcare', 'Consumer_Staple']
eventdays = ['2020-03-09', '2020-03-16', '2021-07-19', '2021-11-26']

def myRound(x, num):
    """
    round the numbers to 3rd decimal place, 
    transfer to scientific notation if the figures are even smaller.
    """
    if x < 0.001 and x > 0:
        return '{:.2e}'.format(x)
    elif x > -0.001 and x < 0:
        return '{:.2e}'.format(x)
    else:
        return round(x,3)

for industry in industries:
    industry_save_path = '_summary.xlsx'
    industry_save_path = '{}'.format(industry)+industry_save_path
    writer = pd.ExcelWriter(industry_save_path, engine='xlsxwriter')

    for ind, window in enumerate(windows):
        Ftable = []
        template = pd.read_csv('/Users/kiyas/Desktop/ESG/1020_journal/1204_summary/summary_template.csv')
        for eventday in eventdays:
            for pillar in ['E','S','G']:
                path = basePath + '/{industry}/{eventday}_({window})_{pillar}.csv'.format(industry = industry,
                    eventday = eventday, window = window, pillar = pillar)
                test = pd.read_csv(path)
                
                liz = []
                coef = myRound(float(test.T[1]['coef']), 3)
                # Add * to the specifications that are statistically siginificant 
                if float(test.T[1]['P_val']) < 0.01:
                    coef = (str(coef)+"***")
                elif float(test.T[1]['P_val']) < 0.05:
                    coef = (str(coef)+"**")
                elif float(test.T[1]['P_val']) < 0.1:
                    coef = (str(coef)+'*')

                if pillar == 'E':
                    liz.append(coef)
                    liz.append(myRound(float(test.T[1]['T_val']), 3))
                    liz.extend(['','','',''])
                elif pillar =='S':
                    liz.extend(['',''])
                    liz.append(coef)
                    liz.append(myRound(float(test.T[1]['T_val']), 3))
                    liz.extend(['',''])
                else:
                    liz.extend(['','','',''])
                    liz.append(coef)
                    liz.append(myRound(float(test.T[1]['T_val']), 3))

                for i in range(2,9):
                    coef = myRound(float(test.T[i]['coef']), 3)
                    tvalue = myRound(float(test.T[i]['T_val']), 3)
                    if float(test.T[i]['P_val']) < 0.01:
                        liz.append(str(coef)+"***")
                    elif float(test.T[i]['P_val']) < 0.05:
                        liz.append(str(coef)+"**")
                    elif float(test.T[i]['P_val']) < 0.1:
                        liz.append(str(coef)+"*")
                    else:
                        liz.append(coef)
                    liz.append(tvalue)
                    
                liz.extend([myRound(float(test.T[0]['coef']), 3), myRound(float(test.T[0]['T_val']), 3), '',
                    int(float(test.T[10]['T_val'])), int(float(test.T[10]['P_val'])), myRound(float(test.T[10]['coef'])*100, 3)])
                Ftable.append(liz)
        
        template.iloc[:,1:] = pd.DataFrame(Ftable).T
        template.to_excel(writer, sheet_name=window)
    workbook  = writer.book
    workbook.close()
