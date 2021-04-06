import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unidecode

import tsm_utils as tsm

def getMaxResidentNumber(df):
    """
    This function returns the maximal resident number in the catalog
    """
    maxRes = df['lakossag'].max()
    return maxRes

def getMinresidentNumber(df):
    """
    """
    minRes = df['lakossag'].min()
    return minRes

def normaliseResidentNumber(df, maxRes, minRes):
    """
    Function for normalising resident numbers into 0..1 interval
    """
    #minRes = df['lakossag'].min()
    #maxRes = df['lakossag'].max()

    slope = 1.0 / float(maxRes - minRes) 
    inters = float(-minRes) / float(maxRes - minRes)
    
    #print('MIN: ')
    #print(minRes)
    #print('MAX: ')
    #print(maxRes)
    
    #df['lakossag'] = df['lakossag'] / maxRes
    df['lakossag'] = slope * df['lakossag'] + inters

    return df

def deleteUnusedColumns(df):
    toDeleteList = [
            'htfovi', 'neme2', 'szulev2', 'szulho2', 'htfovi2', 'neme3', 'szulev3', 
            'szulho3', 'htfovi3', 'neme4', 'szulev4', 'szulho4', 'htfovi4', 'neme5', 
            'szulev5', 'szulho5', 'htfovi5', 'neme6', 'szulev6', 'szulho6', 'htfovi6'] 
    df.drop(toDeleteList, axis=1, inplace=True)

    return df

def typesAndIndex(df):
    """
    Function which sets some column types and the index column
    """
    df['szulev']  = df['szulev'].astype(int)
    df['htnetto'] = df['htnetto'].astype(float)
    df['sorszam'] = df['sorszam'].astype(int)
    df.set_index('sorszam', drop = False, inplace = True)  # Setting sorszam column as index

    return df

def preprocessSzulev(df):
    """
    Function to normalise 'eletpalya' variable from szuletesi ev
    """
    df['eletkor'] = df['szulev'].apply(tsm.szulevMapper)
    
    minAge = df['eletkor'].min()
    maxAge = df['eletkor'].max()
    slope = 1.0 / float(maxAge - minAge) 
    inters = float(-minAge) / float(maxAge - minAge)
    
    df['eletpalya'] = slope * df['eletkor'] + inters
    
    #minStatus = df[df['eletkor'] == minAge].dolg 
    #maxStatus = df[df['eletkor'] == maxAge].dolg 
    #minNorm = df['eletpalya'].min()
    #maxNorm = df['eletpalya'].max()
    #print('MIN: ')
    #print(minAge)
    #print(minNorm)
    #print(minStatus)
    #print('MAX: ')
    #print(maxAge)
    #print(maxNorm)
    #print(maxStatus)
    #print(features[['eletpalya', 'eletkor']].head(50))

    return df
 
def preprocessIskola(df):
    #print(features.iskola.unique())
    #print(features['iskola'].value_counts())
    df.drop(df[df.iskola == 'VM'].index, inplace=True)  # Deleting rows where school type is 'VM'
    df['iskola'] = df['iskola'].apply(tsm.schoolMapper) # Convertion iskola into numerical values

    return df

def preprocessEgyutt(df):
    """
    Function for preprocessing how meny people live together
    """
    #print(features['egyutt'].value_counts())
    df['egyutt'] = df['egyutt'].apply(tsm.egyuttMapper)

    return df

def preprocessIncome(df, numOutliers):
    """
    Function for preprocessing income and adding net income
    """
    #print('Min kereset: ', df['htnetto'].min())
    #print('Max kereset: ', df['htnetto'].max())
    
    # Droping out outiers
    for i in range(0, numOutliers):
        df.drop(df[df['htnetto'] == df['htnetto'].max()].index, inplace=True)
    #features.drop(features[features['htnetto'] == features['htnetto'].max()].index, inplace=True)
    
    # Logarithm as column
    df['lognetto'] = df['htnetto'].apply(np.log)
    
    # Checking normality by sight
    #features.hist(column='htnetto', bins=10)
    #features.hist(column='lognetto', bins=10)
    #plt.show()

    return df

def preprocessPiping(df):

    df['d10'] = df['d10'].apply(tsm.asciify)
    #TODO: this filtering might be useful...
    #df.drop(df[df['d10'] == 'nv'].index, inplace=True)  # Deleting rows where school type is 'VM'
    df['d10'] = df['d10'].apply(tsm.d10Mapper)
    #print(df['d10'].value_counts())
    
    df['d11'] = df['d11'].apply(tsm.asciify)
    #TODO: this filtering might be useful...
    #df.drop(df[df['d11'] == 'nv'].index, inplace=True)  # Deleting rows where school type is 'VM'
    df['d11'] = df['d11'].apply(tsm.d11Mapper)
    #print(df['d11'].value_counts())

    df['piping'] = (df['d10'] + df['d11']) / float(2.0)

    return df

def preprocessSex(df):
    df['neme'] = df['neme'].apply(tsm.nemeMapper)
    #print(df['neme'].value_counts())

    return df

def preprocessTownName(df):
    df['telnev'] = df['telnev'].apply(tsm.asciify)
    df['telnev'] = df['telnev'].apply(tsm.restoreTown)
    #print(df['telnev'])

    return df

def preprocessDolg(df):
    """
    Function for preprocessing the column for work status
    """
    #print(merged['dolg'].unique())
    df['dolg'] = df['dolg'].apply(tsm.dolgMapper)

    return df

def preprocessData(features, jarasok, dropOut):

    # Preprocessing the table
    features = deleteUnusedColumns(features) # Deleting unused columns
    features = typesAndIndex(features) # Setting column types
    features = preprocessSzulev(features)
    features = preprocessIskola(features)
    features = preprocessEgyutt(features)
    features = preprocessIncome(features, dropOut)
    features = preprocessPiping(features)
    features = preprocessSex(features)
    features = preprocessTownName(features)
    # Adding normalised residents data
    merged = pd.merge(features, jarasok, on='telnev') # Merging to put resident numbers into the dataframe
    maxRes = getMaxResidentNumber(jarasok)
    minRes = getMinresidentNumber(jarasok)
    merged = normaliseResidentNumber(merged, maxRes, minRes)

    return merged
 
