import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unidecode

def preprocessingResidentData(fileName):
    """
    Function for reading and preprocessing the resident numbers per town
    """
    jarasok = pd.read_csv(fileName)
    # Delete unused columns
    toDeleteList = [
            'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10',
            'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16']
    jarasok.drop(toDeleteList, axis=1, inplace=True)
    
    # Rename the 'telepules' column
    jarasok.rename(columns={'telepules' : 'telnev'}, inplace = True)
    
    # Transforming names
    #jarasok['telnev'] = jarasok['telnev'].apply(tsm.asciify)
    jarasok['telnev'] = jarasok['telnev'].apply(asciify)

    return jarasok

def checkModelValidity(classifierOut, validationTarget):

    get = classifierOut.tolist()
    target = validationTarget.tolist()
    
    scoreMatrix = pd.DataFrame(np.zeros(shape = (2,2)), columns = ['Target_0', 'Target_1'], index = ['Get_0', 'Get_1'], dtype = 'int32')
    
    scoreMatrix.at['Get_0', 'Target_0'] += 1
    
    
    for i in range(1, len(get)):
        if((target[i] == 0) and (get[i] == 0)):
            scoreMatrix.at['Get_0', 'Target_0'] += 1
        if((target[i] == 0) and (get[i] == 1)):
            scoreMatrix.at['Get_1', 'Target_0'] += 1
        if((target[i] == 1) and (get[i] == 0)):
            scoreMatrix.at['Get_0', 'Target_1'] += 1
        if((target[i] == 1) and (get[i] == 1)):
            scoreMatrix.at['Get_1', 'Target_1'] += 1

    return scoreMatrix

def d10Mapper(inp):
    if(inp.strip() == 'kozmuves ivovizellatas'):
        return 1
    elif(inp.strip() == 'sajat ivoviz kut'):
        return 0
    else:
        return -1

def d11Mapper(inp):
    if(inp.strip() == 'kozmuves szennyvizhalozat'):
        return 1
    elif(inp.strip() == 'szennyvizgyujtes a telken es szennyviz szippantas'):
        return 0
    else:
        return -1
 
def dolgMapper(status):
    if(status.strip() == 'Alkalmazott'):
        return 1
    elif(status.strip() == 'Munkan??lk??li'):
        return 0
    else:
        print("unknown dolg value", status)
        return -1
 
def schoolMapper(schoolType):
    schoolMap = {
            'Kevesebb, mint 8 ??ltal??nos' : 1, 
            '8 ??ltal??nos' : 2,                                                                                                                                        
            'Szakmunk??sk??pz??; szakk??pz??s ??retts??gi n??lk??l' : 3,
            'Szakk??z??piskolai ??retts??gi; szakk??pz??st k??vet?? ??retts??gi' : 4,
            'Gimn??ziumi ??retts??gi' : 5,
            '??retts??git k??vet??, fels??fokra nem akkredit??lt szakk??pz??s; k??z??pfok?? technikum' : 6,
            'Akkredit??lt fels??fok?? szakk??pz??s; fels??fok?? technikum': 7,
            'F??iskola / BA / BSc' : 8,
            'Egyetem / MA / MSc' :9 ,
            'VM' : None 
    }
    return schoolMap[schoolType]

def egyuttMapper(num):
    if(num.strip() == 'Egyed??l ??l'):
        return 1
    else:
        return num

def nemeMapper(neme):
    if(neme.strip() == 'N??'):
        return 2
    elif(neme.strip() == 'F??rfi'):
        return 1
    else:
        return None

def szulevMapper(szulev):
    return 2018-szulev

def asciify(inp):
    return unidecode.unidecode(inp).lower().strip()

def restoreTown(halfName):
    if(halfName == 'kisk?ros'):
        return 'kiskoros'
    elif(halfName == 'hej?papi'):
        return 'hejopapi'
    elif(halfName == 'mez?keresztes'):
        return 'mezokeresztes'
    elif(halfName == 'szendr?'):
        return 'szendro'
    elif(halfName == 'hodmez?vasarhely'):
        return 'hodmezovasarhely'
    elif(halfName == 'cs?sz'):
        return 'csosz'
    elif(halfName == 'gy?r'):
        return 'gyor'
    elif(halfName == 'beseny?telek'):
        return 'besenyotelek'
    elif(halfName == 'sz?csi'):
        return 'szucsi'
    elif(halfName == 'godoll?'):
        return 'godollo'
    elif(halfName == 'jaszkarajen?'):
        return 'jaszkarajeno'
    elif(halfName == '?rbottyan'):
        return 'orbottyan'
    elif(halfName == 'tapioszecs?'):
        return 'tapioszecso'
    elif(halfName == 'gavavencsell?'):
        return 'gavavencsello'
    elif(halfName == '?cseny'):
        return 'ocseny'
    elif(halfName == 'zalalov?'):
        return 'zalalovo'
    elif('budapest' in halfName): # Budapestet most homogennek vesszuk munka szempontjabol
        return 'budapest'
    else:
        return halfName



                       
