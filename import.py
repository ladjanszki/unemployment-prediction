import rpy2
from rpy2.robjects.packages import importr # For importing R objects 
import rpy2.robjects as robjects # For interaction with the R environment
from pandas import DataFrame
from rpy2.robjects import pandas2ri

# For convertiong dataframes between Python and R
pandas2ri.activate()

# Get the version
#print(rpy2.__version__)

# R imports
base = importr('base')
utils = importr('utils')
foreign = importr('foreign')

# Reading SPSS files
path = '/home/borneo/BPM/2017_2018_2/tsm_2/verseny/orig_data'
fileName = 'train_verseny.sav'
fullPath = path + '/' + fileName

# Runing the R command
#robjects.r('''trainData <- read.spss('/home/borneo/BPM/2017_2018_2/tsm_2/verseny/orig_data/train_verseny.sav', to.data.frame = TRUE)''')
robjects.r('''trainData <- read.spss('{}', to.data.frame = TRUE)'''.format(fullPath))
trainData = robjects.r['trainData']
#print(trainData.columns.values)
#print(type(trainData))


