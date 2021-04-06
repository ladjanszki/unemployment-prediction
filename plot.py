import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unidecode
from itertools import product
from datetime import datetime

toPlot = pd.read_csv('2018_06_05_17_35_39_out.csv')

print(toPlot.head(30))
print(toPlot.columns.values)
print(toPlot.dtypes)
toPlot['plotlabel'] = toPlot['plotlabel'].astype(str)
print(toPlot.dtypes)
 
# Plotting the results

myDpi = 100
xres = 1920
yres = 981

xsize = float(xres) / float(myDpi)
ysize = float(yres) / float(myDpi)


#fig = plt.figure()
fig = plt.figure(figsize=(xsize,ysize))
#fig = plt.figure(figsize=(10,12))
ax = plt.subplot(111)
#ax = plt.subplot(1, 1, 1)
ax.bar(toPlot['plotlabel'], toPlot['svc_likelihood'])
ax.set_xticks(toPlot['plotlabel'])
ax.set_xticklabels(toPlot['plotlabel'], rotation=90) 

plt.show()
fig.savefig('svc.png')
 
