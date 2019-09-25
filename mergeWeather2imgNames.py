#python3 merg... > 2013imgs2weather
import pandas as pd

d2 = pd.read_csv("2013Imgs", header = None)  # this is the full image names only
d1 = pd.read_csv("2013imgname2weather", header = None) #this is the weather data: imgName, ADDs

for i in d2[0]: #d2 is a datafram with one column
    try:
        print(i, list(d1[d1[0]==i.split('/')[-1].split()[0]].values[0][1:4]))
        
    except:
        continue

