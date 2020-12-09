import numpy as np
import pandas as pd
import pdb
import sys

class Ellipsoid:
    def __init__(self,*,model:str='wgs84',a:float=0 , f:float=0 ):
        if model =='wgs84':
            self.a = 6378137
            self.f = 1 / 298.257223563
            self.b = self.a * (1-self.f)
            self.e = np.sqrt(2*self.f -self.f**2)
        if model =='grs80':
            self.a = 6378137
            self.f = 1 / 298.257222101
            self.b = self.a * (1-self.f)
            self.e = np.sqrt(2*self.f -self.f**2)
        if model == 'bessel':
            self.a = 6377397.155
            self.f = 1 / 299.152813
            self.b = self.a * (1-self.f)
            self.e = np.sqrt(2*self.f -self.f**2)
        if model == 'other':
            self.a = a 
            self.f = f
            self.b = self.a * (1-self.f)
            self.e = np.sqrt(2*self.f -self.f**2)

def latlonalt2ecef(lat,lon,h,*,ell=Ellipsoid()):
    lat = np.radians(lat)
    lon = np.radians(lon) 
    N = ell.a/np.sqrt(1-((ell.e**2)*(np.sin(lat)**2)))
    x = (N + h) * np.cos(lat)*np.cos(lon)
    y = (N + h) * np.cos(lat)*np.sin(lon)
    z = (N*(1-ell.e**2)+h)*np.sin(lat)
    return x,y,z

def ecef2enu(x,y,z,lat0,lon0,h0,*,ell=Ellipsoid()):
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    N = ell.a/np.sqrt(1-((ell.e**2)*(np.sin(lat0)**2)))
    x0 = (N + h0)*np.cos(lat0)*np.cos(lon0)
    y0 = (N + h0)*np.cos(lat0)*np.sin(lon0)
    z0 = (N*(1-ell.e**2)+h0)*np.sin(lat0)
    t = (x-x0)*np.cos(lon0) + (y-y0)*np.sin(lon0)
    e = (y-y0)*np.cos(lon0) - (x-x0)*np.sin(lon0)
    n = (z-z0)*np.cos(lat0) - t*np.sin(lat0) 
    u = (z-z0)*np.sin(lat0) + t*np.cos(lat0)
    return e,n,u

def latlonalt2enu(lat,lon,h,lat0,lon0,h0,*,ell=Ellipsoid()):
    Ell = ell
    x,y,z = latlonalt2ecef(lat,lon,h,ell=Ell)
    e,n,u = ecef2enu(x,y,z,lat0,lon0,h0,ell=Ell)
    return e,n,u

def do_sort(fname):
    data = pd.read_table(fname,names=["cname","lon","lat"],encoding="shift-jis")
    cable_names=pd.unique(data["cname"])
    for j in range(len(cable_names)):
        cable = data[data["cname"]==cable_names[j]]
        e,n,u = latlonalt2enu(cable.loc[:,"lat"].values,cable.loc[:,"lon"].values,0,cable.loc[:,"lat"].mean(),cable.loc[:,"lon"].mean(),0)
        cable.loc[:,"e"] = np.array(e)
        cable.loc[:,"n"] = np.array(n)
        cable.loc[:,"u"] = np.array(u)
        sort_e = cable.sort_values(by="e")
        sort_e.loc[:,"i"]=np.arange(len(sort_e))
        sort_n = cable.sort_values(by="n")
        sort_n.loc[:,"i"]=np.arange(len(sort_n))
        cn = sort_e.iloc[0]
        cnum_stck1 = cn.name
        D1=0
        while(len(sort_e) > 1):
            i = sort_e.loc[cn.name]["i"]
            p = sort_e.query("i=="+str(i-1)+" or "+"i=="+str(i+1)).loc[:,['e','n']]
            i = sort_n.loc[cn.name]["i"]
            p = p.append(sort_n.query("i=="+str(i-1)+" or "+"i=="+str(i+1)).loc[:,['e','n']])
            p = p[~p.duplicated()]
            p.loc[:,'d'] = np.power((np.power((p.loc[:,'e'] - cn['e']),2) + np.power((p.loc[:,'n'] - cn['n']),2)),1/2)
            sort_e = sort_e.drop(cn.name)
            sort_n = sort_n.drop(cn.name)
            sort_e.loc[:,"i"]=np.arange(len(sort_e))
            sort_n.loc[:,"i"]=np.arange(len(sort_n))
            cn = p[p.loc[:,'d'] == np.min(p.loc[:,'d'])].iloc[0]
            cnum_stck1 = np.hstack((cnum_stck1,cn.name))
            D1 = D1 + cn.d
        sort_e = cable.sort_values(by="e")
        sort_e.loc[:,"i"]=np.arange(len(sort_e))
        sort_n = cable.sort_values(by="n")
        sort_n.loc[:,"i"]=np.arange(len(sort_n))
        cn = sort_n.iloc[0]
        cnum_stck2 = cn.name
        D2=0
        while(len(sort_n) > 1):
            i = sort_e.loc[cn.name]["i"]
            p = sort_e.query("i=="+str(i-1)+" or "+"i=="+str(i+1)).loc[:,['e','n']]
            i = sort_n.loc[cn.name]["i"]
            p = p.append(sort_n.query("i=="+str(i-1)+" or "+"i=="+str(i+1)).loc[:,['e','n']])
            p = p[~p.duplicated()]
            p.loc[:,'d'] = np.power((np.power((p.loc[:,'e'] - cn['e']),2) + np.power((p.loc[:,'n'] - cn['n']),2)),1/2)
            sort_e = sort_e.drop(cn.name)
            sort_n = sort_n.drop(cn.name)
            sort_e.loc[:,"i"]=np.arange(len(sort_e))
            sort_n.loc[:,"i"]=np.arange(len(sort_n))
            cn = p[p.loc[:,'d'] == np.min(p.loc[:,'d'])].iloc[0]
            cnum_stck2 = np.hstack((cnum_stck2,cn.name))
            D2 = D2 + cn.d  
    
        if (D2 <= D1):
            print(cable.loc[cnum_stck2,['cname','lat','lon']].to_csv(sep=',',header=None,index=None),end='')
        else:
            print(cable.loc[cnum_stck1,['cname','lat','lon']].to_csv(sep=',',header=None,index=None),end='')
    return     

argv = sys.argv
if(len(argv)==1):
    print("Usage: cablesort filename \ncablesort: error: the following arguments are required: filename",file=sys.stderr)

if(len(argv)>1):
    for k in range(1,len(argv)): 
        do_sort(argv[k])
exit(0)
