#!/usr/bin/env python3
import io,sys,zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pandas as pd,numpy as np,requests
from grid_runner import sim,FEE,CAP

# H1 top-2 pure-grid candidates per horizon (v2), grouped by symbol.
CONFIGS={
 'ZECUSDT':[(1,.01,15.,3,1),(2,.01,15.,3,1),(3,.01,30.,15,1),(4,.25,10.,3,1)],
 'NEARUSDT':[(1,.01,1.25,3,1),(3,.90,7.,2,0)],
 'SUIUSDT':[(2,.90,5.,3,1)],
 'HBARUSDT':[(4,.98,4.,4,1)],
 'SHIBUSDT':[(5,.65,10.,2,1)],
 'SOLUSDT':[(5,.90,3.,2,0)],
}
UA={'User-Agent':'grid-validation/1.0'}

def mkurl(sym,y,m,d=None):
    if d is None:
        s=f'{y:04d}-{m:02d}';return f'https://data.binance.vision/data/spot/monthly/klines/{sym}/1m/{sym}-1m-{s}.zip'
    s=f'{y:04d}-{m:02d}-{d:02d}';return f'https://data.binance.vision/data/spot/daily/klines/{sym}/1m/{sym}-1m-{s}.zip'

def getzip(u):
    try:
        r=requests.get(u,timeout=90,headers=UA)
        if r.status_code==404:return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open([x for x in z.namelist() if not x.endswith('/')][0]) as f:
                x=pd.read_csv(f,header=None,usecols=[0,1,2,3,4],names=['ts','open','high','low','close'])
        n=pd.to_numeric(x.ts,errors='coerce');med=float(n.dropna().median())
        x.ts=pd.to_datetime(n,unit='us' if med>1e14 else 'ms',utc=True,errors='coerce')
        for c in ['open','high','low','close']:x[c]=pd.to_numeric(x[c],errors='coerce')
        return x.dropna()
    except Exception as e:
        print('GET_FAIL',u,repr(e),flush=True);return None

def months(a,b):
    y,m=a.year,a.month
    while (y,m)<=(b.year,b.month):
        yield y,m;m+=1
        if m==13:y+=1;m=1

def load(sym,years):
    now=datetime.now(timezone.utc);end=now.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(seconds=1);start=end-timedelta(days=366*years+4)
    first=datetime(end.year,end.month,1,tzinfo=timezone.utc);pm=first-timedelta(days=1);fs=[]
    for y,m in months(start,pm):
        x=getzip(mkurl(sym,y,m))
        if x is not None and len(x):fs.append(x)
    d=first;last=end.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=1)
    while d<=last:
        x=getzip(mkurl(sym,d.year,d.month,d.day))
        if x is not None and len(x):fs.append(x)
        d+=timedelta(days=1)
    if not fs:return pd.DataFrame()
    x=pd.concat(fs,ignore_index=True).sort_values('ts').drop_duplicates('ts')
    return x[(x.ts>=pd.Timestamp(start))&(x.ts<=pd.Timestamp(end))].reset_index(drop=True)

def main(sym):
    cfgs=CONFIGS[sym];maxy=max(x[0] for x in cfgs);x=load(sym,maxy)
    Path('validation').mkdir(exist_ok=True);rows=[]
    print('DATA1M',sym,len(x),x.ts.iloc[0] if len(x) else None,x.ts.iloc[-1] if len(x) else None,flush=True)
    if x.empty:pd.DataFrame().to_csv(f'validation/{sym}.csv',index=False);return
    end=x.ts.iloc[-1]
    for years,lm,um,n,kind in cfgs:
        target=end-pd.DateOffset(years=years);w=x[x.ts>=target].reset_index(drop=True)
        if w.empty or w.ts.iloc[0]>target+pd.Timedelta(minutes=5):
            print('NA',sym,years,flush=True);continue
        a=w[['open','high','low','close']].to_numpy(np.float64);p0=float(a[0,0]);p1=float(a[-1,3]);r=sim(a[:,0],a[:,1],a[:,2],a[:,3],p0*lm,p0*um,int(n),int(kind))
        row={'symbol':sym,'years':years,'start':str(w.ts.iloc[0]),'end':str(w.ts.iloc[-1]),'start_price':p0,'end_price':p1,'lower_mult':lm,'upper_mult':um,'grids':n,'grid_type':'geometric' if kind==0 else 'arithmetic','total_pnl_pct':(r[0]/CAP-1)*100,'final_equity':r[0],'grid_profit_pct':r[1]/CAP*100,'grid_profit_usd':r[1],'max_drawdown_pct':r[2],'fills':r[3],'fees_usd':r[4],'outside_low_pct':r[5],'outside_high_pct':r[6],'cycles':r[7],'buy_hold_pct':(p1/(p0*(1+FEE))-1)*100,'resolution':'1m'}
        rows.append(row);print('VALID',row,flush=True)
    pd.DataFrame(rows).to_csv(f'validation/{sym}.csv',index=False)
if __name__=='__main__':main(sys.argv[1])
