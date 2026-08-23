#!/usr/bin/env python3
import io,math,sys,zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np,pandas as pd,requests
from numba import njit
FEE=.0005; CAP=100.; YEARS=(1,2,3,4,5)
GRIDS=np.array([3,4,5,6,8,10,12,15,20,25,30,40,50,70,100,150,200,300,500,700,1000],np.int64)
LOWS=np.array([.10,.20,.25,.35,.45,.55,.65,.75,.85,.90,.95])
UPS=np.array([1.05,1.10,1.25,1.50,1.75,2.,2.5,3.,4.,5.,7.,10.])
UA={'User-Agent':'grid-research/1.0'}

def url(sym,intv,y,m,d=None):
    if d is None:
        s=f'{y:04d}-{m:02d}'; return f'https://data.binance.vision/data/spot/monthly/klines/{sym}/{intv}/{sym}-{intv}-{s}.zip'
    s=f'{y:04d}-{m:02d}-{d:02d}'; return f'https://data.binance.vision/data/spot/daily/klines/{sym}/{intv}/{sym}-{intv}-{s}.zip'

def getzip(u):
    try:
        r=requests.get(u,timeout=45,headers=UA)
        if r.status_code==404:return None
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open([x for x in z.namelist() if not x.endswith('/')][0]) as f:
                x=pd.read_csv(f,header=None,usecols=[0,1,2,3,4],names=['ts','open','high','low','close'])
        n=pd.to_numeric(x.ts,errors='coerce'); med=float(n.dropna().median())
        x.ts=pd.to_datetime(n,unit='us' if med>1e14 else 'ms',utc=True,errors='coerce')
        for c in ['open','high','low','close']:x[c]=pd.to_numeric(x[c],errors='coerce')
        return x.dropna()
    except Exception as e:
        print('GET_FAIL',u,repr(e),flush=True); return None

def months(a,b):
    y,m=a.year,a.month
    while (y,m)<=(b.year,b.month):
        yield y,m; m+=1
        if m==13:y+=1;m=1

def load(sym):
    now=datetime.now(timezone.utc); end=now.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(seconds=1); start=end-timedelta(days=366*5+10)
    first=datetime(end.year,end.month,1,tzinfo=timezone.utc); pm=first-timedelta(days=1); fs=[]
    for y,m in months(start,pm):
        x=getzip(url(sym,'1h',y,m))
        if x is not None and len(x):fs.append(x)
    d=first; last=end.replace(hour=0,minute=0,second=0,microsecond=0)-timedelta(days=1)
    while d<=last:
        x=getzip(url(sym,'1h',d.year,d.month,d.day))
        if x is not None and len(x):fs.append(x)
        d+=timedelta(days=1)
    if not fs:return pd.DataFrame()
    x=pd.concat(fs,ignore_index=True).sort_values('ts').drop_duplicates('ts')
    return x[(x.ts>=pd.Timestamp(start))&(x.ts<=pd.Timestamp(end))].reset_index(drop=True)

@njit(cache=True)
def sim(o,h,l,c,lo,up,n,kind):
    if len(o)<2 or n<3 or lo<=0 or up<=lo:return (np.nan,np.nan,np.nan,0,np.nan,0,0,0)
    lv=np.empty(n,np.float64)
    if kind==0:
        s=math.log(up/lo)/(n-1)
        for i in range(n):lv[i]=lo*math.exp(s*i)
    else:
        s=(up-lo)/(n-1)
        for i in range(n):lv[i]=lo+s*i
    p0=o[0]
    if p0<=lo or p0>=up:return (np.nan,np.nan,np.nan,0,np.nan,0,0,0)
    bi=np.zeros(n,np.int16);si=np.zeros(n,np.int16);bg=np.zeros(n,np.int16);sg=np.zeros(n,np.int16)
    ns=0;bu=0.
    for i in range(n):
        if lv[i]<p0:bi[i]=1;bu+=lv[i]
        elif lv[i]>p0:si[i]=1;ns+=1
    den=(bu+ns*p0)*(1+FEE)
    if den<=0:return (np.nan,np.nan,np.nan,0,np.nan,0,0,0)
    q=CAP/den;base=q*ns;cash=CAP-base*p0*(1+FEE);fees=base*p0*FEE;gp=0.;cycles=0;fills=0;peak=CAP;ddmax=0.;ol=0;oh=0
    for j in range(len(o)):
        a=o[j]
        if c[j]>=o[j]:p1=l[j];p2=h[j];p3=c[j]
        else:p1=h[j];p2=l[j];p3=c[j]
        for z in range(3):
            b=p1 if z==0 else (p2 if z==1 else p3)
            if b>a:
                i0=np.searchsorted(lv,a,side='right');i1=np.searchsorted(lv,b,side='right')-1
                if i0<0:i0=0
                if i1>=n:i1=n-1
                for i in range(i0,i1+1):
                    c0=si[i];cg=sg[i];cnt=c0+cg
                    if cnt:
                        v=q*lv[i]*cnt;cash+=v*(1-FEE);base-=q*cnt;fees+=v*FEE;fills+=cnt
                        if cg and i>0:gp+=q*(lv[i]*(1-FEE)-lv[i-1]*(1+FEE))*cg;cycles+=cg
                        si[i]=0;sg[i]=0
                        if i>0:bg[i-1]+=cnt
            elif b<a:
                i1=np.searchsorted(lv,a,side='left')-1;i0=np.searchsorted(lv,b,side='left')
                if i1>=n:i1=n-1
                if i0<0:i0=0
                for i in range(i1,i0-1,-1):
                    c0=bi[i];cg=bg[i];cnt=c0+cg
                    if cnt:
                        v=q*lv[i]*cnt;cash-=v*(1+FEE);base+=q*cnt;fees+=v*FEE;fills+=cnt
                        if cg and i<n-1:gp+=q*(lv[i+1]*(1-FEE)-lv[i]*(1+FEE))*cg;cycles+=cg
                        bi[i]=0;bg[i]=0
                        if i<n-1:sg[i+1]+=cnt
            a=b
        eq=cash+base*c[j]
        if c[j]<lo:ol+=1
        elif c[j]>up:oh+=1
        if eq>peak:peak=eq
        if peak>0:
            d=(peak-eq)/peak
            if d>ddmax:ddmax=d
    return (cash+base*c[-1],gp,ddmax*100,fills,fees,ol/len(o)*100,oh/len(o)*100,cycles)

def main(sym):
    x=load(sym);print('DATA',sym,len(x),x.ts.iloc[0] if len(x) else None,x.ts.iloc[-1] if len(x) else None,flush=True)
    Path('out').mkdir(exist_ok=True);rows=[]
    if x.empty:pd.DataFrame().to_csv(f'out/{sym}.csv',index=False);return
    end=x.ts.iloc[-1]
    for y in YEARS:
        target=end-pd.DateOffset(years=y);w=x[x.ts>=target].reset_index(drop=True)
        if w.empty or w.ts.iloc[0]>target+pd.Timedelta(days=3):
            print('NA',sym,y,flush=True);continue
        a=w[['open','high','low','close']].to_numpy(np.float64);p0=float(a[0,0]);p1=float(a[-1,3]);bt=None;bg=None;cases=0
        for lm in LOWS:
          for um in UPS:
            lo=p0*lm;up=p0*um
            for n in GRIDS:
              for kind in (0,1):
                r=sim(a[:,0],a[:,1],a[:,2],a[:,3],lo,up,int(n),kind);cases+=1
                if not np.isfinite(r[0]):continue
                d={'symbol':sym,'years':y,'start':str(w.ts.iloc[0]),'end':str(w.ts.iloc[-1]),'start_price':p0,'end_price':p1,'lower_mult':lm,'upper_mult':um,'lower':lo,'upper':up,'grids':int(n),'grid_type':'geometric' if kind==0 else 'arithmetic','total_pnl_pct':(r[0]/CAP-1)*100,'final_equity':r[0],'grid_profit_pct':r[1]/CAP*100,'grid_profit_usd':r[1],'max_drawdown_pct':r[2],'fills':r[3],'fees_usd':r[4],'outside_low_pct':r[5],'outside_high_pct':r[6],'cycles':r[7],'buy_hold_pct':(p1/(p0*(1+FEE))-1)*100,'cases':cases,'resolution':'1h'}
                if bt is None or d['total_pnl_pct']>bt['total_pnl_pct']:bt=d.copy()
                if bg is None or d['grid_profit_pct']>bg['grid_profit_pct']:bg=d.copy()
        bt['metric']='best_total';bg['metric']='best_grid';rows.extend([bt,bg]);print('BEST',sym,y,'TOTAL',round(bt['total_pnl_pct'],2),bt['grid_type'],bt['grids'],bt['lower_mult'],bt['upper_mult'],'GRID',round(bg['grid_profit_pct'],2),bg['grid_type'],bg['grids'],bg['lower_mult'],bg['upper_mult'],flush=True)
    pd.DataFrame(rows).to_csv(f'out/{sym}.csv',index=False)
if __name__=='__main__':main(sys.argv[1])
