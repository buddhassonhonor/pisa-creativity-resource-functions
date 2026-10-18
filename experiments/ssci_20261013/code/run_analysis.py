"""PISA creative-resource functional relations; scales re-estimated in all BRR replicates."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from audit_inputs import ROOT,PROJECT,OUT,DATA,save,digest
PAIRS={'C10':('CREATOP','math'),'C11':('CREATEFF','CREATSCH'),'C13':('CREATOP','PERSEVAGR'),'C31':('CREATEFF','ESCS'),'C32':('CREATSCH','CREATFAM'),'C34':('CREATOP','CREATSCH')}
RAW={'CREATOP':'open_raw','CREATEFF':'eff_raw','CREATSCH':'school_raw','CREATFAM':'family_raw','PERSEVAGR':'per_raw','ESCS':'ESCS','math':'math'}
REPS=[f'W_FSTURWT{i}' for i in range(1,81)];YCOL=[f'PV{i}CRTH_NC' for i in range(1,11)];MATH=[f'PV{i}MATH' for i in range(1,11)];READ=[f'PV{i}READ' for i in range(1,11)]
OLD=['AUS','CAN','CHL','DEU','FIN','KOR','SGP'];FLAGS=['AUS','CAN','DNK','HKG','IRL','JAM','LVA','NLD','NZL','PAN','GBR','USA']
def wz(a,w):
 a=np.asarray(a,float);mu=np.average(a,weights=w,axis=0);sd=np.sqrt(np.average((a-mu)**2,weights=w,axis=0));return (a-mu)/np.maximum(sd,1e-12)
def vals(d,n):return d[MATH].to_numpy() if n=='math' else np.repeat(d[n].to_numpy()[:,None],10,axis=1)
def gm(a,w,g,G):
 a=np.asarray(a);den=np.bincount(g,weights=w,minlength=G);u=np.zeros((G,a.shape[1]));np.add.at(u,g,w[:,None]*a);return (u/np.maximum(den[:,None],1e-20))[g]
def mask(t,pair,raw=False):
 x,z=PAIRS[pair];x,z=(RAW[x],RAW[z]) if raw else (x,z)
 cc=[c for c in [x,z,'ESCS','female','immigrant','w','school'] if c!='math']+YCOL+MATH+READ
 return t[cc].notna().all(axis=1)&t.w.gt(0)&t[REPS].notna().all(axis=1)
def fit(d,w,pair,model='quadratic',raw=False,return_design=False,agg=None,aw=None):
 w=np.asarray(w,float);w=w/w.mean();x,z=PAIRS[pair];x,z=(RAW[x],RAW[z]) if raw else (x,z)
 xx=wz(vals(d,x),w);zz=wz(vals(d,z),w);Y=wz(d[YCOL],w);ss=wz(d.ESCS,w);mm=wz(d[MATH],w);rr=wz(d[READ],w)
 if model=='PV_rotation':mm=np.roll(mm,1,axis=1);rr=np.roll(rr,2,axis=1)
 c=[np.ones((len(d),10)),xx,zz];names=['intercept','x','z']
 def add(name,a):names.append(name);c.append(a if np.ndim(a)==2 else np.repeat(np.asarray(a)[:,None],10,axis=1))
 if 'ESCS' not in (x,z):add('ses',ss)
 add('female',d.female);add('immigrant',d.immigrant)
 if model!='no_cognition':
  if 'math' not in (x,z):add('math',mm)
  add('reading',rr)
 if model!='linear':add('x2',xx*xx);add('z2',zz*zz)
 add('xz',xx*zz)
 if model=='cognitive_curvature':
  if 'math' not in (x,z):add('math2',mm*mm)
  add('reading2',rr*rr)
 if model=='cubic_science':
  add('science',wz(d[[f'PV{i}SCIE' for i in range(1,11)]],w));add('x3',xx**3);add('z3',zz**3);add('x2z',xx*xx*zz);add('xz2',xx*zz*zz)
 if model=='context':
  a,g,G,ind=agg;ww=np.asarray(aw if aw is not None else a.w,float)
  rawmeans=gm(a[[x,z,'ESCS']].to_numpy(),ww,g,G);first=np.unique(g,return_index=True)[1];rawmeans=rawmeans[first][ind]
  dd=d[[x,z,'ESCS']].to_numpy();mu=np.average(dd,weights=w,axis=0);sd=np.sqrt(np.average((dd-mu)**2,weights=w,axis=0));bx,bz,bs=((rawmeans-mu)/np.maximum(sd,1e-12)).T
  bx=np.repeat(bx[:,None],10,axis=1);bz=np.repeat(bz[:,None],10,axis=1);xw=xx-bx;zw=zz-bz
  c=[];names=[]
  for n,a0 in [('intercept',np.ones_like(xx)),('xw',xw),('zw',zw),('xb',bx),('zb',bz),('sesw',ss-bs),('sesb',bs),('female',d.female),('immigrant',d.immigrant),('math',mm),('reading',rr),('xw2',xw*xw),('zw2',zw*zw),('xb2',bx*bx),('zb2',bz*bz),('xwxb',xw*bx),('zwzb',zw*bz),('xwzw',xw*zw),('xwzb',xw*bz),('xbzw',bx*zw),('xbzb',bx*bz)]:add(n,a0)
 X=np.stack(c,axis=2).transpose(1,0,2);yy=Y.T
 if model=='school_FE':
  g,un=pd.factorize(d.school)
  for j in range(10):X[j]-=gm(X[j],w,g,len(un));yy[j]-=gm(yy[j,:,None],w,g,len(un)).ravel()
 gram=np.einsum('mnp,n,mnq->mpq',X,w,X,optimize=True);xy=np.einsum('mnp,n,mn->mp',X,w,yy,optimize=True)
 bread=np.linalg.pinv(gram,rcond=1e-11);b=np.einsum('mpq,mq->mp',bread,xy)
 return (b,names,X,yy,w,bread) if return_design else (b,names)
def estimate(t,pair,model,raw=False,rep=True,paired=None):
 good=mask(t,pair,raw);agg=None
 if model=='cubic_science':good&=t[[f'PV{i}SCIE' for i in range(1,11)]].notna().all(axis=1)
 if paired=='raw':good&=mask(t,pair,True)
 if model=='context':
  x,z=PAIRS[pair];ok=t[[x,z,'ESCS','w']].notna().all(axis=1)&t.w.gt(0);ct=ok.groupby(t.school).transform('sum');good&=ct.ge(10)
  a=t.loc[ok].reset_index(drop=True);g,un=pd.factorize(a.school);d=t.loc[good].reset_index(drop=True);ind=pd.Index(un).get_indexer(d.school);agg=(a,g,len(un),ind)
 else:d=t.loc[good].reset_index(drop=True)
 if len(d)<150 or d.school.nunique()<8:return []
 start=time.time();b,names,X,yy,w,bread=fit(d,d.w,pair,model,raw,True,agg)
 if paired:
  a,n=fit(d,d.w,pair,'linear' if paired=='curvature' else 'quadratic',paired=='raw');b=(b[:,names.index('xz')]-a[:,n.index('xz')])[:,None];names=['difference_xz']
 if rep:
  bb=[]
  for r in REPS:
   a,n=fit(d,d[r],pair,model,raw,agg=agg,aw=agg[0][r] if agg else None)
   if paired:
    o,on=fit(d,d[r],pair,'linear' if paired=='curvature' else 'quadratic',paired=='raw');a=(a[:,n.index('xz')]-o[:,on.index('xz')])[:,None]
   bb.append(a)
  dev=np.asarray(bb)-b;V=.05*np.einsum('rmp,rmq->mpq',dev,dev,optimize=True)
 else:
  g,un=pd.factorize(d.school);G=len(un);vs=[]
  for j in range(10):
   e=yy[j]-X[j]@b[j];u=np.zeros((G,len(names)));np.add.at(u,g,X[j]*(w*e)[:,None]);vs.append(bread[j]@(u.T@u)@bread[j]*G/(G-1)*(len(d)-1)/(len(d)-np.linalg.matrix_rank(X[j])-(G if model=='school_FE' else 0)))
  V=np.asarray(vs)
 mu=b.mean(axis=0);bt=np.atleast_2d(np.cov(b,rowvar=False,ddof=1));wi=V.mean(axis=0);cov=wi+1.1*bt
 terms=['x','z','x2','z2','xz','x2z','xz2','difference_xz','xwzw','xwzb','xbzw','xbzb'];rows=[]
 for term in terms:
  if term not in names:continue
  i=names.index(term);se=np.sqrt(max(0,cov[i,i]));df=min(79 if rep else 1e8,d.school.nunique()-1,9*(1+wi[i,i]/max(1.1*bt[i,i],1e-30))**2);q=stats.t.ppf(.975,df)*se
  rows.append(dict(pair=pair,site=d.site.iloc[0],model=('paired_'+paired) if paired else ('raw' if raw else model),term=term,N=len(d),schools=d.school.nunique(),beta=mu[i],se=se,low=mu[i]-q,high=mu[i]+q,p=2*stats.t.sf(abs(mu[i]/max(se,1e-20)),df),df=df,inference='BRR80_Fay.5_10PV_recomputed' if rep else 'school_cluster_10PV',duration_s=time.time()-start))
 return rows
def summarize(a):
 rows=[]
 for (pair,model,term),t in a.groupby(['pair','model','term']):
  b=t.beta.to_numpy();v=t.se.to_numpy()**2;K=len(t);mu=b.mean();se=np.sqrt(v.sum())/K;df=min(t.df.min(),v.sum()**2/np.sum(v*v/t.df));ci=stats.t.ppf(.975,df)*se;iv=1/np.maximum(v,1e-20);Q=np.sum(iv*(b-np.average(b,weights=iv))**2);C=iv.sum()-(iv**2).sum()/iv.sum();tau2=max(0,(Q-K+1)/C) if C>0 else 0
  rows.append(dict(pair=pair,model=model,term=term,K=K,N=int(t.N.sum()),beta=mu,se=se,low=mu-ci,high=mu+ci,p=2*stats.t.sf(abs(mu/max(se,1e-20)),df),df=df,negative=int((b<0).sum()),min=b.min(),max=b.max(),tau=np.sqrt(tau2),I2=max(0,(Q-K+1)/Q)*100 if Q>0 else 0,loco_min=(b.sum()-b.max())/(K-1) if K>1 else None,loco_max=(b.sum()-b.min())/(K-1) if K>1 else None,equivalent_90=abs(mu)+stats.t.ppf(.95,df)*se<.0125))
 return pd.DataFrame(rows)
def job(site,stage):
 t=pd.read_parquet(DATA/(site+'.parquet'));rows=[];flows=[];mem=[]
 for pair in PAIRS:
  good=mask(t,pair);flows.append(dict(site=site,pair=pair,eligible=len(t),N=int(good.sum()),schools=t.loc[good,'school'].nunique(),included=good.sum()>=150 and t.loc[good,'school'].nunique()>=8))
  if not flows[-1]['included']:continue
  if stage=='primary':
   for model in ['linear','quadratic','school_FE']:rows+=estimate(t,pair,model)
   rows+=estimate(t,pair,'quadratic',paired='curvature')
  elif stage=='sensitivity':
   for model in ['no_cognition','cognitive_curvature','PV_rotation']:rows+=estimate(t,pair,model)
   rows+=estimate(t,pair,'quadratic',raw=True);rows+=estimate(t,pair,'quadratic',paired='raw')
   for sex,label in [(1,'girls'),(0,'boys')]:
    rr=estimate(t[t.female.eq(sex)],pair,'quadratic',rep=False)
    for r in rr:r['model']=label
    rows+=rr
   a=t.loc[good];x,z=PAIRS[pair];xx=vals(a,x).mean(axis=1);zz=vals(a,z).mean(axis=1);cuts=np.quantile(zz,[1/3,2/3]);groups=np.searchsorted(cuts,zz);lo=max(np.quantile(xx[groups==g],.05) for g in range(3) if (groups==g).sum()>0);hi=min(np.quantile(xx[groups==g],.95) for g in range(3) if (groups==g).sum()>0)
   rr=estimate(a.loc[(xx>=lo)&(xx<=hi)],pair,'quadratic',rep=False)
   for r in rr:r['model']='central_overlap'
   rows+=rr
  elif stage=='context' and pair in ['C11','C34']:rows+=estimate(t,pair,'context')
  elif stage=='cubic':rows+=estimate(t,pair,'cubic_science')
 try:
  import psutil;memory=psutil.Process().memory_info().rss/1e6
 except ImportError:memory=None
 return rows,flows,dict(site=site,stage=stage,RSS_MB=memory)
def run(stage,workers=4,sites=None):
 site_names=[p.stem for p in DATA.glob('*.parquet') if len(p.stem)==3];site_names=sites or site_names;checkpoint=OUT/f'site_results_{stage}.csv';rows=pd.read_csv(checkpoint).to_dict('records') if checkpoint.exists() and sites is None else [];completed={r['site'] for r in rows};flow=[];memory=[];start=time.time()
 for site in completed:
  t=pd.read_parquet(DATA/(site+'.parquet'))
  for pair in PAIRS:
   good=mask(t,pair);flow.append(dict(site=site,pair=pair,eligible=len(t),N=int(good.sum()),schools=t.loc[good,'school'].nunique(),included=good.sum()>=150 and t.loc[good,'school'].nunique()>=8))
 with ProcessPoolExecutor(max_workers=workers) as ex:
  fs={ex.submit(job,s,stage):s for s in site_names if s not in completed}
  for f in as_completed(fs):
   rr,ff,mm=f.result();rows+=rr;flow+=ff;memory.append(mm);pd.DataFrame(rows).to_csv(OUT/f'site_results_{stage}.csv',index=False);print(stage,fs[f],'rows',len(rr),'elapsed',round(time.time()-start),flush=True)
 a=pd.DataFrame(rows);summarize(a).to_csv(OUT/f'summary_{stage}.csv',index=False);pd.DataFrame(flow).to_csv(OUT/f'sample_flow_{stage}.csv',index=False);save(OUT/f'run_{stage}.json',dict(seconds=time.time()-start,workers=workers,memory=memory,plan_sha256=digest(ROOT/'PLAN.md')))
def finish():
 a=pd.concat([pd.read_csv(OUT/f'site_results_{s}.csv') for s in ['primary','sensitivity','context','cubic']],ignore_index=True);a.to_csv(OUT/'site_results.csv',index=False);summarize(a).to_csv(OUT/'summary.csv',index=False)
 q=a[a.model.eq('quadratic')]
 for label,ss in [('historical7',q[q.site.isin(OLD)]),('additional',q[~q.site.isin(OLD)]),('unflagged',q[~q.site.isin(FLAGS)])]:summarize(ss).to_csv(OUT/(label+'_summary.csv'),index=False)
 s=summarize(q[q.term.eq('xz')]);s['p_holm']=multipletests(s.p,method='holm')[1];s.to_csv(OUT/'primary_multiplicity.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['primary','sensitivity','context','cubic','finish'],required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--sites',nargs='*');a=p.parse_args();finish() if a.stage=='finish' else run(a.stage,a.workers,a.sites)
