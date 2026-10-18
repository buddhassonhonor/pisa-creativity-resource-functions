from pathlib import Path
import json,hashlib,sys,platform,os,importlib.metadata as im
import numpy as np
import pandas as pd
import pyreadstat
ROOT=Path(__file__).resolve().parents[1]; PROJECT=ROOT.parents[1]
OUT=ROOT/'exp';OUT.mkdir(parents=True,exist_ok=True)
DATA=PROJECT/'data/derived/ssci_20261013';DATA.mkdir(parents=True,exist_ok=True)
S=Path(os.environ.get('PISA2022_RAW','D:/data/pisa_2022/raw'))/'CY08MSP_STU_QQQ.SAV'; C=S.parent/'CY08MSP_CRT_COG.SAV'
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for a in iter(lambda:f.read(8388608),b''):h.update(a)
 return h.hexdigest()
def run():
 _,source_meta=pyreadstat.read_sav(str(S),metadataonly=True);v=source_meta.column_names
 items=[n for n in v if n.startswith(('ST334Q','ST335Q','ST336Q','ST340Q','ST307Q'))]
 base=['CNT','CNTSCHID','CNTSTUID','ST004D01T','ESCS','IMMIG','W_FSTUWT','CREATOP','CREATEFF','CREATSCH','CREATFAM','PERSEVAGR']
 cols=base+items+[f'W_FSTURWT{i}' for i in range(1,81)]+[f'PV{i}{d}' for d in ['MATH','READ','SCIE'] for i in range(1,11)]
 d,m=pyreadstat.read_sav(str(S),usecols=cols);c,cm=pyreadstat.read_sav(str(C),usecols=base[:3]+[f'PV{i}CRTH_NC' for i in range(1,11)])
 assert not d.duplicated(base[:3]).any() and not c.duplicated(base[:3]).any()
 d=d.merge(c,on=base[:3],how='inner',validate='one_to_one').replace([np.inf,-np.inf],np.nan)
 d['site']=d.CNT.astype(str);d['school']=d.site+'_'+d.CNTSCHID.astype(str);d['female']=d.ST004D01T.map({1:1.,2:0.});d['immigrant']=d.IMMIG.map({1:0.,2:1.,3:1.});d['w']=d.W_FSTUWT
 keys={k:[n for n in items if n.startswith(p)] for k,p in [('open_raw','ST340Q'),('eff_raw','ST334Q'),('school_raw','ST335Q'),('family_raw','ST336Q'),('per_raw','ST307Q')]}
 # ST307 negative item keys inspected in metadata before scoring.
 neg=[f'ST307Q{i:02d}JA' for i in [4,6,7,10]]
 for key,cc in keys.items():
  hi=5 if key=='per_raw' else 4
  a=d[cc].where(d[cc].ge(1)&d[cc].le(hi))
  if key=='per_raw':
   for n in neg:a[n]=6-a[n]
  d[key]=a.mean(axis=1).where(a.notna().sum(axis=1)>=int(np.ceil(len(cc)/2)));d[key+'_count']=a.notna().sum(axis=1)
  for n in cc:d[n]=a[n]
 paths=[]
 for site,t in d.groupby('site'):
  p=DATA/(site+'.parquet');t.to_parquet(p,index=False);paths.append(dict(site=site,N=len(t),path=str(p.relative_to(PROJECT)),sha256=digest(p)))
 save(OUT/'provenance.json',dict(sources=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in [S,C]],source_questionnaire_N=m.number_rows,source_creative_N=cm.number_rows,matched_N=len(d),variables=m.column_names_to_labels,value_labels=m.variable_value_labels,raw_items=keys,perseverance_negative=neg,scoring_version=3,files=paths,scope='project-owned derived data; local analysis only; source public use does not imply redistribution permission'))
 save(PROJECT/'data/analysis_manifest.json',dict(experiment='ssci_20261013',files=paths,source_provenance='experiments/ssci_20261013/exp/provenance.json'))
 save(OUT/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),libraries={p:im.version(p) for p in ['numpy','pandas','scipy','statsmodels','pyreadstat','scikit-learn','pyarrow','matplotlib']},device='CPU',BLAS_threads=1,workers=4,environment_choice='Existing verified round2 Python3.11 environment; CPU survey regression; no GPU required'))
 print('Matched',len(d),'systems',d.site.nunique(),'negative keys',neg,flush=True)
if __name__=='__main__':run()
