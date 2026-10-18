"""Nested five-fold education-system transfer, with no target outcome in fitting."""
from prediction import *
def pool(pair,cap):
 rows=[];selected=[];x,z=PAIRS[pair]
 for p in DATA.glob('*.parquet'):
  if len(p.stem)!=3:continue
  t=pd.read_parquet(p);d=t.loc[mask(t,pair)].copy()
  if len(d)<150 or d.school.nunique()<8:continue
  cc=list(dict.fromkeys([x,z,'ESCS']+MATH+READ));cc=[c for c in cc if c!='math']
  # Target-system predictor distributions are used, not target outcomes.
  for c in cc:d[c]=wz(d[c],d.w)
  d['system_fold']=fold(p.stem,20261013,5);d['transfer_weight']=d.w/d.w.sum()
  order=d.CNTSTUID.map(lambda s:int(hashlib.sha256((p.stem+'|'+str(s)+'|20261013').encode()).hexdigest()[:16],16)).to_numpy();chosen=np.argsort(order)[:cap];d['train_selected']=False;d.iloc[chosen,d.columns.get_loc('train_selected')]=True
  for i in chosen:selected.append(dict(pair=pair,site=p.stem,CNTSTUID=d.iloc[i].CNTSTUID,school=d.iloc[i].school,fold=int(d.iloc[i].system_fold)))
  need=list(dict.fromkeys(cc+YCOL+['site','school','female','immigrant','w','transfer_weight','system_fold','train_selected','CNTSTUID']))
  rows.append(d[need])
 pd.DataFrame(selected).to_parquet(DATA/(f'transfer_selection_{pair}_{cap}.parquet'),index=False)
 return pd.concat(rows,ignore_index=True)
def job_transfer(pair,cap=1000,only_fold=None):
 d=pool(pair,cap);rows=[];choices=[];start=time.time()
 for outer in (range(5) if only_fold is None else [only_fold]):
  train=d[d.system_fold.ne(outer)&d.train_selected].copy();test=d[d.system_fold.eq(outer)].copy()
  train['w']=train.w/train.groupby('site').w.transform('sum');test['w']=test.transfer_weight
  for model in ['linear','quadratic_product','spline','spline_product','tensor']:
   knots,alpha=choose(train,pair,model,20261013+outer,group='site');choices.append(dict(pair=pair,fold=outer,cap=cap,model=model,knots=knots,alpha=alpha,training_systems=train.site.nunique(),N_train=len(train),held_systems=test.site.unique().tolist()))
   for site,t in test.groupby('site'):
    sse,den=evaluate(train,t,pair,model,knots,alpha,True);rows.append(dict(pair=pair,site=site,fold=outer,cap=cap,model=model,N=len(t),R2=1-np.mean(sse/den),knots=knots,alpha=alpha))
 return rows,choices,dict(pair=pair,cap=cap,seconds=time.time()-start)
def run_transfer(workers):
 rows=pd.read_csv(OUT/'transfer_results.csv').to_dict('records') if (OUT/'transfer_results.csv').exists() else [];choices=json.loads((OUT/'transfer_choices.json').read_text(encoding='utf8')) if (OUT/'transfer_choices.json').exists() else [];runs=json.loads((OUT/'transfer_runtime.json').read_text(encoding='utf8')) if (OUT/'transfer_runtime.json').exists() else [];completed={(r['pair'],r['cap']) for r in runs}
 with ProcessPoolExecutor(max_workers=workers) as ex:
  jobs=[(p,1000,None) for p in PAIRS]+[(p,500,0) for p in ['C11','C32','C34']]
  fs={ex.submit(job_transfer,*j):j for j in jobs if (j[0],j[1]) not in completed}
  for f in as_completed(fs):
   rr,cc,run=f.result();rows+=rr;choices+=cc;runs.append(run);pd.DataFrame(rows).to_csv(OUT/'transfer_results.csv',index=False);save(OUT/'transfer_choices.json',choices);save(OUT/'transfer_runtime.json',runs);print('transfer',fs[f],len(rr),run['seconds'],flush=True)
 a=pd.DataFrame(rows);out=[];rng=np.random.default_rng(20261013)
 for (pair,cap),t in a.groupby(['pair','cap']):
  p=t.pivot(index='site',columns='model',values='R2')
  for model,base in [('spline_product','spline'),('tensor','spline'),('spline','linear')]:
   v=(p[model]-p[base]).to_numpy();b=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1);out.append(dict(pair=pair,cap=cap,comparison=model+'-'+base,K=len(v),delta=v.mean(),low=np.quantile(b,.025),high=np.quantile(b,.975)))
 pd.DataFrame(out).to_csv(OUT/'transfer_summary.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();run_transfer(a.workers)
