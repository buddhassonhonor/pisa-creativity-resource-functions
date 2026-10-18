"""Nested school validation with training-only scale and spline fitting."""
from run_analysis import *
from sklearn.preprocessing import SplineTransformer
SEEDS=[20261013,20261014,20261015]
MODELS=['linear','linear_product','quadratic','quadratic_product','spline','spline_product','tensor']
def fold(s,seed,k):return int(hashlib.sha256((str(s)+'|'+str(seed)).encode()).hexdigest()[:16],16)%k
def arrays(t,pair,j):
 x,z=PAIRS[pair];cols=[t[f'PV{j}MATH'].to_numpy() if v=='math' else t[v].to_numpy() for v in [x,z]]
 if 'ESCS' not in (x,z):cols.append(t.ESCS.to_numpy())
 if 'math' not in (x,z):cols.append(t[f'PV{j}MATH'].to_numpy())
 cols.append(t[f'PV{j}READ'].to_numpy());return np.column_stack(cols)
def design(a,b,da,db,w,model,knots):
 mu=np.average(a,weights=w,axis=0);sd=np.sqrt(np.average((a-mu)**2,weights=w,axis=0));a=(a-mu)/np.maximum(sd,1e-12);b=(b-mu)/np.maximum(sd,1e-12)
 X=[np.ones((len(a),1)),a,da];Z=[np.ones((len(b),1)),b,db]
 if model.startswith('quadratic'):X.append(a[:,:2]**2);Z.append(b[:,:2]**2)
 if model.startswith('spline') or model=='tensor':
  X=[np.ones((len(a),1)),da];Z=[np.ones((len(b),1)),db];ta=[];tb=[]
  for c in range(a.shape[1]):
   # Weighted quantile knots determined on training predictors only.
   o=np.argsort(a[:,c]);q=np.interp(np.linspace(0,1,knots),(np.cumsum(w[o])-.5*w[o])/w.sum(),a[o,c]);q=np.unique(q)
   if len(q)<2:q=np.array([a[:,c].min()-1e-5,a[:,c].max()+1e-5])
   st=SplineTransformer(degree=3,knots=q[:,None],include_bias=False,extrapolation='linear');aa=st.fit_transform(a[:,c:c+1]);bb=st.transform(b[:,c:c+1]);X.append(aa);Z.append(bb);ta.append(aa);tb.append(bb)
  if model=='tensor':X.append(np.einsum('ni,nj->nij',ta[0],ta[1]).reshape(len(a),-1));Z.append(np.einsum('ni,nj->nij',tb[0],tb[1]).reshape(len(b),-1))
 if model.endswith('product'):X.append((a[:,0]*a[:,1])[:,None]);Z.append((b[:,0]*b[:,1])[:,None])
 return np.column_stack(X),np.column_stack(Z)
def evaluate(train,test,pair,model,knots=3,alpha=1,return_sums=False):
 wt=train.w.to_numpy();wt=wt/wt.mean();wv=test.w.to_numpy();da=train[['female','immigrant']].to_numpy();db=test[['female','immigrant']].to_numpy();sses=[];den=[]
 for j in range(1,11):
  a=arrays(train,pair,j);b=arrays(test,pair,j);X,Z=design(a,b,da,db,wt,model,knots);yy=train[f'PV{j}CRTH_NC'].to_numpy();y=test[f'PV{j}CRTH_NC'].to_numpy();mu=np.average(yy,weights=wt);sd=np.sqrt(np.average((yy-mu)**2,weights=wt));yy=(yy-mu)/sd;y=(y-mu)/sd
  penalty=np.eye(X.shape[1])*alpha;penalty[0,0]=0;gram=X.T@(wt[:,None]*X)+penalty;rhs=X.T@(wt*yy);coef=np.linalg.solve(gram,rhs) if alpha>0 else np.linalg.pinv(gram,rcond=1e-11)@rhs;pred=Z@coef;sses.append(np.sum(wv*(y-pred)**2));den.append(np.sum(wv*y*y))
 return (np.array(sses),np.array(den)) if return_sums else float(np.mean(np.array(sses)/np.array(den)))
def evaluate_grid(train,test,pair,model,knots,alphas):
 wt=train.w.to_numpy();wt/=wt.mean();wv=test.w.to_numpy();da=train[['female','immigrant']].to_numpy();db=test[['female','immigrant']].to_numpy();loss={a:[] for a in alphas}
 for j in range(1,11):
  X,Z=design(arrays(train,pair,j),arrays(test,pair,j),da,db,wt,model,knots);yy=train[f'PV{j}CRTH_NC'].to_numpy();y=test[f'PV{j}CRTH_NC'].to_numpy();mu=np.average(yy,weights=wt);sd=np.sqrt(np.average((yy-mu)**2,weights=wt));yy=(yy-mu)/sd;y=(y-mu)/sd;G=X.T@(wt[:,None]*X);rhs=X.T@(wt*yy);den=np.sum(wv*y*y)
  for alpha in alphas:
   pen=np.eye(X.shape[1])*alpha;pen[0,0]=0;coef=np.linalg.solve(G+pen,rhs) if alpha>0 else np.linalg.pinv(G,rcond=1e-11)@rhs;loss[alpha].append(np.sum(wv*(y-Z@coef)**2)/den)
 return {a:float(np.mean(v)) for a,v in loss.items()}
def choose(t,pair,model,seed,group='school'):
 inner=t[group].map(lambda s:fold(s,seed,2)).to_numpy();choices=[]
 for knots in ([3,5] if model in ['spline','spline_product','tensor'] else [3]):
  alphas=[.01,1,100] if model in ['spline','spline_product','tensor'] else [0];scores={a:[] for a in alphas}
  for k in range(2):
   a=t.loc[inner!=k];b=t.loc[inner==k]
   if len(a)<50 or len(b)<50:continue
   result=evaluate_grid(a,b,pair,model,knots,alphas)
   for alpha,v in result.items():scores[alpha].append(v)
  for alpha,score in scores.items():
   if score:choices.append((np.mean(score),knots,alpha))
 if not choices:raise ValueError('No feasible inner split')
 _,knots,alpha=min(choices);return knots,alpha
def job_prediction(site):
 t=pd.read_parquet(DATA/(site+'.parquet'));rows=[];t0=time.time()
 for pair in PAIRS:
  d=t.loc[mask(t,pair)].reset_index(drop=True)
  if len(d)<150 or d.school.nunique()<8:continue
  for seed in SEEDS:
   outer=d.school.map(lambda s:fold(s,seed,3)).to_numpy()
   for k in range(3):
    train=d.loc[outer!=k];test=d.loc[outer==k]
    for model in MODELS:
     knots,alpha=choose(train,pair,model,seed+100+k);sse,den=evaluate(train,test,pair,model,knots,alpha,True)
     rows.append(dict(site=site,pair=pair,seed=seed,fold=k,model=model,knots=knots,alpha=alpha,N_train=len(train),N_test=len(test),schools_test=test.school.nunique(),R2=1-np.mean(sse/den),SSE=sse.tolist(),DEN=den.tolist()))
 try:
  import psutil;rss=psutil.Process().memory_info().rss/1e6
 except ImportError:rss=None
 return rows,dict(site=site,seconds=time.time()-t0,RSS_MB=rss)
def run_prediction(workers=4,sites=None):
 rows=json.loads((OUT/'prediction_loss_sums.json').read_text(encoding='utf8')) if (OUT/'prediction_loss_sums.json').exists() and sites is None else [];runs=[];start=time.time();completed={r['site'] for r in rows}
 with ProcessPoolExecutor(max_workers=workers) as ex:
  fs={ex.submit(job_prediction,s):s for s in (sites or [p.stem for p in DATA.glob('*.parquet') if len(p.stem)==3]) if s not in completed}
  for f in as_completed(fs):
   rr,run=f.result();rows+=rr;runs.append(run);pd.DataFrame(rows).drop(columns=['SSE','DEN']).to_csv(OUT/'prediction_folds.csv',index=False);save(OUT/'prediction_loss_sums.json',rows);print('prediction',fs[f],len(rr),round(time.time()-start),flush=True)
 save(OUT/'prediction_runtime.json',dict(workers=workers,total_seconds=time.time()-start,site_runs=runs));finish_prediction()
def finish_prediction():
 raw=json.loads((OUT/'prediction_loss_sums.json').read_text(encoding='utf8'));groups={}
 for r in raw:
  k=(r['site'],r['pair'],r['seed'],r['model']);groups.setdefault(k,[]).append(r)
 rows=[]
 for (site,pair,seed,model),rr in groups.items():
  sse=np.sum([r['SSE'] for r in rr],axis=0);den=np.sum([r['DEN'] for r in rr],axis=0);rows.append(dict(site=site,pair=pair,seed=seed,model=model,R2=1-np.mean(sse/den),N=sum(r['N_test'] for r in rr)))
 a=pd.DataFrame(rows);a.to_csv(OUT/'prediction_site_seed.csv',index=False);a=a.groupby(['site','pair','model'],as_index=False).agg(R2=('R2','mean'),N=('N','first'));out=[];contrasts=[];rng=np.random.default_rng(20261013)
 for pair,t in a.groupby('pair'):
  p=t.pivot(index='site',columns='model',values='R2')
  for model in MODELS:
   vals0=p[model].to_numpy();boot=vals0[rng.integers(0,len(vals0),(2000,len(vals0)))].mean(axis=1);out.append(dict(pair=pair,model=model,K=len(vals0),R2=vals0.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975)))
  for model,base in [('linear_product','linear'),('quadratic_product','quadratic'),('spline_product','spline'),('tensor','spline'),('spline','linear')]:
   v=(p[model]-p[base]).to_numpy();boot=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1);contrasts.append(dict(pair=pair,comparison=model+'-'+base,K=len(v),delta=v.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975),low90=np.quantile(boot,.05),high90=np.quantile(boot,.95),below_margin_001=np.quantile(boot,.95)<.001,positive_systems=int((v>0).sum()),min=v.min(),max=v.max()))
 pd.DataFrame(out).to_csv(OUT/'prediction_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'prediction_contrasts.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--sites',nargs='*');p.add_argument('--finish',action='store_true');a=p.parse_args();finish_prediction() if a.finish else run_prediction(a.workers,a.sites)
