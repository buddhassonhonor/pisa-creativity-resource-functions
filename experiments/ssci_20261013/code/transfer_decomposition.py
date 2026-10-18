"""Post-result transfer loss decomposition with frozen selected hyperparameters."""
from transfer import *
def run_pair(pair):
 d=pool(pair,1000);choices=json.loads((OUT/'transfer_choices.json').read_text(encoding='utf8'));choices=[r for r in choices if r['pair']==pair and r['cap']==1000];rows=[]
 for r in choices:
  outer=r['fold'];train=d[d.system_fold.ne(outer)&d.train_selected].copy();test=d[d.system_fold.eq(outer)].copy();train['w']=train.w/train.groupby('site').w.transform('sum');test['w']=test.transfer_weight;wt=train.w.to_numpy();wt/=wt.mean();wv=test.w.to_numpy();da=train[['female','immigrant']].to_numpy();db=test[['female','immigrant']].to_numpy();parts={site:np.flatnonzero(test.site.to_numpy()==site) for site in test.site.unique()};sums={site:[] for site in parts}
  for j in range(1,11):
   X,Z=design(arrays(train,pair,j),arrays(test,pair,j),da,db,wt,r['model'],r['knots']);yy=train[f'PV{j}CRTH_NC'].to_numpy();y=test[f'PV{j}CRTH_NC'].to_numpy();mu=np.average(yy,weights=wt);sd=np.sqrt(np.average((yy-mu)**2,weights=wt));yy=(yy-mu)/sd;y=(y-mu)/sd;pen=np.eye(X.shape[1])*r['alpha'];pen[0,0]=0;G=X.T@(wt[:,None]*X)+pen;rhs=X.T@(wt*yy);coef=np.linalg.solve(G,rhs) if r['alpha']>0 else np.linalg.pinv(G,rcond=1e-11)@rhs;pred=Z@coef
   for site,ix in parts.items():
    w=wv[ix];y0=y[ix];p0=pred[ix];ym=np.average(y0,weights=w);pm=np.average(p0,weights=w);total=np.sum(w*(y0-p0)**2);shape=np.sum(w*((y0-ym)-(p0-pm))**2);offset=w.sum()*(ym-pm)**2;den=np.sum(w*y0*y0);within=np.sum(w*(y0-ym)**2);assert abs(total-shape-offset)<1e-8
    sums[site].append([total,shape,offset,den,within])
  for site,values in sums.items():
   v=np.asarray(values);rows.append(dict(pair=pair,site=site,fold=outer,model=r['model'],raw_R2=1-np.mean(v[:,0]/v[:,3]),shape_R2=1-np.mean(v[:,1]/v[:,4]),shape_loss_fraction=np.mean(v[:,1]/v[:,3]),offset_loss_fraction=np.mean(v[:,2]/v[:,3]),N=len(parts[site])))
 return rows
def main():
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(run_pair,p):p for p in PAIRS}
  for f in as_completed(fs):rows+=f.result();pd.DataFrame(rows).to_csv(OUT/'transfer_decomposition.csv',index=False);print('decomposed',fs[f],flush=True)
 a=pd.DataFrame(rows);original=pd.read_csv(OUT/'transfer_results.csv');match=a.merge(original[original.cap.eq(1000)],on=['pair','site','model'],suffixes=('_new','_old'));err=np.max(abs(match.raw_R2-match.R2));assert err<1e-8;out=[];rng=np.random.default_rng(20261013)
 for pair,t in a.groupby('pair'):
  for model in ['spline_product','tensor']:
   p=t.pivot(index='site',columns='model',values='shape_R2');v=(p[model]-p.spline).to_numpy();b=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)
   off=t.pivot(index='site',columns='model',values='offset_loss_fraction');shape=t.pivot(index='site',columns='model',values='shape_loss_fraction');out.append(dict(pair=pair,comparison=model+'-spline',K=len(v),shape_delta=v.mean(),shape_low=np.quantile(b,.025),shape_high=np.quantile(b,.975),offset_reduction=(off.spline-off[model]).mean(),shape_reduction_on_raw_den=(shape.spline-shape[model]).mean()))
 pd.DataFrame(out).to_csv(OUT/'transfer_decomposition_summary.csv',index=False);save(OUT/'transfer_decomposition_verification.json',dict(original_R2_max_error=err,identity='Total weighted squared loss = centered shape loss + weighted system-mean offset loss; matched fixed choices; target means only used for evaluation decomposition, never model fitting/tuning',status='post-result exploratory diagnostic'))
if __name__=='__main__':main()
