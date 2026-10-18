from run_analysis import *
import statsmodels.api as sm
def simulation():
 rows=[]
 for N in [500,2000,8000]:
  for truth in [0,.03,-.03]:
   for seed in range(20):
    rng=np.random.default_rng(20261013+seed);u=rng.multivariate_normal([0,0],[[1,.55],[.55,1]],N);x=np.exp(.6*u[:,0]);z=np.exp(.6*u[:,1]);x=(x-x.mean())/x.std();z=(z-z.mean())/z.std();y=.2*x+.1*z-.08*x*x-.08*z*z+truth*x*z+rng.normal(0,1,N)
    for model in ['linear','quadratic']:
     X=np.column_stack([np.ones(N),x,z]+([x*x,z*z] if model=='quadratic' else [])+[x*z]);fit=sm.OLS(y,X).fit(cov_type='HC3');b=fit.params[-1];se=fit.bse[-1];rows.append(dict(N=N,truth=truth,seed=seed,model=model,beta=b,se=se,p=fit.pvalues[-1],coverage=abs(b-truth)<=1.96*se,bias=b-truth))
 a=pd.DataFrame(rows);a.to_csv(OUT/'simulation_runs.csv',index=False);a.groupby(['N','truth','model']).agg(beta=('beta','mean'),bias=('bias','mean'),SD=('beta','std'),coverage=('coverage','mean'),reject=('p',lambda p:np.mean(p<.05))).reset_index().to_csv(OUT/'simulation_summary.csv',index=False)
def diagnostics():
 prov=json.loads((OUT/'provenance.json').read_text(encoding='utf8'));rows=[];items=[];cases=[]
 for p in DATA.glob('*.parquet'):
  if len(p.stem)!=3:continue
  t=pd.read_parquet(p);site=p.stem
  for k,cc in prov['raw_items'].items():
   a=t[cc];cov=a.cov();K=len(cc);alpha=K/(K-1)*(1-np.trace(cov)/cov.to_numpy().sum());count=t[k+'_count'];rows.append(dict(site=site,scale=k,N=len(t),any_items=int((count>0).sum()),half_items=int(t[k].notna().sum()),all_items=int((count==K).sum()),item_count=K,pairwise_alpha=alpha))
   for n in cc:items.append(dict(site=site,item=n,N=int(a[n].notna().sum()),mean=a[n].mean(),min=a[n].min(),max=a[n].max()))
  for pair,(x,z) in PAIRS.items():
   a=t.loc[mask(t,pair)];good=mask(t,pair)&mask(t,pair,True)
   if len(a)<150:continue
   ax=wz(vals(a,x).mean(axis=1),a.w);az=wz(vals(a,z).mean(axis=1),a.w)
   keep=good.sum();xr,zr=RAW[x],RAW[z];same=t.loc[good]
   # Correlations are descriptive weighted observed-score agreement, not invariance tests.
   if keep>30:
    for name,raw in [(x,xr),(z,zr)]:
     o=wz(vals(same,name).mean(axis=1),same.w);r=wz(vals(same,raw).mean(axis=1),same.w);rows.append(dict(site=site,pair=pair,scale=name+'_official_raw',N=keep,weighted_r=np.average(o*r,weights=same.w)))
   for cx,cz in [(-1,-1),(-1,1),(1,-1),(1,1)]:
    near=(abs(ax-cx)<=.35)&(abs(az-cz)<=.35);cases.append(dict(site=site,pair=pair,x=cx,z=cz,N_near=int(near.sum()),weighted_fraction=np.sum(a.w.to_numpy()[near])/a.w.sum(),within_observed_bounds=bool(ax.min()<=cx<=ax.max() and az.min()<=cz<=az.max())))
 pd.DataFrame(rows).to_csv(OUT/'measurement_diagnostics.csv',index=False);pd.DataFrame(items).to_csv(OUT/'item_diagnostics.csv',index=False);pd.DataFrame(cases).to_csv(OUT/'support_corners.csv',index=False)
if __name__=='__main__':simulation();diagnostics();print('Simulation and measurement/support diagnostics saved')
