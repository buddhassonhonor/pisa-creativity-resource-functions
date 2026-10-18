from run_analysis import *
import statsmodels.api as sm
t=pd.read_parquet(DATA/'ALB.parquet');d=t.loc[mask(t,'C11')].reset_index(drop=True);b,n,X,y,w,br=fit(d,d.w,'C11',return_design=True)
ref=sm.WLS(y[0],X[0],weights=w).fit();err=float(np.max(np.abs(ref.params-b[0])));assert err<1e-9
reps=np.asarray([fit(d,d[r],'C11')[0] for r in REPS]);dev=reps-b;V=.05*np.einsum('rmp,rmq->mpq',dev,dev);manual=sum(np.outer(a[0],a[0]) for a in dev)/20;errV=float(np.max(abs(V[0]-manual)));assert errV<1e-12
bf,nf,Xf,yf,wf,_=fit(d,d.w,'C11','school_FE',return_design=True);g,u=pd.factorize(d.school);centering=max(np.max(abs(gm(Xf[0],wf,g,len(u)))),np.max(abs(gm(yf[0,:,None],wf,g,len(u)))));assert centering<1e-8
r=estimate(t,'C11','quadratic');assert len(r)==5 and all(np.isfinite(q['se']) for q in r)
save(OUT/'estimator_verification.json',dict(WLS_max_error=err,BRR_max_error=errV,FE_weighted_group_mean_max=centering,site='ALB',N=len(d),actual_quadratic_results=r))
print('Estimator verification passed',err,errV,centering)
