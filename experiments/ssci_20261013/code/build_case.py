"""Export a preselected FIN family-school illustration from saved coefficients."""
from run_analysis import *
a=pd.read_csv(OUT/'site_results_primary.csv');t=a[a.site.eq('FIN')&a.pair.eq('C32')&a.model.eq('quadratic')].set_index('term');d=pd.read_parquet(DATA/'FIN.parquet');d=d.loc[mask(d,'C32')];x=wz(d.CREATSCH,d.w);z=wz(d.CREATFAM,d.w);rows=[]
for cx in np.linspace(-1.5,1.5,31):
 for cz in np.linspace(-1.5,1.5,31):
  near=(abs(x-cx)<=.35)&(abs(z-cz)<=.35);change=t.loc['x','beta']*cx+t.loc['z','beta']*cz+t.loc['x2','beta']*cx*cx+t.loc['z2','beta']*cz*cz+t.loc['xz','beta']*cx*cz
  rows.append(dict(site='FIN',pair='C32',x=cx,z=cz,change=change,N_near=int(near.sum()),weight_fraction=np.sum(d.w.to_numpy()[near])/d.w.sum()))
pd.DataFrame(rows).to_csv(OUT/'case_surface.csv',index=False)
save(OUT/'case_metadata.json',dict(site='FIN',pair='C32',selection='FIN fixed illustration, not selected for product significance',N=len(d),model='quadratic',display='Conditional change from x=z=0, all other standardized controls fixed; points with fewer than20 local respondents masked; no interval displayed',grid=[-1.5,1.5],step=.1,density_halfwidth=.35))
