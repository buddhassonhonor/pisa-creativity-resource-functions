"""Independent arithmetic checks on public aggregate result tables."""
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/ssci_20261013/exp'
def main():
 a=pd.read_csv(OUT/'site_results.csv');p=pd.read_csv(OUT/'primary_multiplicity.csv')
 assert not a.duplicated(['pair','site','model','term']).any()
 assert np.isfinite(a[['beta','se','low','high','p']]).all().all()
 for _,r in p.iterrows():
  b=a[a.pair.eq(r.pair)&a.model.eq('quadratic')&a.term.eq('xz')]
  assert len(b)==r.K and b.N.sum()==r.N and abs(b.beta.mean()-r.beta)<1e-12
 assert np.max(abs(multipletests(p.p,method='holm')[1]-p.p_holm))<1e-12
 seed=pd.read_csv(OUT/'prediction_site_seed.csv');c=pd.read_csv(OUT/'prediction_contrasts.csv')
 v=seed.groupby(['site','pair','model']).R2.mean().reset_index()
 for _,r in c.iterrows():
  m,b=r.comparison.split('-');z=v[v.pair.eq(r.pair)].pivot(index='site',columns='model',values='R2')
  assert abs((z[m]-z[b]).mean()-r.delta)<1e-12
 d=pd.read_csv(OUT/'transfer_decomposition_summary.csv');t=pd.read_csv(OUT/'transfer_summary.csv')
 for _,r in d.iterrows():
  z=t[t.pair.eq(r.pair)&t.cap.eq(1000)&t.comparison.eq(r.comparison)].iloc[0]
  assert abs(r.offset_reduction+r.shape_reduction_on_raw_den-z.delta)<1e-8
 print('Verified: six primary averages, Holm adjustment, school prediction contrasts, and transfer-loss identities.')
if __name__=='__main__':main()
