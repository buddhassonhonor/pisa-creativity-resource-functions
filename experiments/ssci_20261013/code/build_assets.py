"""Plot saved aggregate results; never re-estimate survey models for styling."""
from audit_inputs import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=PROJECT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':18,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16})
LABEL={'C10':'Openness × mathematics','C11':'Efficacy × school','C13':'Openness × perseverance','C31':'Efficacy × socioeconomic status','C32':'School × peers/family','C34':'Openness × school'}
def export(name):plt.tight_layout();plt.savefig(FIG/(name+'.png'),dpi=100);plt.savefig(FIG/(name+'.pdf'));plt.close()
def main():
 s=pd.read_csv(OUT/'summary.csv');s=s[s.term.eq('xz')];fig,ax=plt.subplots(figsize=(10,6));order=list(LABEL);ys=np.arange(6)
 for delta,model,color in [(-.18,'linear','#9b4154'),(0,'quadratic','#226a9d'),(.18,'school_FE','#3a865d')]:
  t=s[s.model.eq(model)].set_index('pair').reindex(order);ax.errorbar(t.beta,ys+delta,xerr=[t.beta-t.low,t.high-t.beta],fmt='o',label=model.replace('_',' '),color=color,capsize=3)
 ax.axvline(0,color='black',lw=1);ax.axvspan(-.0125,.0125,color='gray',alpha=.12);ax.set_yticks(ys,[LABEL[p] for p in order]);ax.set_xlabel('Standardized product coefficient');ax.invert_yaxis();ax.legend(loc='lower right');export('products')
 p=pd.read_csv(OUT/'prediction_contrasts.csv');fig,ax=plt.subplots(figsize=(10,6))
 for delta,comparison,color,label in [(-.14,'spline_product-spline','#226a9d','Bilinear addition'),(.14,'tensor-spline','#9b4154','Tensor addition')]:
  t=p[p.comparison.eq(comparison)].set_index('pair').reindex(order);ax.errorbar(t.delta,ys+delta,xerr=[t.delta-t.low,t.high-t.delta],fmt='o',color=color,label=label,capsize=3)
 ax.axvline(0,color='black',lw=1);ax.axvline(.001,color='gray',ls='--');ax.set_yticks(ys,[LABEL[p] for p in order]);ax.set_xlabel('Added prediction statistic');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
 from matplotlib.ticker import MaxNLocator
 ax.xaxis.set_major_locator(MaxNLocator(5));ax.invert_yaxis();ax.legend();export('prediction')
 a=pd.read_csv(OUT/'site_results.csv');fig,ax=plt.subplots(figsize=(10,6));values=[a[a.pair.eq(p)&a.model.eq('quadratic')&a.term.eq('xz')].beta.to_numpy() for p in order];ax.boxplot(values,orientation='horizontal',tick_labels=[LABEL[p] for p in order],showfliers=True);ax.axvline(0,color='black',lw=1);ax.set_xlabel('Quadratic product coefficient');ax.invert_yaxis();export('heterogeneity')
 a=pd.read_csv(OUT/'simulation_summary.csv');a=a[a.truth.eq(0)];fig,ax=plt.subplots(figsize=(10,6))
 for model,color in [('linear','#9b4154'),('quadratic','#226a9d')]:
  t=a[a.model.eq(model)];ax.plot(t.N,t.reject,'o-',label=model,color=color)
 ax.axhline(.05,color='gray',ls='--');ax.set_xscale('log');ax.set_ylim(0,1.03);ax.set_xlabel('Simulated sample size');ax.set_ylabel('Detection rate when true product = 0');ax.legend();export('simulation')
 case=pd.read_csv(OUT/'case_surface.csv');fig,ax=plt.subplots(figsize=(10,6));g=case.pivot(index='z',columns='x',values='change');n=case.pivot(index='z',columns='x',values='N_near');masked=np.ma.masked_where(n.to_numpy()<20,g.to_numpy());im=ax.pcolormesh(g.columns,g.index,masked,cmap='RdBu_r',shading='auto');cb=fig.colorbar(im,ax=ax);cb.set_label('Change from average (SD)');ax.set_xlabel('School environment (SD)');ax.set_ylabel('Peers/family environment (SD)');export('case_surface')
if __name__=='__main__':main()
