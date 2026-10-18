from prediction import *
t=pd.read_parquet(DATA/'ALB.parquet');d=t.loc[mask(t,'C11')];f=d.school.map(lambda s:fold(s,20261013,3));train=d[f.ne(0)];test=d[f.eq(0)]
g=evaluate_grid(train,test,'C11','tensor',3,[.01,1,100]);errs={a:abs(evaluate(train,test,'C11','tensor',3,a)-v) for a,v in g.items()};assert max(errs.values())<1e-10
# Compare positive-definite solve and former pseudoinverse on actual tensor Gram.
w=train.w.to_numpy();w/=w.mean();X,Z=design(arrays(train,'C11',1),arrays(test,'C11',1),train[['female','immigrant']],test[['female','immigrant']],w,'tensor',3);pen=np.eye(X.shape[1]);pen[0,0]=0;G=X.T@(w[:,None]*X)+pen;rhs=X.T@(w*train.PV1CRTH_NC.to_numpy());a=np.linalg.solve(G,rhs);b=np.linalg.pinv(G,rcond=1e-11)@rhs;err=np.max(abs(Z@a-Z@b));assert err<1e-8
save(OUT/'prediction_verification.json',dict(grid_error=errs,solve_vs_pseudoinverse_prediction_error=err,change='Reuse spline/Gram per candidate alpha; same inner objective; numerically equivalent positive-definite solve'))
print('Prediction optimization verified',errs,err)
