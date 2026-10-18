"""Recompute in a fresh copy; never overwrite the distributed result snapshot."""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parent
CODE=ROOT/'experiments/ssci_20261013/code'
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',action='store_true',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if any((CODE.parent/'exp').glob('*.csv')):raise SystemExit('Use a fresh copy and move exp outside it before recomputing; the released result snapshot is protected.')
 if not os.environ.get('PISA2022_RAW'):raise SystemExit('Set PISA2022_RAW to the directory containing the two OECD SAV files.')
 def call(name,*args):subprocess.run([sys.executable,str(CODE/name),*map(str,args)],cwd=ROOT,check=True)
 call('audit_inputs.py');call('verify_estimators.py');call('verify_prediction.py')
 for stage in ['primary','sensitivity','context','cubic']:call('run_analysis.py','--stage',stage,'--workers',a.workers)
 call('run_analysis.py','--stage','finish');call('prediction.py','--workers',a.workers);call('transfer.py','--workers',min(a.workers,2));call('transfer_decomposition.py');call('simulation_diagnostics.py');call('build_case.py');call('build_assets.py')
 subprocess.run([sys.executable,str(ROOT/'verify_aggregates.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
