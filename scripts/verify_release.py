"""Verify the code-only release manifest and reject tracked study artifacts."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
PUBLIC_ATLAS='src/toolkit/atlases/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
FORBIDDEN_DIRS=('results/','data/','outputs/','runs/','logs/','docs/','doc/','checkpoints/')
FORBIDDEN_SUFFIXES={'.csv','.tsv','.xlsx','.xls','.npy','.npz','.pkl','.pt','.pth','.pdf','.png','.svg','.doc','.docx','.ppt','.pptx','.log','.zip','.nii'}
def check_path(name):
    assert not name.startswith(FORBIDDEN_DIRS),f'Artifact is tracked: {name}'
    if name!=PUBLIC_ATLAS:
        assert Path(name).suffix.lower() not in FORBIDDEN_SUFFIXES,f'Data/output file is tracked: {name}'
def main():
    manifest=json.loads((ROOT/'reproducibility/release_manifest.json').read_text())
    for r in manifest['files']:
        check_path(r['path'])
        p=ROOT/r['path'];assert p.is_file(),str(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
    if (ROOT/'.git').exists():
        for f in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines():check_path(f)
    print(json.dumps({'scope':'code_only','manifest_files':len(manifest['files']),'numerical_results_tracked':False,'checks':'passed'},indent=2))
if __name__=='__main__':main()
