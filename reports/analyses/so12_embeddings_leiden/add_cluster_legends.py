"""Versioned presentation-only legend addition; canonical run evidence unchanged."""
from pathlib import Path
import json,hashlib,base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent
source=root/'index.html'
summary=json.loads((root/'analysis_summary.json').read_text())
methods=['pca','kernel_pca','spatial_pca'];names=['Pearson PCA','RBF kernel PCA','Spatial neighborhood PCA']
fig,axes=plt.subplots(1,3,figsize=(16,3.6));rows=[]
for method,name,axis in zip(methods,names,axes):
 n=summary['method_metrics'][method]['n_clusters']
 colors=plt.get_cmap('turbo')(np.linspace(.04,.96,max(n,2)))
 handles=[Patch(facecolor=colors[k],label=str(k)) for k in range(n)]
 axis.axis('off');axis.legend(handles=handles,loc='center',ncol=3,frameon=False,title=name,fontsize=11,title_fontsize=12)
 rows.extend({'method':method,'cluster_id':k,'color_hex':to_hex(colors[k])} for k in range(n))
fig.suptitle('Primary Leiden cluster color keys — resolution 1.0, seed 0',fontsize=14)
fig.tight_layout();fig.savefig(root/'cluster_color_keys.png',dpi=175,bbox_inches='tight');fig.savefig(root/'cluster_color_keys.pdf',bbox_inches='tight');plt.close(fig)
pd.DataFrame(rows).to_csv(root/'cluster_color_keys.csv',index=False)
encoded=base64.b64encode((root/'cluster_color_keys.png').read_bytes()).decode()
addition='<section><h2>Cluster color keys</h2><p>These keys apply to the embedding views and all spatial maps. Cluster IDs are local to each method; matching numbers or colors across methods do not imply matching cell groups.</p><img alt="Color keys mapping primary cluster IDs to colors for each encoder" style="width:100%;height:auto" src="data:image/png;base64,'+encoded+'"></section>'
original=source.read_text();assert '<h2>Representation views</h2>' in original
(root/'index.with_legends.html').write_text(original.replace('<h2>Representation views</h2>',addition+'<h2>Representation views</h2>',1))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=['index.with_legends.html','cluster_color_keys.png','cluster_color_keys.pdf','cluster_color_keys.csv','add_cluster_legends.py']
receipt={'revision':2,'kind':'presentation_only_posthoc_report','source_report_sha256':sha(source),
 'source_run_id':json.loads((root/'run_reference.json').read_text())['run_id'],
 'numerical_results_changed':False,'original_immutable_bundle_changed':False,
 'change':'Add explicit cluster color keys; original plots, labels, data and calculations unchanged.',
 'files':{s:{'sha256':sha(root/s),'size_bytes':(root/s).stat().st_size} for s in files}}
(root/'presentation_revision_v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Created version2 report with explicit cluster color keys')
