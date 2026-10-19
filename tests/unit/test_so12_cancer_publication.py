"""Publication guards and result-schema compatibility without publishing real data."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

DIRECTORY=Path(__file__).resolve().parents[2]/'scripts/analysis'
sys.path.insert(0,str(DIRECTORY))
publication=importlib.import_module('publish_so12_cancer_comparison')
promotion=importlib.import_module('promote_so12_cancer_comparison_result')


def test_core_mixing_checks_inputs_outputs_and_external_graph(tmp_path):
    for name in ['cells.csv.gz','clustering_summary.json','core_mixing.csv','edges.npy']:
        (tmp_path/name).write_bytes(name.encode())
    summary={'passed':True,'status':'complete','n_cells':81607,
             'methods':['h0','h1','h2','h3','h4','pca','scvi'],'resolution':1.,'seed':0,
             'input_sha256':{name:publication.sha256_file(tmp_path/name) for name in ['cells.csv.gz','clustering_summary.json']},
             'output_sha256':{'core_mixing.csv':publication.sha256_file(tmp_path/'core_mixing.csv')},
             'graph_sources':{'h0':{'path':str(tmp_path/'edges.npy'),'relative':False,**publication.record(tmp_path/'edges.npy')}}}
    publication.verify_core_mixing(tmp_path,summary)
    (tmp_path/'core_mixing.csv').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Core-mixing bytes changed'):
        publication.verify_core_mixing(tmp_path,summary)
    summary['output_sha256']['core_mixing.csv']=publication.sha256_file(tmp_path/'core_mixing.csv')
    (tmp_path/'edges.npy').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Core-mixing graph source changed'):
        publication.verify_core_mixing(tmp_path,summary)


def test_error_maps_and_registration_helper_are_publication_requirements(tmp_path):
    assert 'oracle_union_errors' in publication.REQUIRED_FIGURES
    paths=SimpleNamespace(project_root=tmp_path)
    assert tmp_path/'scripts/analysis/register_so12_scvi_run.py' in publication.publication_sources(paths)


def test_identity_and_parent_checkpoint_cannot_be_substituted(tmp_path):
    with pytest.raises(RuntimeError,match='identity differs'):
        publication.validate_publication_identity({'evaluation_id':'unrelated'})
    publication.write(tmp_path/'scvi_run_registration.json',{
        'status':'complete','model_family':'scvi','source_bagm_run_id':publication.RUN,
        'comparison_evaluation_id':publication.EVALUATION,'sealed_test_claim':False,
        'derived_run_id':publication.RUN})
    with pytest.raises(RuntimeError,match='cannot belong to the BAGM'):
        publication.verify_scvi_registration(tmp_path,SimpleNamespace())


def test_promoted_result_schema_and_keys_match_report_contract(tmp_path,monkeypatch):
    from spatial_benchmark.paths import ProjectPaths
    from spatial_benchmark.result_catalog import validate_result_manifest
    paths=ProjectPaths.from_environment({'BAGM_ROOT':str(tmp_path)})
    root=paths.report_root/'analyses/so12_cancer_comparison'/publication.EVALUATION
    (root/'metrics').mkdir(parents=True)
    scvi_id='r_20261018T052052Z_04078a70_s000_f00_a01_14d78fb9'
    summary={'evaluation_id':publication.EVALUATION,'source_run_id':publication.RUN,'n_cells':81607,'test_cores':[15,23],
             'primary_scores':[{'Method':method,'Oracle core 15':.5,'Oracle core 23':.6,
                                'Transfer core 15':.2,'Transfer core 23':.3} for method in ['h0','h1','h2','h3','h4','PCA','scVI']],
             'pca_training_explained_variance':.42,'coverage':[{'all_positive_footprint_dice':.3},{'all_positive_footprint_dice':.4}],
             'maximum_defensible_claim':'Exploratory approximate-section agreement in two cores.'}
    manifest={'evaluation_id':publication.EVALUATION,'campaign_id':publication.CAMPAIGN,'source_run_id':publication.RUN,
              'status':'complete','passed':True,'summary':summary}
    publication.write(root/'manifest.json',manifest)
    publication.write(root/'scvi_run_registration.json',{'derived_run_id':scvi_id})
    publication.write(root/'metrics/pixel_agreement.json',{'seed_stability':[{'method':'h0','resolution':1.,'all_cells_ari':.5}]})
    target=paths.result_root/'descriptive_analysis/representation_comparison/res_so12_test_cancer_representation_agreement'
    target.mkdir(parents=True)
    original=DIRECTORY.parents[1]/'results/descriptive_analysis/representation_comparison/res_so12_test_cancer_representation_agreement/result.yaml'
    (target/'result.yaml').write_text(original.read_text())
    monkeypatch.setattr(promotion,'current_paths',lambda:paths)
    monkeypatch.setattr(promotion,'verify_final',lambda p:manifest)
    monkeypatch.setattr(promotion,'verify_scvi_registration',lambda *args:{'derived_run_id':scvi_id})
    class Connection:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,*args):return self
        def fetchone(self):return {'run_id':publication.RUN,'status':'completed','artifact_path':str(root)}
    monkeypatch.setattr(promotion,'Registry',lambda *args:SimpleNamespace(connect=lambda:Connection()))
    commands=[]
    monkeypatch.setattr(promotion.subprocess,'run',lambda command,**kwargs:commands.append(command))
    promotion.promote()
    result=validate_result_manifest(target/'result.yaml',paths=paths,verify_payloads=True,verify_sources=True)
    payload=yaml.safe_load((target/'result.yaml').read_text())
    assert payload['status']=='verified' and payload['outcome']=='inconclusive'
    assert payload['provenance']['run_ids']==[publication.RUN,scvi_id]
    assert len(commands)==2
