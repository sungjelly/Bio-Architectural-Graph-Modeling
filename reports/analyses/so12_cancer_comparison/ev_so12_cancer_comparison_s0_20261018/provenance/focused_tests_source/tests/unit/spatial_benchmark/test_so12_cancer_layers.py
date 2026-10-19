from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/analysis'))
from test_so12_paper_test_h0_extraction import small_model_core
import so12_cancer_layers as layers


def references(model,core,norm):
    arguments = dict(input_expression=layers.base.visible_count_pearson_input(torch.tensor(core.raw_counts),
        torch.zeros_like(torch.tensor(core.raw_counts),dtype=torch.bool),**norm),
        gene_mask=torch.zeros_like(torch.tensor(core.raw_counts),dtype=torch.bool),
        node_covariates=torch.tensor(core.node_covariates),edge_index=torch.tensor(core.edge_index),
        relative_geometry=torch.tensor(core.relative_geometry))
    with torch.inference_mode():
        result=model(**arguments,return_intermediate_embeddings=True,return_graph_step_embeddings=True)
    return result


def test_layers_match_explicit_api_and_preserve_predictions():
    torch.set_num_threads(1)
    model,core,norm,_=small_model_core()
    ref=references(model,core,norm)
    arrays,audit=layers.extract_layers(model,core,norm,torch.device('cpu'),
        ref.node_encoder_embedding.numpy(),ref.final_graph_embedding.numpy())
    for i in range(3):
        assert np.array_equal(arrays[f'h{i+1}'],ref.graph_step_embeddings[i].numpy())
    assert audit['predictions_exact'] and audit['published_h0_exact'] and audit['published_h4_exact']
    assert '_chunked_layer' not in model.__dict__ and not model.encoder._forward_hooks


@pytest.mark.parametrize('which',[0,4])
def test_published_reference_change_rejected(which):
    model,core,norm,_=small_model_core()
    ref=references(model,core,norm)
    a,b=ref.node_encoder_embedding.numpy().copy(),ref.final_graph_embedding.numpy().copy()
    changed=a if which==0 else b
    changed[0,0]=np.nextafter(changed[0,0],np.float32(np.inf))
    with pytest.raises(ValueError,match=f'Published h{which} mismatch'):
        layers.extract_layers(model,core,norm,torch.device('cpu'),a,b)
    assert '_chunked_layer' not in model.__dict__ and not model.encoder._forward_hooks
