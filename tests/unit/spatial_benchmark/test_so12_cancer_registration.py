"""Synthetic geometry and contour invariants; never read clinical assets."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

SPEC=importlib.util.spec_from_file_location("so12_cancer_registration",Path(__file__).resolve().parents[3]/"scripts/analysis/so12_cancer_registration.py")
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_blue_mask_ignores_outside_handwriting_and_fills_closed_loops():
    image=Image.new("RGB",(160,120),"white")
    draw=ImageDraw.Draw(image)
    draw.rectangle((2,2,12,12),fill="blue")
    draw.ellipse((40,30,100,90),outline="blue",width=3)
    roi,stroke,records=module.annotation_mask(np.asarray(image),(25,15,150,115),1,minimum_stroke=50)
    assert roi[60,70] and not roi[5,5]
    assert records[0]["interior_pixels"]>1000 and stroke.sum()>100


def test_open_contour_is_rejected():
    image=Image.new("RGB",(150,150),"white")
    ImageDraw.Draw(image).line([(20,20),(100,20),(100,100)],fill="blue",width=3)
    with pytest.raises(ValueError,match="open"):
        module.annotation_mask(np.asarray(image),(0,0,150,150),1,minimum_stroke=50)


def test_tissue_excludes_blue_annotation():
    a=np.full((80,80,3),255,np.uint8)
    a[15:55,15:55]=[180,100,170]
    a[60:70,60:70]=[0,0,255]
    tissue=module.tissue_mask(a,(0,0,80,80),minimum=50)
    assert tissue[30,30] and not tissue[65,65]


def test_raster_owner_is_global_and_order_independent():
    square=np.array([[1,1],[5,1],[5,5],[1,5],[1,1]])
    a,count=module.rasterize_owners([square,square+2],[500,7],np.eye(2),np.zeros(2),(10,10))
    b,_=module.rasterize_owners([square+2,square],[7,500],np.eye(2),np.zeros(2),(10,10))
    assert np.array_equal(a,b)
    assert a[3,3]==7 and a[1,1]==500 and a[0,0]==-1 and count==4


def test_similarity_has_declared_scale_and_reflection():
    for reflected in (False,True):
        matrix=module.similarity_matrix(-90,.31,reflected)
        assert np.allclose(matrix.T@matrix,np.eye(2)*.31**2)
        assert np.sign(np.linalg.det(matrix))==(-1 if reflected else 1)


def test_inverse_warp_translation_preserves_source():
    source=np.zeros((20,20),bool);source[5:12,3:15]=True
    warped=module.warp_source(source,np.array([0.,0.]),1.,np.eye(2),np.array([2.,3.]),(25,25))>=.5
    expected=np.zeros((25,25),bool);expected[8:15,5:17]=True
    assert np.array_equal(warped,expected)


def test_boundary_audit_perfect_and_shifted():
    a=np.zeros((30,30),bool);a[5:25,5:25]=True
    assert module.boundary_audit(a,a,2)["silhouette_dice"]==1
    shifted=np.roll(a,3,axis=0)
    result=module.boundary_audit(a,shifted,2)
    assert result["silhouette_dice"]<1
    assert result["symmetric_boundary_distance_um"]["p95"]==6


def test_geometry_fit_recovers_asymmetric_rotation_without_reflection():
    source=np.zeros((70,85),bool)
    source[10:50,10:25]=True;source[35:55,20:60]=True
    source[5:18,57:70]=True;source[57:65,30:40]=True
    matrix=module.similarity_matrix(-75,1.05,False)
    translation=np.array([75.,75.])-matrix@np.array([40.,35.])
    target=module.warp_source(source,np.zeros(2),1,matrix,translation,(150,150))>=.5
    result=module.fit_geometry(source,np.zeros(2),1,target,1.,stride=2)[0]
    assert result["reflected"] is False
    assert abs(result["angle_degrees"]+75)<2
    assert abs(result["relative_scale"]-1.05)<.025
    assert result["silhouette_dice_soft"]>.95
