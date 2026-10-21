import numpy as np
import pytest
from projects.EMTomo.studies.geysers2011.preview_topography import exposed_faces, interpolate_surface, terrain_mask


def test_interpolation_is_linear_inside_and_nearest_outside():
    xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
    surface,outside=interpolate_surface(xy,np.array([0.,100.,200.]),[0.,.25,1.,2.],[0.,.25,1.])
    assert surface[0,0]==0. and surface[2,0]==100. and surface[0,2]==200.
    assert surface[1,1]==75.
    assert outside[-1,0] and surface[-1,0]==100.
    assert surface.min()>=0. and surface.max()<=200.


def test_air_mask_and_upward_surface_rounding():
    surface=np.array([[1166.,1041.,1000.,327.]])
    mask,top=terrain_mask(surface,1166.,125.,56)
    np.testing.assert_array_equal(top,[[1166.,1041.,1041.,416.]])
    assert mask[0,0].all()
    assert not mask[0,1,0] and mask[0,1,1:].all()
    assert not mask[0,3,:6].any() and mask[0,3,6:].all()
    assert np.all(top>=surface) and np.all(top-surface<125.)
    assert np.all(np.diff(mask.astype(int),axis=2)>=0)


def test_column_below_domain_is_all_air():
    mask,_=terrain_mask(np.array([[-9000.]]),1166.,125.,56)
    assert not mask.any()


@pytest.mark.parametrize("surface,cell",[(np.array([[1200.]]),125.),(np.array([[np.nan]]),125.),(np.array([[0.]]),0.)])
def test_invalid_grid_or_surface_rejected(surface,cell):
    with pytest.raises(ValueError):
        terrain_mask(surface,1166.,cell,56)


def test_air_generates_no_faces():
    vertices,values,_=exposed_faces(np.zeros((2,2,2),dtype=bool),np.ones((2,2,2)),125.,1166.)
    assert vertices.shape==(0,4,3) and values.size==0


def test_adjacent_cubes_have_no_internal_faces_and_keep_velocities():
    velocity=np.array([[[3800.]],[[4600.]]])
    original=velocity.copy()
    vertices,values,_=exposed_faces(np.ones_like(velocity,dtype=bool),velocity,125.,1166.)
    assert vertices.shape==(10,4,3)
    assert (values==3800.).sum()==5 and (values==4600.).sum()==5
    np.testing.assert_array_equal(velocity,original)
    assert vertices[:,:,2].min()==pytest.approx(1.041)
    assert vertices[:,:,2].max()==pytest.approx(1.166)


def test_interface_is_rock_face_not_air_material():
    rock=np.array([[[False,True]]])
    vertices,values,_=exposed_faces(rock,np.array([[[99999.,3800.]]]),125.,1166.)
    assert len(vertices)==6 and np.all(values==3800.)
