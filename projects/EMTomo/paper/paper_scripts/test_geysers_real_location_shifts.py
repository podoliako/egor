"""Synthetic post-relocation diagnostics; no real predictions or output files."""
from io import BytesIO
from pathlib import Path
import copy
import hashlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_geysers_real_location_shifts as shifts


@pytest.fixture
def geometry():
    return dict(reference_latitude=38.80275, reference_longitude=-122.7707,
                north_km_per_latitude_degree=110.574, x_azimuth_degrees=135.,
                y_azimuth_degrees=225., geographic_reference_in_new_xy_m=[13000., 3500.],
                top_elevation_m=1166., z_direction='positive_down',
                domain_origin_m=[0, 0, 0], domain_extent_m=[23000, 11000, 7000])


def test_projection_reference_and_depth(geometry):
    shifts.validate_geometry(geometry)
    position = shifts.geographic_to_local(38.80275, -122.7707, 0., geometry)
    np.testing.assert_allclose(position, [13000, 3500, 1166], atol=1e-9)
    position = shifts.geographic_to_local(38.80275, -122.7707, [1., -0.803], geometry)
    np.testing.assert_allclose(position[:, 2], [2166, 363], atol=1e-9)


def test_projection_rotation_and_roundtrip(geometry):
    lat0, lon0 = geometry['reference_latitude'], geometry['reference_longitude']
    one_km_north = shifts.geographic_to_local(lat0 + 1 / 110.574, lon0, 0, geometry)
    one_km_east = shifts.geographic_to_local(lat0, lon0 + 1 / (111.320 * np.cos(np.deg2rad(lat0))), 0, geometry)
    reference = np.array([13000, 3500, 1166])
    d = 1000 / np.sqrt(2)
    np.testing.assert_allclose(one_km_north - reference, [-d, -d, 0], atol=1e-8)
    np.testing.assert_allclose(one_km_east - reference, [d, -d, 0], atol=1e-8)
    geographic = np.array([[lat0, lon0, 0], [38.82, -122.81, 2.5], [38.77, -122.71, -1.]])
    local = shifts.geographic_to_local(*geographic.T, geometry)
    np.testing.assert_allclose(shifts.local_to_geographic(local, geometry), geographic, rtol=0, atol=1e-11)


def test_geometry_rejects_wrong_vertical_direction(geometry):
    geometry['z_direction'] = 'positive_up'
    with pytest.raises(ValueError, match='positive-down'):
        shifts.validate_geometry(geometry)


def test_summary_distances_signs_fractions_and_p90():
    reference = np.zeros((3, 3))
    position = np.array([[3, 4, 0], [0, 0, 10], [-3, -4, -10]])
    result = shifts.shift_summary(position, reference)
    assert result['event_count'] == 3
    assert result['median_distance_3d_m'] == 10
    assert result['p90_distance_3d_m'] == pytest.approx(10 + .8 * (np.sqrt(125) - 10))
    assert result['median_horizontal_distance_m'] == 5
    assert result['median_signed_xyz_m'] == [0, 0, 0]
    assert result['median_absolute_vertical_shift_m'] == 10
    assert result['fraction_deeper'] == pytest.approx(1 / 3)
    assert result['fraction_shallower'] == pytest.approx(1 / 3)
    assert result['fraction_same_depth'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('position,reference', [([], []), ([[1, 2]], [[1, 2]]),
    ([[1, 2, np.nan]], [[1, 2, 3]]), ([[1, 2, 3]], [[1, 2, 3], [4, 5, 6]])])
def test_summary_invalid_inputs_rejected(position, reference):
    with pytest.raises(ValueError, match='matching nonempty finite'):
        shifts.shift_summary(position, reference)


def test_pairwise_summary_and_event_csv_use_fitted_initial():
    catalog = np.zeros((2, 3))
    initial = np.array([[0, 0, 10], [0, 0, 20.]])
    positions = dict(initial=initial, hard=initial + [3, 4, 0], soft=initial + [3, 4, 12])
    summary = shifts.summarize(positions, catalog)
    assert summary['against_initial_fitted']['hard']['median_distance_3d_m'] == 5
    assert summary['against_initial_fitted']['soft']['median_distance_3d_m'] == 13
    assert summary['hard_soft']['median_distance_3d_m'] == 12
    assert summary['hard_soft']['median_signed_xyz_m'] == [0, 0, 12]
    assert summary['hard_soft']['fraction_deeper'] == 1
    rows = list(shifts.event_rows(['a', 'b'], positions, catalog, np.zeros((2, 3))))
    assert len(rows) == 2 and rows[1]['event_id'] == 'b'
    assert rows[0]['hard_initial_distance_3d_m'] == 5
    assert rows[0]['soft_hard_distance_3d_m'] == 12
    assert rows[1]['initial_minus_catalog_dz_m'] == 20


@pytest.fixture
def prediction_arrays():
    return dict(event_ids=np.array(['b', 'a']), station_ids=np.array(['Y', 'X']),
                position_m=np.array([[4., 5., 6.], [1., 2., 3.]]),
                predicted_relative_s=np.ones((2, 2)), origin_relative_s=np.ones(2), train_nll=np.ones(2))


def test_npz_alignment_uses_ids_not_row_order(prediction_arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **prediction_arrays)
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as arrays:
        result = shifts.validate_predictions(arrays, ['a', 'b'], ['X', 'Y'])
    np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize('field,value', [
    ('event_ids', np.array(['a', 'a'])), ('event_ids', np.array(['a', 'c'])),
    ('event_ids', np.array([1, 2])), ('station_ids', np.array(['X', 'Z'])),
    ('position_m', np.zeros((2, 2))), ('train_nll', np.array([1., np.nan])),
    ('predicted_relative_s', np.zeros((2, 3))),
])
def test_prediction_contract_rejects_bad_ids_shapes_and_values(prediction_arrays, field, value):
    prediction_arrays[field] = value
    with pytest.raises(ValueError):
        shifts.validate_predictions(prediction_arrays, ['a', 'b'], ['X', 'Y'])


def test_missing_prediction_field(prediction_arrays):
    del prediction_arrays['origin_relative_s']
    with pytest.raises(ValueError, match='Missing prediction fields'):
        shifts.validate_predictions(prediction_arrays, ['a', 'b'], ['X', 'Y'])


def test_mask_hash_encoding_is_bytewise_and_order_sensitive():
    mask = np.array([[True, False], [True, True]])
    assert shifts.mask_sha256(mask) == hashlib.sha256(bytes([1, 0, 1, 1])).hexdigest()
    assert shifts.mask_sha256(mask) == shifts.mask_sha256(np.asfortranarray(mask))
    assert shifts.mask_sha256(mask) != shifts.mask_sha256(mask.T)


@pytest.fixture
def binding_ledger():
    path = Path(shifts.__file__).resolve()
    digest = shifts.sha(path)
    return dict(holdout_opened=False, binding=dict(
        shortlist=50, dmin=7, grid_m=125, noise=[.01, .05, .02],
        training_sha256='training-file-hash', train_mask_sha256='actual-mask-hash',
        origin='TRAIN-only precision weighted offset', prediction='MAP location',
        model=dict(source_sha256={str(path): digest})))


def test_binding_checks_protocol_model_hash_and_actual_mask(binding_ledger):
    path = Path(shifts.__file__).resolve()
    hashes = {}
    result = shifts.validate_binding(binding_ledger, 'training-file-hash', 'actual-mask-hash',
                                     path, shifts.sha(path), hashes)
    assert result == binding_ledger['binding']
    assert hashes[shifts.path_label(path)] == shifts.sha(path)


@pytest.mark.parametrize('field,value', [('shortlist', 128), ('dmin', 2), ('noise', [.01, 0., .02]),
    ('grid_m', 250), ('training_sha256', 'wrong'), ('train_mask_sha256', None),
    ('origin', 'catalog origin'), ('prediction', 'posterior mean')])
def test_binding_rejects_incompatible_protocol(binding_ledger, field, value):
    binding_ledger['binding'][field] = value
    path = Path(shifts.__file__).resolve()
    with pytest.raises(ValueError):
        shifts.validate_binding(binding_ledger, 'training-file-hash', 'actual-mask-hash', path, shifts.sha(path), {})


def test_binding_rejects_holdout_or_wrong_model(binding_ledger):
    path = Path(shifts.__file__).resolve()
    other = copy.deepcopy(binding_ledger)
    other['holdout_opened'] = True
    with pytest.raises(ValueError, match='holdout_opened'):
        shifts.validate_binding(other, 'training-file-hash', 'actual-mask-hash', path, shifts.sha(path), {})
    with pytest.raises(ValueError, match='expected final velocity'):
        shifts.validate_binding(binding_ledger, 'training-file-hash', 'actual-mask-hash', path, 'wrong', {})


def test_source_hash_mismatch_fails_closed():
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        shifts.verify_file(Path(shifts.__file__), '0' * 64, {})


def test_missing_inputs_fail_immediately_without_catalog_read(monkeypatch):
    monkeypatch.setattr(Path, 'is_file', lambda self: False)
    monkeypatch.setattr(Path, 'read_text', lambda self: pytest.fail('Must not read catalog or manifest yet'))
    with pytest.raises(FileNotFoundError, match='no waiting or outputs written'):
        shifts.load_inputs(shifts.DEFAULT_INPUT)


def test_plot_three_projections_depth_down_and_memory_render(geometry):
    import matplotlib.pyplot as plt
    catalog = np.array([[5000., 4000., 2000.], [-1000., 2000., 3000.]])
    positions = {mode: catalog + [100, 200, 300] for mode in shifts.MODES}
    stations = np.array([[1000., 2000., 500.]])
    figure = shifts.make_figure(positions, catalog, stations, geometry)
    assert len(figure.axes) == 9
    for row in range(3):
        assert not figure.axes[3 * row].yaxis_inverted()
        assert figure.axes[3 * row + 1].yaxis_inverted()
        assert figure.axes[3 * row + 2].yaxis_inverted()
        assert figure.axes[3 * row].get_xlim()[0] < -1.
    for format_name in ('png', 'pdf'):
        stream = BytesIO()
        figure.savefig(stream, format=format_name, dpi=40)
        assert stream.tell() > 1000
    plt.close(figure)
