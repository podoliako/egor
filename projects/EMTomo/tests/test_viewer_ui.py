"""Browserless regressions for viewer controls, publication polling, and plot state.

QuickJS is optional for the Python project; tests skip when it is unavailable.
"""
import json
from pathlib import Path

import pytest

quickjs = pytest.importorskip("quickjs")

VIEWER = Path(__file__).resolve().parents[1] / "viewer.html"


def _context():
    script = VIEWER.read_text(encoding="utf-8").split("<script>", 1)[1].split("</script>", 1)[0]
    context = quickjs.Context()
    context.eval("""
        const elements = {};
        function makeElement(tag = 'div') {
          return {
            tag, value: '0', max: '0', step: '1', style: {}, dataset: {}, children: [],
            listeners: {}, classList: {
              classes: new Set(), contains(name) { return this.classes.has(name); },
              add(name) { this.classes.add(name); }, remove(name) { this.classes.delete(name); },
              toggle(name, force) {
                const on = force === undefined ? !this.contains(name) : force;
                on ? this.add(name) : this.remove(name); return on;
              }
            },
            set innerHTML(value) { this.html = value; this.children = []; },
            get innerHTML() { return this.html || ''; },
            appendChild(child) { this.children.push(child); return child; },
            addEventListener(name, callback) { this.listeners[name] = callback; },
            closest() { return null; }, querySelectorAll() { return []; },
            querySelector() { return null; }
          };
        }
        const document = {
          getElementById(id) { return elements[id] ||= makeElement(); },
          createElement: makeElement, querySelectorAll() { return []; }, body: makeElement()
        };
        const window = {listeners: {}, addEventListener(name, cb) { this.listeners[name] = cb; }};
        const timers = new Map(); let nextTimer = 0;
        function setTimeout(cb) { timers.set(++nextTimer, cb); return nextTimer; }
        function clearTimeout(id) { timers.delete(id); }
        const rendered = {}, purged = [], resized = [], relayouts = [], downloads = [];
        const Plotly = {
          react(element, traces, layout) {
            rendered[Object.keys(elements).find(id => elements[id] === element)] = {traces, layout};
            element.layout = layout; element._fullLayout = layout;
            return Promise.resolve();
          },
          purge(element) {
            if (typeof element === 'string') element = elements[element];
            purged.push(Object.keys(elements).find(id => elements[id] === element));
            delete element.layout; delete element._fullLayout;
          },
          Plots: {resize(el) { resized.push(Object.keys(elements).find(id => elements[id] === el)); }},
          relayout(el, changes) {
            relayouts.push(changes);
            for (const [key, value] of Object.entries(changes)) {
              if (key === 'margin.t') el.layout.margin.t = value;
              else el.layout[key] = value;
            }
            return Promise.resolve();
          },
          downloadImage(el, options) {
            downloads.push({options, title: el.layout.title, layout: JSON.parse(JSON.stringify(el.layout))});
            return Promise.resolve();
          }
        };
    """)
    # The HTTP server is not running in this test, so skip the initial fetch.
    context.eval(script.replace("\ninit();", "\n// initial HTTP fetch disabled"))
    context.eval("""
      ST.run = 'run-test'; ST.meta = {
        grid_info: {coarse_side_m:[240000,120000,120000], coarse_shape:[24,12,12],
                    coarse_cell_size:10000, fine_cell_size:2500, fine_shape:[96,48,48]}
      };
      ST.iter = 0; ST.info = {iterations: [0], completed_iterations: 1, planned_iterations: 12, n_events: 1, n_stations: 1};
    """)
    return context


def _drain(context):
    for _ in range(500):
        if not context.execute_pending_job():
            return
    raise AssertionError('QuickJS promise jobs did not settle')


def _value(context, expression):
    return json.loads(context.eval(f"JSON.stringify({expression})"))


def test_y_control_is_in_km_and_default_step_is_one_tenth_of_width():
    context = _context()
    context.eval("configureModelY(true)")
    assert _value(context, "ST.vel.yStep") == 12
    assert _value(context, "Number(document.getElementById('vel-y').max)") == 120
    assert _value(context, "Number(document.getElementById('vel-y').step)") == 12
    assert _value(context, "inversionResolution()") == "&nx=192&ny=96&nz=96"
    context.eval("setModelY(60)")
    assert _value(context, "ST.vel.y") == 60
    context.eval("""
      const field = document.getElementById('vel-y-step');
      field.value = '5'; field.listeners.change({target: field});
      setModelY(25);
    """)
    assert _value(context, "ST.vel.y") == 25
    assert _value(context, "Number(document.getElementById('vel-y').step)") == 5


def test_diagnostic_y_slider_and_station_overlay_use_kilometres():
    context = _context()
    context.eval("""
      configureModelY(true);
      syncY('rc-y', 'rc-y-v', 'rc-y-num', 12, ST.rc);
      document.getElementById('rc-y-inc').listeners.click();
    """)
    assert _value(context, "ST.rc.y") == 12
    assert _value(context, "Number(document.getElementById('rc-y').max)") == 120
    assert _value(context, "Number(document.getElementById('rc-y').step)") == 12
    assert _value(context, "document.getElementById('rc-y-v').textContent") == "12.00 km"
    context.eval("ST.meta.station_locs = [[60000,30000,120000]]")
    assert _value(context, "stationTraces(-1, false, 12, false)[0].x") == [60]
    assert _value(context, "stationTraces(-1, false, 12, false)[0].y") == [120]
    context.eval("""
      const step = document.getElementById('vel-y-step');
      step.value = '5'; step.listeners.change({target: step});
    """)
    assert _value(context, "ST.rc.y") == 10
    assert _value(context, "document.getElementById('rc-y-v').textContent") == "10.00 km"
    assert _value(context, "Number(document.getElementById('rc-y').step)") == 5


def test_native_truth_block_axes_and_inversion_share_color_scale():
    context = _context()
    context.eval("""
      configureModelY(true);
      ST.vel.y = 60;
      const truth = {
        slice:[[4750,5250],[5250,4750],[4750,5250],[5250,4750]],
        shape:[4,2], full_shape:[4,2,2], cell_size:60000,
        vmin:4750, vmax:5250,
        x_km:[30,90,150,210], z_km:[30,90],
        x_edges_km:[0,60,120,180,240], z_edges_km:[0,60,120]
      };
      const inv = {
        slice:[[4900,5100],[5000,5000]], shape:[2,2], full_shape:[192,96,96],
        cell_size:1250, vmin:4900, vmax:5100,
        x_km:[0.625,1.875], z_km:[0.625,1.875]
      };
      const urls = [];
      const fetch = url => {
        urls.push(url);
        return Promise.resolve({ok: true, json: () => Promise.resolve(url.includes('model_type=true') ? truth : inv)});
      };
      renderVelPair();
    """)
    for _ in range(30):
        if not context.execute_pending_job():
            break
    assert _value(context, "urls") == [
        "/api/runs/run-test/slice?type=model&model_type=iter&iter=0&y_km=60&nx=192&ny=96&nz=96",
        "/api/runs/run-test/slice?type=model&model_type=true&y_km=60",
    ]
    for key, expected in (("cmin", 4750), ("cmax", 5250), ("cauto", False)):
        assert _value(context, f"rendered['plot-vel'].layout.coloraxis.{key}") == expected
        assert _value(context, f"rendered['plot-tr'].layout.coloraxis.{key}") == expected
    for plot in ('plot-vel', 'plot-tr'):
        assert _value(context, f"rendered['{plot}'].layout.coloraxis.colorscale") == 'Cividis'
        assert _value(context, f"rendered['{plot}'].traces[0].colorscale") == 'Cividis'
        for axis in ('xaxis', 'yaxis'):
            assert _value(context, f"rendered['{plot}'].layout.{axis}.tickmode") == 'auto'
            assert _value(context, f"rendered['{plot}'].layout.{axis}.nticks") == 6
            assert _value(context, f"rendered['{plot}'].layout.{axis}.tickvals || null") is None
    assert _value(context, "rendered['plot-tr'].traces[0].x") == [30, 90, 150, 210]
    assert _value(context, "rendered['plot-tr'].traces[0].y") == [30, 90]
    assert _value(context, "rendered['plot-vel'].traces[1].x.slice(0,6)") == [0, 0, None, 60, 60, None]
    assert _value(context, "rendered['plot-vel'].traces[1].x.length") == (5 + 3) * 3


def test_dense_truth_edges_stay_in_grid_not_axis_labels():
    context = _context()
    context.eval("ST.truthEdges = {x: Array.from({length:25}, (_, i) => i * 10), "
                 "z: Array.from({length:13}, (_, i) => i * 10)}")
    for axis in ('xaxis', 'yaxis'):
        assert _value(context, f"mkLayout(24,12).{axis}.tickmode") == 'auto'
        assert _value(context, f"mkLayout(24,12).{axis}.nticks") == 6
        assert _value(context, f"mkLayout(24,12).{axis}.tickvals || null") is None
    assert _value(context, "gridTrace(24,12).x.slice(0,9)") == [0, 0, None, 20, 20, None, 40, 40, None]
    assert _value(context, "gridTrace(24,12).y.length") == (13 + 13) * 3
    context.eval("ST.meta.grid_info.coarse_side_m = [8000, 4000, 4000]; ST.truthEdges = null")
    assert _value(context, "mkLayout(8,4).xaxis.range") == [0, 8]
    assert _value(context, "mkLayout(8,4).yaxis.range") == [4, 0]
    assert _value(context, "mkLayout(8,4).xaxis.tickmode") == 'auto'
    assert _value(context, "mkLayout(8,4).xaxis.nticks") == 6


def test_dense_true_grid_overlay_uses_real_boundaries_and_keeps_endpoints():
    context = _context()
    context.eval("ST.truthEdges = {x: Array.from({length:97}, (_, i) => i * 2.5), "
                 "z: Array.from({length:49}, (_, i) => i * 2.5)}; ST.truthEdges.x[7] = 17.75")
    trace = _value(context, "gridTrace(96,48)")
    x_edges = [i * 2.5 for i in range(0, 97, 7)] + [240]
    x_edges[1] = 17.75
    z_edges = [i * 2.5 for i in range(0, 49, 4)]
    assert len(x_edges) <= 16 and len(z_edges) <= 16
    assert trace['x'][::3][:len(x_edges)] == x_edges
    assert trace['y'][len(x_edges) * 3::3] == z_edges
    assert len(trace['x']) == (len(x_edges) + len(z_edges)) * 3
    assert _value(context, "ST.truthEdges.x[7]") == 17.75
    assert _value(context, "ST.truthEdges.x.length") == 97
    assert _value(context, "gridTrace(96,48,[1,1],true).x.slice(0,9)") == [0, 0, None, 2.5, 2.5, None, 5, 5, None]
    assert _value(context, "gridTrace(96,48,[1,1],true).x.length") == (97 + 49) * 3
    context.eval("ST.showGrid = false")
    assert _value(context, "gridTrace(96,48)") is None


def test_small_true_grid_overlay_keeps_every_boundary():
    context = _context()
    context.eval("ST.truthEdges = {x: Array.from({length:13}, (_, i) => i * 20), "
                 "z: Array.from({length:7}, (_, i) => i * 20)}")
    trace = _value(context, "gridTrace(12,6)")
    assert trace['x'][::3][:13] == [i * 20 for i in range(13)]
    assert trace['y'][13 * 3::3] == [i * 20 for i in range(7)]
    assert len(trace['x']) == (13 + 7) * 3


def test_events_fade_continuously_without_grid_snapping():
    context = _context()
    context.eval("ST.meta.event_locs = [[1000, 0, 5000], [1000, 1000, 5000], [1000, 12000, 5000], [1000, 120000, 5000]]")
    opacity = _value(context, "eventTraces(-1, false, 0, false)[0].marker.opacity")
    assert 1 == opacity[0] > opacity[1] > opacity[2] > opacity[3] > 0
    context.eval("ST.meta.grid_info.fine_cell_size = 100; ST.vel.yStep = 30")
    assert _value(context, "eventTraces(-1, false, 0, false)[0].marker.opacity") == opacity
    assert _value(context, "eventTraces(-1, false, 1, false)[0].marker.opacity[1]") == 1


def test_surface_station_markers_are_not_clipped():
    context = _context()
    context.eval("ST.meta.station_locs = [[1000, 0, 0]]")
    assert _value(context, "stationTraces(-1, false, 0, false)[0].cliponaxis") is False
    assert _value(context, "mkLayout(4,2).margin.t") >= 24
    assert _value(context, "mkLayout(4,2).yaxis.layer") == "below traces"


def test_weights_navigate_fine_layers_and_find_reference_event():
    context = _context()
    context.eval("configureModelY(true); syncWeightsY(); ST.meta.event_locs = [[1000, 30500, 5000]]")
    assert _value(context, "ST.wt.y") == 1.25
    assert _value(context, "Number(document.getElementById('wt-y').max)") == 47
    context.eval("document.getElementById('wt-y-inc').listeners.click()")
    assert _value(context, "ST.wt.y") == 3.75
    context.eval("document.getElementById('wt-true-y').listeners.click()")
    assert _value(context, "Number(document.getElementById('wt-y').value)") == 12
    assert _value(context, "ST.wt.y") == 31.25
    context.eval("const step = document.getElementById('vel-y-step'); step.value = '7'; step.listeners.change({target:step})")
    assert _value(context, "ST.wt.y") == 31.25
    assert _value(context, "Number(document.getElementById('wt-y').step)") == 1
    context.eval("ST.truthEdges = {x:[0,60,120,180,240],z:[0,60,120]}")
    assert _value(context, "gridTrace(96,48,[1,1],true).x.slice(0,6)") == [0, 0, None, 2.5, 2.5, None]
    assert _value(context, "mkLayout(96,48,'w',null).xaxis.tickmode") == 'auto'


def test_header_bounds_run_selector_and_removes_scale_badge():
    html = VIEWER.read_text(encoding="utf-8")
    assert 'id="run-control"' in html
    assert '#run-select { width: 100%; min-width: 0;' in html
    assert '>Shared scale<' not in html


def test_by_event_mean_exact_top_n_and_original_ids():
    context = _context()
    context.eval("""
      ST.meta.reference_event_ids = ['event-Z', '<event-A>', 'event-M'];
      const rows = [{event:2,rms:4}, {event:1,rms:4}, {event:0,rms:2}];
      _renderHypoHist({rows, valueKey:'rms', plotId:'plot-hypoq-hist', emptyId:'hypoq-hist-empty',
        yLabel:'RMS', hoverFmt:'%{y}', topN:1, hilightEv:0});
      _renderHypoWorst({rows, valueKey:'rms', listId:'hypoq-worst-list', topN:1,
        hilightEv:1, fmt:v=>v.toFixed(2), onClick:()=>{}});
    """)
    _drain(context)
    shape = _value(context, "rendered['plot-hypoq-hist'].layout.shapes[0]")
    assert (shape['xref'], shape['yref']) == ('paper', 'y')
    assert (shape['x0'], shape['x1']) == (0, 1)
    assert shape['y0'] == shape['y1'] == pytest.approx(10 / 3)
    trace = _value(context, "rendered['plot-hypoq-hist'].traces[0]")
    assert trace['x'] == [2, 1, 0]
    assert trace['marker']['color'] == [
        'rgba(74,127,212,0.55)', 'rgba(201,64,64,0.75)', '#a04800']
    assert trace['customdata'] == ['event-M', '&lt;event-A&gt;', 'event-Z']
    assert trace['marker']['line']['width'] == [0, 0, 2]
    assert '&lt;event-A&gt;' in _value(context, "elements['hypoq-worst-list'].innerHTML")
    assert '<event-A>' not in _value(context, "elements['hypoq-worst-list'].innerHTML")
    assert _value(context, "rankedEvents(rows, 'rms', 1).map(r=>r.event)") == [1]
    assert 'Distribution' not in VIEWER.read_text(encoding='utf-8')


def test_run_summary_counts_follow_name_and_use_text_nodes():
    context = _context()
    context.eval("""
      populateRunSelect([
        {id:'run_checker_20261009_120000',completed_iterations:5,planned_iterations:12},
        {id:'unsafe<name>',completed_iterations:0,planned_iterations:null}
      ], 'run_checker_20261009_120000');
    """)
    assert _value(context, "elements['run-select'].children[0].label") == '2026-10-09'
    assert _value(context, "elements['run-select'].children[0].children[0].textContent") == 'checker — 5/12 ит. · 12:00:00'
    assert _value(context, "elements['run-select'].children[1].children[0].textContent") == 'unsafe<name> — 0/? ит.'
    assert _value(context, "elements['run-select'].children[1].children[0].innerHTML") == ''
    assert _value(context, "elements['run-select'].value") == 'run_checker_20261009_120000'
    html = VIEWER.read_text(encoding='utf-8')
    assert '<select id="run-select"><option' in html
    assert '</select><option' not in html


@pytest.mark.parametrize('run_name,description,expected', [
    ('checker', 'checker_v1.4_mode-soft_grid-24', 'checker — 5/12 ит. · v1.4 · mode-soft · grid-24 · 12:00:00'),
    ('Checker board', 'checker-board_v1.4_mode-soft', 'Checker board — 5/12 ит. · v1.4 · mode-soft · 12:00:00'),
    ('checker_grid', 'checker_grid_v1.4', 'checker_grid — 5/12 ит. · v1.4 · 12:00:00'),
    ('checker', 'checker_mode-soft', 'checker — 5/12 ит. · mode-soft · 12:00:00'),
    (None, 'checker_v1.4_mode-soft', 'checker — 5/12 ит. · v1.4 · mode-soft · 12:00:00'),
    ('run_checker_v1.4_mode-soft_20261009_120000', 'checker_v1.4_mode-soft', 'checker — 5/12 ит. · v1.4 · mode-soft · 12:00:00'),
    ('', 'checker_grid_v1.4_mode-soft', 'checker · grid — 5/12 ит. · v1.4 · mode-soft · 12:00:00'),
    ('other', 'checker_v1.4_mode-soft', 'other — 5/12 ит. · checker · v1.4 · mode-soft · 12:00:00'),
    ('check', 'checker_v1.4_mode-soft', 'check — 5/12 ит. · checker · v1.4 · mode-soft · 12:00:00'),
    ('checker', 'archive_checker_v1.4', 'checker — 5/12 ит. · archive · checker · v1.4 · 12:00:00'),
    ('<name>', 'checker_v1.4', '<name> — 5/12 ит. · checker · v1.4 · 12:00:00'),
])
def test_run_metadata_name_precedes_count_and_preserves_id_context(run_name, description, expected):
    context = _context()
    summary = dict(id=f'run_{description}_20261009_120000', run_name=run_name,
                   completed_iterations=5, planned_iterations=12)
    context.eval(f'populateRunSelect([{json.dumps(summary)}], {json.dumps(summary["id"])})')
    assert _value(context, "elements['run-select'].children[0].children[0].textContent") == expected
    assert _value(context, "elements['run-select'].children[0].children[0].innerHTML") == ''
    assert _value(context, "elements['run-select'].children[0].label") == '2026-10-09'
    assert _value(context, "elements['run-select'].value") == summary['id']


def test_quality_axis_is_post_update_and_selected_state_is_pre_update():
    context = _context()
    context.eval("""
      ST.info.iterations = [0,1,2];
      ST.quality.data = [{iter:0,avg_abs_pct_dev:5}, {iter:1,avg_abs_pct_dev:4},
        {iter:2,avg_abs_pct_dev:3}, {iter:3,avg_abs_pct_dev:2}];
      ST.iter = 1; renderQuality();
    """)
    assert _value(context, "rendered['plot-quality'].traces[0].x") == [1, 2, 3]
    assert _value(context, "rendered['plot-quality'].layout.xaxis.title.text") == 'Applied updates'
    assert _value(context, "rendered['plot-quality'].layout.shapes[0].x0") == 1
    context.eval("ST.vel.type = 'initial'; renderQuality()")
    assert _value(context, "rendered['plot-quality'].layout.shapes") == []
    context.eval("ST.vel.type = 'delta_s'; renderQuality(); syncIterationControls()")
    assert _value(context, "rendered['plot-quality'].layout.shapes[0].x0") == 2
    assert _value(context, "elements['model-state-label'].textContent") == 'Update 1 → 2'
    context.eval("ST.vel.type = 'iter'; ST.iter = 0; renderQuality()")
    assert _value(context, "rendered['plot-quality'].layout.shapes") == []


def _sample_slice(context):
    context.eval("""
      const sample = {slice:[[4900,5100],[5000,5000]],shape:[2,2],full_shape:[2,2,2],
        cell_size:10000,vmin:4900,vmax:5100};
      const urls = [];
    """)


@pytest.mark.parametrize('failed', ['velocity', 'truth'])
def test_velocity_and_truth_fail_independently(failed):
    context = _context()
    _sample_slice(context)
    context.eval(f"const failed = {json.dumps(failed)}")
    context.eval("""
      const fetch = url => {
        urls.push(url);
        const isTruth = url.includes('model_type=true');
        const fail = failed === 'truth' ? isTruth : !isTruth;
        return Promise.resolve({ok:!fail,status:404,json:()=>Promise.resolve(sample)});
      };
      renderVelPair();
    """)
    _drain(context)
    good, bad = ('plot-vel', 'plot-tr') if failed == 'truth' else ('plot-tr', 'plot-vel')
    assert _value(context, f"Boolean(rendered['{good}'])") is True
    assert _value(context, f"Boolean(rendered['{bad}'])") is False
    assert _value(context, f"elements['{bad}'].style.display") == 'none'
    assert _value(context, 'purged') == []
    assert _value(context, f"rendered['{good}'].layout.coloraxis.cmin") == 4900


def test_api_failure_is_not_reported_as_missing_truth():
    context = _context()
    _sample_slice(context)
    context.eval("""
      const fetch = url => Promise.resolve({ok:!url.includes('model_type=true'),status:500,
        json:()=>Promise.resolve(sample)});
      renderVelPair();
    """)
    _drain(context)
    assert _value(context, "elements['tr-empty'].textContent") == 'HTTP 500'
    assert _value(context, "Boolean(rendered['plot-vel'])") is True


def test_no_completed_iterations_fetch_only_initial_and_truth():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.info.iterations = []; ST.info.completed_iterations = 0;
      const fetch = url => { urls.push(url); return Promise.resolve({ok:true,json:()=>Promise.resolve(sample)}); };
      renderAll(); reloadGWeights(); reloadRcWeights();
    """)
    _drain(context)
    urls = _value(context, 'urls')
    assert len(urls) == 2
    assert any('model_type=initial' in url for url in urls)
    assert any('model_type=true' in url for url in urls)
    assert not any('/weights_list' in url or 'type=weights' in url for url in urls)
    assert _value(context, "elements['iter-slider'].disabled") is True
    assert _value(context, "selectedUpdateState()") == 0
    context.eval("urls.length = 0; ST.vel.type = 'delta_s'; renderVelPair()")
    _drain(context)
    assert len(_value(context, 'urls')) == 1
    assert 'No completed iteration' in _value(context, "elements['vel-empty'].textContent")


@pytest.mark.parametrize('field,units', [
    ('sensitivity_diagonal', 'diag(H) (m²/s²)'),
    ('coverage_confidence', 'Confidence (0–1)'),
    ('delta_s', 'Δs (s/m)'),
])
def test_extra_fields_use_named_slice_routes_and_independent_units(field, units):
    context = _context()
    _sample_slice(context)
    context.eval(f"ST.vel.type = {json.dumps(field)}")
    context.eval("""
      const fetch = url => { urls.push(url); return Promise.resolve({ok:true,json:()=>Promise.resolve(sample)}); };
      renderVelPair();
    """)
    _drain(context)
    assert f'type={field}&iter=0' in _value(context, 'urls[0]')
    assert _value(context, "rendered['plot-vel'].layout.coloraxis.colorbar.title.text") == units
    assert _value(context, "rendered['plot-tr'].layout.coloraxis.cmin") == 4900
    if field == 'coverage_confidence':
        assert _value(context, "rendered['plot-vel'].layout.coloraxis.cmin") == 0
        assert _value(context, "rendered['plot-vel'].layout.coloraxis.cmax") == 1
    elif field == 'delta_s':
        assert _value(context, "rendered['plot-vel'].layout.coloraxis.cmin") == -5100
        assert _value(context, "rendered['plot-vel'].layout.coloraxis.colorscale[0][1]") == '#3a78c9'


def test_fixed_scale_is_cached_by_run_field_and_velocity_pair_remains_shared():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.vel.scale = 'fixed';
      const fetch = url => { urls.push(url); return Promise.resolve({ok:true,json:()=>Promise.resolve({...sample})}); };
      renderVelPair();
    """)
    _drain(context)
    context.eval('sample.vmin = 4500; sample.vmax = 5500; ST.vel.y = 60; renderVelPair()')
    _drain(context)
    for plot in ('plot-vel', 'plot-tr'):
        assert _value(context, f"rendered['{plot}'].layout.coloraxis.cmin") == 4900
        assert _value(context, f"rendered['{plot}'].layout.coloraxis.cmax") == 5100
    context.eval("ST.vel.scale = 'auto'; renderVelPair()")
    _drain(context)
    assert _value(context, "rendered['plot-vel'].layout.coloraxis.cmin") == 4500
    context.eval("ST.vel.scale = 'fixed'; ST.colorRanges.clear(); renderVelPair()")
    _drain(context)
    assert _value(context, "rendered['plot-tr'].layout.coloraxis.cmax") == 5500


def test_unchanged_refresh_only_fetches_summary_and_does_not_follow_new_run():
    context = _context()
    context.eval("""
      const urls = [];
      const fetch = url => {
        urls.push(url); return Promise.resolve({ok:true,json:()=>Promise.resolve([
          {id:'new-run',iterations:[],completed_iterations:0,planned_iterations:12},
          {id:'run-test',iterations:[0],completed_iterations:1,planned_iterations:12}
        ])});
      };
      doAutoRefresh();
    """)
    _drain(context)
    assert _value(context, 'urls') == ['/api/runs/summary']
    assert _value(context, 'ST.run') == 'run-test'
    assert _value(context, 'ST.iter') == 0
    assert _value(context, 'rendered') == {}
    assert _value(context, 'ST.followIteration') is False
    assert _value(context, 'ST.followRun') is False


def test_refresh_new_completion_updates_diagnostics_not_unchanged_spatial_plots():
    context = _context()
    context.eval("""
      const urls = [];
      const nextInfo = {id:'run-test',iterations:[0,1],completed_iterations:2,planned_iterations:12,n_events:1,n_stations:1};
      const fetch = url => {
        urls.push(url);
        const data = url.endsWith('/summary') ? [nextInfo] : url.endsWith('/info') ? nextInfo
          : url.endsWith('/quality') ? [{iter:0,avg_abs_pct_dev:5},{iter:1,avg_abs_pct_dev:4}]
          : url.endsWith('/timing') ? [{iter:0,elapsed_s:2},{iter:1,elapsed_s:3}] : {};
        return Promise.resolve({ok:true,json:()=>Promise.resolve(data)});
      };
      doAutoRefresh();
    """)
    _drain(context)
    urls = _value(context, 'urls')
    assert urls[:2] == ['/api/runs/summary', '/api/runs/run-test/info']
    assert not any('/slice' in url or '/weights_list' in url for url in urls)
    assert _value(context, 'ST.iter') == 0
    assert _value(context, "elements['iter-slider'].max") == 1
    assert _value(context, "rendered['plot-quality'].traces[0].x") == [1, 2]
    context.eval('urls.length = 0; doAutoRefresh()')
    _drain(context)
    assert _value(context, 'urls') == ['/api/runs/summary']


def test_weight_select_uses_real_sparse_indices_and_ignores_stale_event_response():
    context = _context()
    context.eval("""
      ST.info.n_events = 3;
      const pending = [], urls = [];
      const fetch = url => { urls.push(url); return new Promise(resolve=>pending.push(resolve)); };
      ST.G.event = 0; reloadGWeights();
      ST.G.event = 2; reloadGWeights();
      pending[1]({ok:true,json:()=>Promise.resolve(['weight_9','weight_3'])});
    """)
    _drain(context)
    context.eval("pending[0]({ok:true,json:()=>Promise.resolve(['weight_0'])})")
    _drain(context)
    assert _value(context, 'ST.G.weight') == 3
    assert _value(context, "elements['g-wt'].children.map(o=>Number(o.value))") == [3, 9]
    assert _value(context, "Number(elements['g-wt'].value)") == 3
    context.eval("ST.G.event = 99; ST.wt.event = -5; ST.rc.event = 12; ST.G.station = 9; syncSelections()")
    assert _value(context, '[ST.G.event, ST.wt.event, ST.rc.event, ST.G.station]') == [2, 0, 2, 0]
    assert _value(context, "Number(elements['g-ev'].value)") == 2


def test_iteration_change_snaps_to_published_states_and_reloads_hypotheses():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.info.iterations = [0,3]; ST.G.weight = ST.rc.weight = 9;
      const fetch = url => {
        urls.push(url);
        const data = url.includes('/weights_list') ? ['weight_4','weight_9']
          : url.includes('/hypo_metrics') ? {} : sample;
        return Promise.resolve({ok:true,json:()=>Promise.resolve(data)});
      };
      setIter(2);
    """)
    _drain(context)
    assert _value(context, 'ST.iter') == 3
    assert _value(context, "Number(elements['iter-slider'].value)") == 3
    assert _value(context, 'ST.G.weight') == 9
    assert _value(context, "Number(elements['g-wt'].value)") == 9
    assert sum('/weights_list?iter=3' in url for url in _value(context, 'urls')) == 2
    assert not any('iter=2' in url for url in _value(context, 'urls'))


def test_same_run_viewport_survives_controls_empty_data_and_plotly_reset():
    context = _context()
    context.eval("trackPlotRender('plot-vel', [], mkLayout(24,12), PLY_CFG)")
    _drain(context)
    context.eval("""
      elements['plot-vel'].layout.xaxis.range = [20,80];
      elements['plot-vel'].layout.yaxis.range = [70,10];
      ST.iter = 1; ST.vel.type = 'coverage_confidence';
      emptyP('plot-vel','vel-empty','Temporarily empty');
      trackPlotRender('plot-vel', [], mkLayout(24,12), PLY_CFG);
    """)
    _drain(context)
    assert _value(context, "rendered['plot-vel'].layout.xaxis.range") == [20, 80]
    assert _value(context, "rendered['plot-vel'].layout.yaxis.range") == [70, 10]
    assert _value(context, "rendered['plot-vel'].layout.uirevision") == 'run-test:plot-vel'
    assert _value(context, 'purged') == []
    context.eval("""
      elements['plot-vel'].layout.xaxis.range = [0,240];
      elements['plot-vel'].layout.yaxis.range = [120,0];
      trackPlotRender('plot-vel', [], mkLayout(24,12), PLY_CFG);
    """)
    _drain(context)
    assert _value(context, "rendered['plot-vel'].layout.xaxis.range") == [0, 240]
    context.eval("ST.run = 'different'; trackPlotRender('plot-vel', [], mkLayout(24,12), PLY_CFG)")
    _drain(context)
    assert _value(context, "rendered['plot-vel'].layout.uirevision") == 'different:plot-vel'


def test_export_adds_context_and_restores_title_without_axis_relayout():
    context = _context()
    _sample_slice(context)
    context.eval("""
      const fetch = url => Promise.resolve({ok:true,json:()=>Promise.resolve(sample)});
      renderVelPair();
    """)
    _drain(context)
    context.eval("""
      elements['plot-vel'].layout.xaxis.range = [20,80];
      ST.vel.y = 60; downloadPlot('plot-vel', 'velocity');
    """)
    _drain(context)
    download = _value(context, 'downloads[0]')
    assert 'run-test_iter_0_iter_y_60_velocity' == download['options']['filename']
    assert 'run-test' in download['title']['text'] and 'Y 60 km' in download['title']['text']
    assert download['layout']['xaxis']['range'] == [20, 80]
    assert _value(context, "elements['plot-vel'].layout.title") == {}
    assert _value(context, "elements['plot-vel'].layout.margin.t") == 26
    assert all(not any(key.startswith(('xaxis', 'yaxis')) for key in change)
               for change in _value(context, 'relayouts'))


def test_run_switch_clears_old_data_immediately_and_ignores_stale_model():
    context = _context()
    _sample_slice(context)
    context.eval("""
      const pending = [];
      const fetch = url => new Promise(resolve=>pending.push({url,resolve}));
      renderVelPair(); loadRun('next-run');
    """)
    assert _value(context, "elements['plot-vel'].style.display") == 'none'
    assert _value(context, 'ST.meta') is None
    assert _value(context, 'purged').count('plot-vel') == 1
    context.eval("""
      pending[0].resolve({ok:true,json:()=>Promise.resolve(sample)});
      pending[1].resolve({ok:true,json:()=>Promise.resolve(sample)});
      pending.slice(2).forEach(p=>p.resolve({ok:false,status:500}));
    """)
    _drain(context)
    assert _value(context, 'rendered') == {}
    assert _value(context, "elements['vel-empty'].textContent") == 'HTTP 500'
    assert _value(context, 'ST.run') == 'next-run'


def test_linked_worst_selection_updates_all_event_panels():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.info.n_events = 3; ST.meta.reference_event_ids = ['Z','A','M'];
      const fetch = url => Promise.resolve({ok:true,json:()=>Promise.resolve(
        url.includes('/weights_list') ? ['weight_7'] : sample)});
      _highlightHypoEvent(2, 'hypoD');
    """)
    _drain(context)
    assert _value(context, '[ST.hypoQ.hilightEv,ST.hypoD.hilightEv,ST.wt.event,ST.G.event,ST.rc.event]') == [2] * 5
    assert _value(context, "['wt-ev','g-ev','rc-ev'].map(id=>Number(elements[id].value))") == [2] * 3
    assert _value(context, '[ST.G.weight,ST.rc.weight]') == [7, 7]


def test_model_header_subgrid_and_independent_resize_are_guarded():
    context = _context()
    context.eval('observePlotSizes()')  # ResizeObserver absent in QuickJS.
    html = VIEWER.read_text(encoding='utf-8')
    assert '#model-row > .panel { display: grid; grid-row: span 2; grid-template-rows: subgrid;' in html
    context.eval("""
      let resizeCallback;
      const ResizeObserver = class {constructor(callback) {resizeCallback=callback;} observe() {}};
      const body = {querySelector:()=>elements['plot-vel']};
      trackPlotRender('plot-vel', [], mkLayout(24,12), PLY_CFG);
      observePlotSizes();
      resizeCallback([{target:body,contentRect:{width:600,height:400}}]);
    """)
    _drain(context)
    assert _value(context, 'resized') == ['plot-vel']
    assert _value(context, 'relayouts') == []


def test_header_fallback_measures_wrap_shrinks_and_does_not_touch_plot_axes():
    context = _context()
    context.eval("""
      const CSS = {supports:(property,value)=>false};
      const observers = [], frames = [];
      const ResizeObserver = class {
        constructor(callback) {this.callback=callback;this.targets=[];observers.push(this);}
        observe(target) {this.targets.push(target);}
      };
      window.requestAnimationFrame = callback => frames.push(callback);
      const headers = ['vel-header','tr-header'].map(id=>document.getElementById(id));
      const panels = [makeElement(),makeElement()];
      const naturalHeights = [84.5,34];
      headers.forEach((header,i)=>{
        header.closest = ()=>panels[i];
        header.getBoundingClientRect = ()=>({height:Math.max(naturalHeights[i],parseFloat(header.style.minHeight)||0)});
      });
      trackPlotRender('plot-vel', [], mkLayout(24,12), PLY_CFG);
      elements['plot-vel'].layout.xaxis.range = [20,80];
      observeModelHeaders();
    """)
    _drain(context)
    assert _value(context, 'observers[0].targets.length') == 2
    assert _value(context, 'headers.map(header=>header.style.minHeight)') == ['84.5px'] * 2
    context.eval('naturalHeights[0]=119.25; observers[0].callback(); observers[0].callback()')
    assert _value(context, 'frames.length') == 1
    context.eval('frames.shift()()')
    assert _value(context, 'headers.map(header=>header.style.minHeight)') == ['119.25px'] * 2
    context.eval('naturalHeights[0]=50; naturalHeights[1]=70; observers[0].callback(); frames.shift()()')
    assert _value(context, 'headers.map(header=>header.style.minHeight)') == ['70px'] * 2
    context.eval("panels[0].classList.add('fullscreen'); observers[0].callback(); frames.shift()()")
    assert _value(context, 'headers.map(header=>header.style.minHeight)') == ['', '']
    context.eval("panels[0].classList.remove('fullscreen'); observers[0].callback(); frames.shift()()")
    assert _value(context, 'headers.map(header=>header.style.minHeight)') == ['70px'] * 2
    assert _value(context, "elements['plot-vel'].layout.xaxis.range") == [20, 80]
    assert _value(context, 'relayouts') == []
    assert _value(context, 'purged') == []


def test_header_fallback_is_disabled_with_subgrid_and_guarded_without_browser_apis():
    context = _context()
    context.eval('observeModelHeaders()')  # No CSS or ResizeObserver in the default mock.
    context.eval("""
      const supportCalls = [];
      const CSS = {supports:(property,value)=>{supportCalls.push([property,value]);return true;}};
      const ResizeObserver = class {constructor() {throw new Error('Subgrid must not use the header fallback');}};
      observeModelHeaders();
    """)
    assert _value(context, 'supportCalls') == [['grid-template-rows', 'subgrid']]
    assert _value(context, "elements['vel-header']?.style.minHeight ?? null") is None


def test_header_fallback_guards_missing_geometry_and_uses_timer_without_animation_frames():
    context = _context()
    context.eval('const CSS = {supports:()=>false}; observeModelHeaders();')  # ResizeObserver absent.
    context.eval("""
      let callback;
      const ResizeObserver = class {constructor(cb) {callback=cb;} observe() {}};
      observeModelHeaders();  // Geometry absent in lightweight mocks.
      const headers = ['vel-header','tr-header'].map(id=>document.getElementById(id));
      let height = 90;
      headers.forEach(header=>header.getBoundingClientRect=()=>({height}));
      observeModelHeaders();
      height=40; callback();
      timers.forEach(cb=>cb()); timers.clear();
    """)
    assert _value(context, 'headers.map(header=>header.style.minHeight)') == ['40px'] * 2


def test_escape_closes_fullscreen_panels():
    context = _context()
    context.eval("""
      const panel = makeElement(); panel.classList.add('fullscreen');
      const button = makeElement(); panel.querySelector=()=>button;
      document.querySelectorAll = selector => selector === '.panel.fullscreen' ? [panel] : [];
      document.body.classList.add('has-fullscreen');
      window.listeners.keydown({key:'Escape'});
    """)
    assert _value(context, "panel.classList.contains('fullscreen')") is False
    assert _value(context, "document.body.classList.contains('has-fullscreen')") is False
    assert _value(context, 'button.textContent') == '⤢'


@pytest.mark.parametrize('follow', [False, True])
def test_first_publication_respects_follow_mode_and_auto_initial(follow):
    context = _context()
    _sample_slice(context)
    context.eval(f'ST.followIteration = {json.dumps(follow)}')
    context.eval("""
      ST.info.iterations = []; ST.info.completed_iterations = 0;
      ST.autoInitial = true; ST.vel.type = 'initial'; elements['vel-type'].value = 'initial';
      const next = {id:'run-test',iterations:[0],completed_iterations:1,planned_iterations:12,n_events:1,n_stations:1};
      const fetch = url => {
        urls.push(url);
        const data = url.endsWith('/summary') ? [next] : url.endsWith('/info') ? next
          : url.includes('/weights_list') ? ['weight_4'] : url.includes('/slice') ? sample
          : url.includes('/hypo_metrics') ? {} : [];
        return Promise.resolve({ok:true,json:()=>Promise.resolve(data)});
      };
      doAutoRefresh();
    """)
    _drain(context)
    expected_type = 'iter' if follow else 'initial'
    assert _value(context, 'ST.vel.type') == expected_type
    assert _value(context, "elements['vel-type'].value") == expected_type
    assert any(f'model_type={expected_type}' in url for url in _value(context, 'urls'))
    assert _value(context, 'ST.iter') == 0
    assert _value(context, "elements['iter-slider'].disabled") is False


def test_follow_latest_toggle_immediately_selects_latest_published_iteration():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.info.iterations = [0,2];
      const fetch = url => Promise.resolve({ok:true,json:()=>Promise.resolve(
        url.includes('/weights_list') ? ['weight_0'] : url.includes('/hypo_metrics') ? {} : sample)});
      elements['cb-follow-iter'].listeners.change({target:{checked:true}});
    """)
    _drain(context)
    assert _value(context, 'ST.iter') == 2
    assert _value(context, 'ST.followIteration') is True
    assert _value(context, 'ST.followRun') is False


def test_follow_new_run_is_independent_of_follow_iteration():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.followRun = true;
      const next = {id:'new-run',iterations:[],completed_iterations:0,planned_iterations:12,n_events:1,n_stations:1};
      const fetch = url => {
        urls.push(url);
        const data = url.endsWith('/summary') ? [next] : url.endsWith('/info') ? next
          : url.endsWith('/meta') ? ST.meta : url.includes('/slice') ? sample : [];
        return Promise.resolve({ok:true,json:()=>Promise.resolve(data)});
      };
      doAutoRefresh();
    """)
    _drain(context)
    assert _value(context, 'ST.run') == 'new-run'
    assert _value(context, 'ST.vel.type') == 'initial'
    assert _value(context, 'ST.followIteration') is False
    assert not any('/weights_list' in url or 'model_type=iter' in url for url in _value(context, 'urls'))
    assert _value(context, "elements['run-select'].value") == 'new-run'


def test_late_velocity_response_cannot_overwrite_new_y_field_selection():
    context = _context()
    _sample_slice(context)
    context.eval("""
      const pending = [];
      const fetch = url => new Promise(resolve=>pending.push({url,resolve}));
      renderVelPair();
      ST.vel.type = 'coverage_confidence'; ST.vel.y = 60; renderVelPair();
      pending.slice(2).forEach(p=>p.resolve({ok:true,json:()=>Promise.resolve({...sample,vmin:0,vmax:1})}));
    """)
    _drain(context)
    context.eval("pending.slice(0,2).forEach(p=>p.resolve({ok:true,json:()=>Promise.resolve(sample)}))")
    _drain(context)
    assert _value(context, "rendered['plot-vel'].layout.coloraxis.cmax") == 1
    assert _value(context, "elements['plot-vel'].dataset.renderField") == 'coverage_confidence'
    assert _value(context, "elements['plot-vel'].dataset.renderY") == '60'


def test_weight_loading_blocks_wrong_indices_and_old_iteration_responses():
    context = _context()
    context.eval("""
      ST.info.iterations = [0,1]; ST.G.weight = 9;
      const pending = [], urls = [];
      const fetch = url => { urls.push(url); return new Promise(resolve=>pending.push(resolve)); };
      reloadGWeights(); renderG(true);
      ST.iter = 1; reloadGWeights();
      pending[0]({ok:true,json:()=>Promise.resolve(['weight_9'])});
      pending[1]({ok:true,json:()=>Promise.resolve(['weight_3'])});
    """)
    _drain(context)
    assert not any('/slice' in url for url in _value(context, 'urls'))
    assert _value(context, 'ST.G.weight') == 3
    assert _value(context, 'ST.G.weightsLoading') is False
    assert _value(context, "Number(elements['g-wt'].value)") == 3


def test_aggregate_ray_counts_are_available_without_saved_g_hypothesis_folders():
    context = _context()
    _sample_slice(context)
    context.eval("""
      ST.rc.weight = null;
      const fetch = url => {urls.push(url);return Promise.resolve({ok:true,json:()=>Promise.resolve(sample)});};
      renderRayCount(true);
    """)
    _drain(context)
    assert 'type=ray_count' in _value(context, 'urls[0]')
    assert _value(context, "Boolean(rendered['plot-rc'])") is True


def test_only_one_panel_can_be_fullscreen():
    context = _context()
    context.eval("""
      const panels = [makeElement(), makeElement()], buttons = [makeElement(),makeElement()];
      panels.forEach((p,i)=>p.querySelector=()=>buttons[i]);
      document.querySelectorAll = selector => selector === '.panel.fullscreen'
        ? panels.filter(p=>p.classList.contains('fullscreen')) : [];
      toggleFullscreen(panels[0],buttons[0]); toggleFullscreen(panels[1],buttons[1]);
    """)
    assert _value(context, "panels.map(p=>p.classList.contains('fullscreen'))") == [False, True]
    assert _value(context, 'buttons.map(b=>b.textContent)') == ['⤢', '✕']
    context.eval('toggleFullscreen(panels[1],buttons[1])')
    assert _value(context, "panels.map(p=>p.classList.contains('fullscreen'))") == [False, False]


def test_network_errors_are_explicit_and_refresh_guard_is_released():
    context = _context()
    context.eval("""
      const fetch = url => Promise.reject(new Error('network unavailable'));
      renderVelPair(); doAutoRefresh();
    """)
    _drain(context)
    assert 'API error: network unavailable' == _value(context, "elements['vel-empty'].textContent")
    assert 'API error: network unavailable' == _value(context, "elements['tr-empty'].textContent")
    assert 'API error: network unavailable' == _value(context, "elements['refresh-status'].textContent")
    assert _value(context, 'ST._refreshing') is False
