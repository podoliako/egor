"""Minimal browserless interaction checks for physical coordinates and shared scales.

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
        const document = {
          getElementById(id) {
            return elements[id] ||= {
              value: '0', max: '0', step: '1', style: {}, dataset: {},
              listeners: {}, addEventListener(name, callback) { this.listeners[name] = callback; },
              closest() { return null; }
            };
          },
          querySelectorAll() { return []; }
        };
        const window = {addEventListener() {}};
        const rendered = {};
        const Plotly = {
          react(element, traces, layout) {
            rendered[Object.keys(elements).find(id => elements[id] === element)] = {traces, layout};
            return Promise.resolve();
          },
          purge() {}
        };
    """)
    # The HTTP server is not running in this test, so skip the initial fetch.
    context.eval(script.replace("\ninit();", "\n// initial HTTP fetch disabled"))
    context.eval("""
      ST.run = 'run-test'; ST.meta = {
        grid_info: {coarse_side_m:[240000,120000,120000], coarse_shape:[24,12,12],
                    coarse_cell_size:10000, fine_cell_size:2500, fine_shape:[96,48,48]}
      };
      ST.iter = 0;
    """)
    return context


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
    assert _value(context, "rendered['plot-vel'].layout.xaxis.tickvals") == [0, 60, 120, 180, 240]
    assert _value(context, "rendered['plot-tr'].traces[0].x") == [30, 90, 150, 210]
    assert _value(context, "rendered['plot-tr'].traces[0].y") == [30, 90]
    assert _value(context, "rendered['plot-vel'].traces[1].x.slice(0,6)") == [0, 0, None, 60, 60, None]


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
    assert _value(context, "mkLayout(96,48,'w',null,true).xaxis.tickmode || null") is None


def test_header_bounds_run_selector_and_removes_scale_badge():
    html = VIEWER.read_text(encoding="utf-8")
    assert 'id="run-control"' in html
    assert '#run-select { width: 100%; min-width: 0;' in html
    assert '>Shared scale<' not in html
