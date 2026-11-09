"""Optional real-Plotly regressions; no server process or runtime network access.

Install Playwright and Chromium, then supply EMTOMO_PLOTLY_JS pointing to the
Plotly 2.35.2 bundle (or cache it in .pytest_cache/plotly-2.35.2.min.js).
"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest

playwright = pytest.importorskip("playwright.sync_api")
from test_viewer_api import viewer, _save_viewer_cycle, _ray_status_fixture

ROOT = Path(__file__).resolve().parents[1]
RUN = "run_browser_v1.4_20261009_120000"


@pytest.fixture
def browser_viewer(viewer, request):
    client, add_run, *_ = viewer
    bundle = Path(os.environ.get("EMTOMO_PLOTLY_JS", ROOT / ".pytest_cache/plotly-2.35.2.min.js"))
    if not bundle.is_file():
        pytest.skip("Supply the local Plotly bundle via EMTOMO_PLOTLY_JS")
    rd = add_run(RUN)
    meta = json.loads((rd / "meta.json").read_text())
    meta["run_name"] = "browser"
    meta["run_params"]["n_cycles"] = 3
    (rd / "meta.json").write_text(json.dumps(meta))
    _save_viewer_cycle(rd, 1)
    _save_viewer_cycle(rd, 2, complete=False)
    for iteration in (0, 1, 2):
        for event in range(3):
            path = rd / f"iter_{iteration}/event_{event}/weights.npz"
            np.savez(path, weight_shape=[24, 12, 12], weight_indices=[[0, 0, 0]],
                     positions=[[0., 0., 0.]], weight_values=[1.])
    errors, requests = [], []
    with playwright.sync_playwright() as manager:
        try:
            browser = manager.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        page = browser.new_page(viewport={"width": 1560, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        fallback = getattr(request, "param", False)
        if fallback:
            page.add_init_script("""
              const originalSupports = CSS.supports.bind(CSS);
              CSS.supports = (...args) => args.includes('subgrid') ? false : originalSupports(...args);
            """)

        def route_request(route):
            url = urlsplit(route.request.url)
            if url.hostname == "cdn.plot.ly":
                route.fulfill(content_type="application/javascript", body=bundle.read_bytes())
                return
            requests.append(url.path + (f"?{url.query}" if url.query else ""))
            response = client.get(url.path + (f"?{url.query}" if url.query else ""))
            body = response.data
            if fallback and url.path == "/":
                body = body.replace(b"@supports (grid-template-rows: subgrid)", b"@supports (unknown: never)")
            route.fulfill(status=response.status_code, content_type=response.content_type, body=body)

        page.route("**/*", route_request)
        page.goto("http://emtomo.test/")
        page.wait_for_function("document.getElementById('plot-vel').dataset.renderRun === ST.run")
        page.wait_for_function("document.getElementById('plot-vel-top').dataset.renderRun === ST.run")
        page.wait_for_function("document.getElementById('plot-hypod-hist')._fullLayout !== undefined")
        yield page, rd, requests, add_run
        browser.close()
    assert not errors, errors


def _ranges(page, plot="plot-vel"):
    return page.evaluate("""id => {
      const el = document.getElementById(id);
      return [el._fullLayout.xaxis.range, el._fullLayout.yaxis.range];
    }""", plot)


def _wait_iteration(page, iteration):
    page.wait_for_function("""i => {
      const el = document.getElementById('plot-vel');
      return el.dataset.renderIter === String(i) && el.style.display !== 'none';
    }""", arg=iteration)


def test_hypocenter_dots_fit_fine_cells_after_zoom_and_resize(browser_viewer):
    page, *_ = browser_viewer
    page.wait_for_function("document.getElementById('plot-wt').dataset.renderRun === ST.run")
    page.locator('#cb-wt-points').check()
    page.wait_for_function("document.getElementById('plot-wt').data.some(t => t.meta?.hypocenter)")

    def assert_dots_fit():
        result = page.evaluate("""() => {
          const plot = document.getElementById('plot-wt');
          const {xaxis: x, yaxis: z} = plot._fullLayout;
          const cellKm = csz(false) / 1000;
          return {
            cellPixels: cellKm * Math.min(
              x._length / Math.abs(x.range[1] - x.range[0]),
              z._length / Math.abs(z.range[1] - z.range[0])),
            markers: plot.data.filter(t => t.meta?.hypocenter).map(t => t.marker)
          };
        }""")
        assert result["markers"]
        for marker in result["markers"]:
            assert marker["symbol"] == "circle"
            assert marker["color"] == "#000000"
            assert marker["line"]["width"] == 0
            assert marker["size"] == pytest.approx(min(5, 0.6 * result["cellPixels"]))

    assert_dots_fit()
    page.evaluate("""async () => {
      const plot = document.getElementById('plot-wt');
      await Plotly.relayout(plot, {'xaxis.range': [0, 480], 'yaxis.range': [240, 0]});
    }""")
    assert_dots_fit()
    page.evaluate("""async () => {
      const plot = document.getElementById('plot-wt');
      await Plotly.relayout(plot, {width: 360, height: 260});
    }""")
    assert_dots_fit()


def test_ten_thousand_weights_render_hover_zoom_and_export(browser_viewer):
    page, rd, *_ = browser_viewer
    count = 10000
    positions = np.random.default_rng(42).uniform([0, 0, 0], [23, 11, 11], (count, 3))
    weights = np.arange(1, count + 1, dtype=float) / count
    for iteration in (0, 1):
        np.savez(rd / f"iter_{iteration}/event_0/weights.npz",
                 weight_shape=[24, 12, 12], weight_indices=np.floor(positions).astype(int),
                 positions=positions, weight_values=weights)
    timings = page.evaluate("""async () => {
      ST.wt.event = 0; ST.wt.dim = true; ST.wt.showPoints = true; ST.wt.showBest = false;
      const start = performance.now();
      await renderWt();
      const renderMs = performance.now() - start;
      const plot = document.getElementById('plot-wt');
      const zoom = performance.now();
      await Plotly.relayout(plot, {'xaxis.range': [0, 480], 'yaxis.range': [240, 0]});
      return {renderMs, zoomMs: performance.now() - zoom};
    }""")
    print(f"10k weights (API + render): {timings}")
    result = page.evaluate("""() => {
      const plot = document.getElementById('plot-wt');
      const traces = plot.data.filter(t => t.meta?.hypocenter);
      const trace = traces[0];
      const {xaxis: x, yaxis: z} = plot._fullLayout;
      return {
        traces: traces.length, type: trace.type, count: trace.x.length,
        first: trace.customdata[0], last: trace.customdata[trace.x.length - 1],
        opacityCount: trace.marker.opacity.length,
        expectedSize: Math.min(5, 0.6 * csz(false) / 1000 * Math.min(
          x._length / Math.abs(x.range[1] - x.range[0]),
          z._length / Math.abs(z.range[1] - z.range[0]))),
        size: trace.marker.size, svgPoints: plot.querySelectorAll('path.point').length,
        webgl: !!plot._fullLayout._plots.xy._scene
      };
    }""")
    assert result["traces"] == 1
    assert result["type"] == "scattergl" and result["webgl"]
    assert result["count"] == result["opacityCount"] == count
    assert result["svgPoints"] < 100
    assert result["size"] == pytest.approx(result["expectedSize"])
    for key, index in (("first", 0), ("last", count - 1)):
        assert result[key] == pytest.approx([index, *((positions[index] + 0.5) * 10000), weights[index]])
    page.locator('#plot-wt').scroll_into_view_if_needed()
    point = page.evaluate("""async () => {
      const plot = document.getElementById('plot-wt');
      const trace = plot.data.find(t => t.meta?.hypocenter);
      const px = trace.x[9999], py = trace.y[9999];
      await Plotly.relayout(plot, {'xaxis.range': [px - 1, px + 1], 'yaxis.range': [py + 1, py - 1]});
      const {xaxis: x, yaxis: y} = plot._fullLayout;
      const rect = plot.getBoundingClientRect();
      return {x: rect.left + x._offset + x.l2p(px), y: rect.top + y._offset + y.l2p(py)};
    }""")
    page.mouse.move(point['x'], point['y'])
    page.wait_for_function("document.querySelector('#plot-wt .hoverlayer').textContent.includes('Hypo 9999')")
    assert "w=1.0000" in page.locator('#plot-wt .hoverlayer').text_content()
    png = page.evaluate("Plotly.toImage('plot-wt', {format: 'png', width: 800, height: 450})")
    assert png.startswith("data:image/png;base64,") and len(png) > 10000
    page.evaluate("""async () => {
      ST.wt.dim = false;
      await renderWt();
    }""")
    assert page.evaluate("document.getElementById('plot-wt').data.find(t => t.meta?.hypocenter).marker.opacity") == 1
    page.evaluate("""async () => {
      ST.wt.showBest = true; ST.wt.showMean = true;
      await renderWt();
    }""")
    summaries = page.evaluate("document.getElementById('plot-wt').data.filter(t => t.meta?.weightSummary)")
    assert len(summaries) == 2
    assert all(t['type'] == 'scattergl' and t['marker']['size'] == 12 for t in summaries)
    assert summaries[0]['customdata'][0] == pytest.approx((np.average(positions, weights=weights, axis=0) + 0.5) * 10000)
    assert summaries[1]['customdata'][0] == pytest.approx((positions[-1] + 0.5) * 10000)
    page.evaluate("""async () => {
      ST.wt.event = 1;
      await renderWt();
    }""")
    assert page.evaluate("document.getElementById('plot-wt').data.find(t => t.meta?.hypocenter).type") == 'scatter'


def test_weights_visibility_controls_and_weighted_summary(browser_viewer):
    page, rd, *_ = browser_viewer
    assert not page.locator('#cb-wt-points').is_checked()
    assert page.locator('#cb-wt-best').is_checked()
    assert not page.locator('#cb-wt-mean').is_checked()
    for iteration in (0, 1):
        np.savez(rd / f"iter_{iteration}/event_0/weights.npz",
                 weight_shape=[24, 12, 12], weight_indices=[[1, 2, 3], [9, 6, 7]],
                 positions=[[1., 2., 3.], [9., 6., 7.]], weight_values=[1., 3.])
    page.evaluate("async () => { ST.wt.event = 0; await renderWt(); }")

    def overlays():
        return page.evaluate("document.getElementById('plot-wt').data.filter(t => t.meta?.hypocenter || t.meta?.weightSummary)")

    best, = overlays()
    assert best['meta'] == {'weightSummary': 'best'}
    assert best['marker']['symbol'] == 'circle'
    assert best['marker']['color'] == '#d62728'
    assert best['marker']['size'] == 12
    assert best['customdata'] == [[95000, 65000, 75000]]
    page.locator('#cb-wt-mean').check()
    page.wait_for_function("document.getElementById('plot-wt').data.some(t => t.meta?.weightSummary === 'mean')")
    mean, best = overlays()
    assert mean['meta'] == {'weightSummary': 'mean'}
    assert mean['marker']['symbol'] == 'diamond'
    assert mean['customdata'][0] == pytest.approx([75000, 55000, 65000])
    page.locator('#cb-wt-points').check()
    page.wait_for_function("document.getElementById('plot-wt').data.some(t => t.meta?.hypocenter)")
    points, mean, best = overlays()
    assert len(points['x']) == 2 and points['marker']['color'] == '#000000'
    page.evaluate("Plotly.relayout('plot-wt', {'xaxis.range': [0, 480], 'yaxis.range': [240, 0]})")
    points, mean, best = overlays()
    assert points['marker']['size'] <= 5
    assert best['marker']['size'] == mean['marker']['size'] == 12
    page.locator('#cb-wt-best').uncheck()
    page.wait_for_function("!document.getElementById('plot-wt').data.some(t => t.meta?.weightSummary === 'best')")
    assert len(overlays()) == 2
    page.locator('#cb-wt-mean').uncheck()
    page.wait_for_function("!document.getElementById('plot-wt').data.some(t => t.meta?.weightSummary)")
    assert len(overlays()) == 1
    page.locator('#cb-wt-points').uncheck()
    page.wait_for_function("!document.getElementById('plot-wt').data.some(t => t.meta?.hypocenter)")
    assert overlays() == []
    page.locator('#wt-ev').select_option('1')
    page.wait_for_function("document.getElementById('plot-wt').dataset.renderContext.includes('Ev eA')")
    assert overlays() == []


def test_velocity_palette_is_shared_and_lightness_increases(browser_viewer):
    page, *_ = browser_viewer
    palettes = page.evaluate("""() => ['plot-vel','plot-tr'].map(id => {
      const el = document.getElementById(id);
      return {name: el.layout.coloraxis.colorscale, colors: el._fullLayout.coloraxis.colorscale,
              range: [el._fullLayout.coloraxis.cmin, el._fullLayout.coloraxis.cmax]};
    })""")
    assert palettes[0] == palettes[1]
    assert palettes[0]["name"] == "Cividis"
    # Relative luminance checks the grayscale ordering of Plotly's actual palette.
    def luminance(color):
        channels = [float(part) / 255 for part in color.removeprefix("rgb(").removesuffix(")").split(",")]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722)))
    levels = [luminance(color) for _, color in palettes[0]["colors"]]
    assert all(a < b for a, b in zip(levels, levels[1:]))


def test_mean_does_not_expand_event_axis_and_selected_bar_is_distinct(browser_viewer):
    page, *_ = browser_viewer
    layout = page.evaluate("document.getElementById('plot-hypod-hist')._fullLayout")
    assert layout["xaxis"]["range"][1] < 4
    line = layout["shapes"][0]
    assert line["xref"] == "paper" and line["yref"] == "y"
    assert line["x0"] == 0 and line["x1"] == 1
    assert line["y0"] == line["y1"] > 1000
    page.evaluate("""() => {
      ST.hypoD.topN = 1;
      ST.hypoD.distanceIter = [0,1,2].map(event => ({event, dist_m: 2500}));
      ST.hypoD.hilightEv = 2;
      renderHypoData();
    }""")
    page.evaluate("() => PLOT_RENDER_PROMISES.get('plot-hypod-hist')")
    style = page.evaluate("document.getElementById('plot-hypod-hist').data[0].marker")
    assert style["color"].count("rgba(201,64,64,0.75)") == 1
    assert len(set(style["line"]["width"])) > 1
    assert "browser — 2/3 ит." in page.locator("#run-select option:checked").inner_text()


def test_zoom_survives_controls_publication_and_reset(browser_viewer):
    page, rd, requests, add_run = browser_viewer
    page.evaluate("""() => Plotly.relayout('plot-vel', {
      'xaxis.range': [40, 100], 'yaxis.range': [90, 30],
      'xaxis.autorange': false, 'yaxis.autorange': false
    })""")
    zoom = _ranges(page)
    page.evaluate("setIter(1)")
    _wait_iteration(page, 1)
    assert _ranges(page) == zoom
    page.evaluate("setModelY(72)")
    page.wait_for_function("document.getElementById('plot-vel').dataset.renderY === '72'")
    assert _ranges(page) == zoom
    page.locator("#vel-type").select_option("sensitivity_diagonal")
    page.wait_for_function("document.getElementById('plot-vel').dataset.renderField === 'sensitivity_diagonal'")
    assert _ranges(page) == zoom
    assert "diag(H)" in page.evaluate("document.getElementById('plot-vel').layout.coloraxis.colorbar.title.text")
    page.locator("#cb-grid").uncheck()
    page.evaluate("() => PLOT_RENDER_PROMISES.get('plot-vel')")
    assert _ranges(page) == zoom
    before = len([url for url in requests if "/slice?" in url])
    page.evaluate("() => doAutoRefresh()")
    assert len([url for url in requests if "/slice?" in url]) == before
    assert page.evaluate("ST.iter") == 1
    (rd / "iter_2/complete.json").write_text(json.dumps({"iter": 2, "viewer_completion_protocol": 1}))
    page.locator("#cb-follow-iter").check()
    page.evaluate("() => doAutoRefresh()")
    _wait_iteration(page, 2)
    assert _ranges(page) == zoom
    assert "3/3 ит." in page.locator("#run-select option:checked").inner_text()
    page.locator('#plot-vel .modebar-btn[data-title="Reset axes"]').click(force=True)
    page.wait_for_function("document.getElementById('plot-vel')._fullLayout.xaxis.range[0] !== 40")
    assert _ranges(page) != zoom
    reset = _ranges(page)
    page.evaluate("() => renderVelPair()")
    assert _ranges(page) == reset
    add_run("run_other_v1.4_20261009_120001")
    page.evaluate("() => loadRun('run_other_v1.4_20261009_120001')")
    page.wait_for_function("document.getElementById('plot-vel').dataset.renderRun === 'run_other_v1.4_20261009_120001'")
    assert _ranges(page) == [[0, 240], [120, 0]]


def _assert_alignment(page, top=False):
    plots = ['plot-vel-top', 'plot-tr-top'] if top else ['plot-vel', 'plot-tr']
    geometry = page.evaluate("""ids => ids.map(id => {
      const el = document.getElementById(id), rect = el.getBoundingClientRect(), size = el._fullLayout._size;
      return [rect.top + size.t, rect.top + size.t + size.h];
    })""", plots)
    assert geometry[0] == pytest.approx(geometry[1], abs=1)
    headers = ['vel-top-header', 'tr-top-header'] if top else ['vel-header', 'tr-header']
    heights = page.evaluate("ids => ids.map(id => document.getElementById(id).getBoundingClientRect().height)", headers)
    assert heights[0] == pytest.approx(heights[1], abs=1)


@pytest.mark.parametrize("width", [1560, 1100, 800])
def test_wrapped_model_headers_and_plot_areas_align(browser_viewer, width):
    page, *_ = browser_viewer
    page.set_viewport_size({"width": width, "height": 1000})
    page.wait_for_timeout(450)
    _assert_alignment(page)
    _assert_alignment(page, top=True)
    page.locator("#panel-vel .btn-full").click()
    assert page.locator(".panel.fullscreen").count() == 1
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)
    assert page.locator(".panel.fullscreen").count() == 0
    _assert_alignment(page)


@pytest.mark.parametrize("browser_viewer", [True], indirect=True)
def test_header_alignment_fallback_grows_and_shrinks(browser_viewer):
    page, *_ = browser_viewer
    for width in (800, 1560):
        page.set_viewport_size({"width": width, "height": 1000})
        page.wait_for_timeout(500)
        _assert_alignment(page)
        _assert_alignment(page, top=True)


def test_png_download_has_context_and_preserves_zoom(browser_viewer, tmp_path):
    page, *_ = browser_viewer
    page.evaluate("() => Plotly.relayout('plot-vel', {'xaxis.range': [40, 100], 'yaxis.range': [90, 30]})")
    zoom = _ranges(page)
    with page.expect_download() as pending:
        page.locator("#panel-vel .btn-dl").click()
    download = pending.value
    assert RUN in download.suggested_filename
    assert "iter_0_iter_y_0" in download.suggested_filename
    target = tmp_path / download.suggested_filename
    download.save_as(target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    page.wait_for_function("document.querySelector('#panel-vel .btn-dl').disabled === false")
    assert _ranges(page) == zoom


@pytest.mark.parametrize("failed", [False, True])
def test_ray_accounting_badge_changes_with_selected_complete_cycle(browser_viewer, failed):
    page, rd, *_ = browser_viewer
    page.wait_for_function("document.getElementById('ray-status-badge').textContent === 'Лучи: нет данных'")
    (rd / "iter_0/ray_status.json").write_text(json.dumps(_ray_status_fixture(failed)))
    page.evaluate("() => renderRayStatus()")
    badge = page.locator("#ray-status-badge")
    assert badge.inner_text() == ("Лучи: 5 / 6 · потеряно 1 · использовано 4" if failed else "Лучи: 6 / 6")
    assert badge.evaluate("el => el.classList.contains('ray-status-warning')") is failed
    assert badge.evaluate("el => el.classList.contains('ray-status-ok')") is not failed
    if failed:
        assert "Ev eM, гипотеза 0: станции 1" in badge.get_attribute("title")
    page.evaluate("setIter(1)")
    page.wait_for_function("document.getElementById('ray-status-badge').textContent === 'Лучи: нет данных'")
    assert page.evaluate("ST.rayStatus") is None
    _wait_iteration(page, 1)


def test_top_view_uses_native_horizontal_plane_and_independent_depth(browser_viewer):
    page, _, requests, _ = browser_viewer
    assert page.evaluate("document.getElementById('model-top-row').previousElementSibling.id") == 'model-row'
    assert _ranges(page, 'plot-vel-top') == [[0, 240], [0, 120]]
    assert page.evaluate("document.getElementById('plot-vel-top').layout.yaxis.title.text") == 'Y (km)'
    page.locator('#vel-top-z-num').fill('60')
    page.locator('#vel-top-z-num').press('Enter')
    page.wait_for_function("document.getElementById('plot-tr-top').dataset.renderDepthZ === '60'")
    assert page.evaluate('ST.vel.y') == 0
    assert page.evaluate('ST.top.z') == 60
    assert page.evaluate("document.getElementById('plot-tr-top').data[0].z") == [
        [4001, 4005, 4009, 4013], [4003, 4007, 4011, 4015],
    ]
    assert page.evaluate("document.getElementById('plot-tr-top').data[0].y") == [30, 90]
    assert any('model_type=true&plane=xy&z_km=60' in url for url in requests)
    scales = page.evaluate("""() => ['plot-vel-top','plot-tr-top'].map(id => {
      const c = document.getElementById(id).layout.coloraxis;
      return [c.colorscale, c.cmin, c.cmax];
    })""")
    assert scales[0] == scales[1]
    assert scales[0][0] == 'Cividis'
    events = page.evaluate("document.getElementById('plot-vel-top').data.find(t => t.marker?.symbol === 'star')")
    assert events['x'] == [5, 15, 25]
    assert events['y'] == [5, 5, 5]
    assert max(events['marker']['opacity']) < 0.1


def test_top_zoom_is_independent_and_survives_depth_fields_iterations(browser_viewer):
    page, *_ = browser_viewer
    page.evaluate("() => Plotly.relayout('plot-vel-top', {'xaxis.range':[40,100], 'yaxis.range':[20,70]})")
    top_zoom = _ranges(page, 'plot-vel-top')
    vertical_ranges = _ranges(page)
    page.evaluate('setModelZ(72)')
    page.wait_for_function("document.getElementById('plot-vel-top').dataset.renderDepthZ === '72'")
    assert _ranges(page, 'plot-vel-top') == top_zoom
    assert _ranges(page) == vertical_ranges
    page.evaluate('setIter(1)')
    page.wait_for_function("document.getElementById('plot-vel-top').dataset.renderIter === '1' && document.getElementById('plot-vel-top').style.display !== 'none'")
    assert _ranges(page, 'plot-vel-top') == top_zoom
    page.evaluate('setModelY(60)')
    page.wait_for_function("document.getElementById('plot-vel').dataset.renderY === '60'")
    assert page.evaluate('ST.top.z') == 72
    assert _ranges(page, 'plot-vel-top') == top_zoom
    page.locator('#vel-top-type').select_option('coverage_confidence')
    page.wait_for_function("document.getElementById('plot-vel-top').dataset.renderField === 'coverage_confidence'")
    assert _ranges(page, 'plot-vel-top') == top_zoom
    assert page.evaluate('ST.vel.type') == 'iter'
    assert page.evaluate("document.getElementById('plot-vel-top').layout.coloraxis.cmin") == 0
    assert page.evaluate("document.getElementById('plot-vel-top').layout.coloraxis.cmax") == 1
    page.locator('#cb-events').uncheck()
    page.evaluate('() => renderTopPair()')
    assert page.evaluate("document.getElementById('plot-vel-top').data.some(t => t.marker?.symbol === 'star')") is False
    assert _ranges(page, 'plot-vel-top') == top_zoom


def test_top_initial_is_available_before_completed_cycles(browser_viewer):
    page, _, requests, add_run = browser_viewer
    add_run('pending', iteration=False)
    start = len(requests)
    page.evaluate("() => loadRun('pending')")
    page.wait_for_function("document.getElementById('plot-vel-top').dataset.renderRun === 'pending'")
    assert page.evaluate("document.getElementById('plot-vel-top').dataset.renderField") == 'initial'
    assert page.evaluate('ST.top.z') == 0
    assert _ranges(page, 'plot-vel-top') == [[0, 240], [0, 120]]
    assert not any('model_type=iter' in url for url in requests[start:])
    assert any('model_type=initial' in url and 'plane=xy' in url for url in requests[start:])


def test_top_png_contains_xy_depth_and_preserves_zoom(browser_viewer, tmp_path):
    page, *_ = browser_viewer
    page.evaluate('setModelZ(60)')
    page.wait_for_function("document.getElementById('plot-vel-top').dataset.renderDepthZ === '60'")
    page.evaluate("() => Plotly.relayout('plot-vel-top', {'xaxis.range':[40,100], 'yaxis.range':[20,70]})")
    zoom = _ranges(page, 'plot-vel-top')
    with page.expect_download() as pending:
        page.locator('#panel-vel-top .btn-dl').click()
    download = pending.value
    assert 'XY_depthZ_60' in download.suggested_filename
    assert '_y_' not in download.suggested_filename
    path = tmp_path / download.suggested_filename
    download.save_as(path)
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    page.wait_for_function("document.querySelector('#panel-vel-top .btn-dl').disabled === false")
    assert _ranges(page, 'plot-vel-top') == zoom
