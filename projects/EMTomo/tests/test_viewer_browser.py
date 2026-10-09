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
from test_viewer_api import viewer, _save_viewer_cycle

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


def _assert_alignment(page):
    geometry = page.evaluate("""() => ['plot-vel','plot-tr'].map(id => {
      const el = document.getElementById(id), rect = el.getBoundingClientRect(), size = el._fullLayout._size;
      return [rect.top + size.t, rect.top + size.t + size.h];
    })""")
    assert geometry[0] == pytest.approx(geometry[1], abs=1)
    heights = page.evaluate("""() => ['vel-header','tr-header'].map(id => document.getElementById(id).getBoundingClientRect().height)""")
    assert heights[0] == pytest.approx(heights[1], abs=1)


@pytest.mark.parametrize("width", [1560, 1100, 800])
def test_wrapped_model_headers_and_plot_areas_align(browser_viewer, width):
    page, *_ = browser_viewer
    page.set_viewport_size({"width": width, "height": 1000})
    page.wait_for_timeout(450)
    _assert_alignment(page)
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
