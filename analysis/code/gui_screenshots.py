"""Capture real screenshots of the updated Streamlit application (Figures 16-17): input panel, then output after 'Run Optimization'."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import ROOT
import subprocess, sys, time, os, urllib.request
from playwright.sync_api import sync_playwright

ROOT = ROOT + ""; OUT = ROOT + "analysis/figures/"; PORT = 8512
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", ROOT + "streamlit_app.py", "--server.port", str(PORT), "--server.headless", "true", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                       stdout=open(ROOT + "analysis/results/log_streamlit.txt", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        try: urllib.request.urlopen(f"http://localhost:{PORT}", timeout=2); break
        except Exception: time.sleep(2)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True); pg = b.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=2)
        pg.goto(f"http://localhost:{PORT}"); pg.wait_for_selector("text=Design Inputs", timeout=120000); time.sleep(4)
        pg.screenshot(path=OUT + "fig16_gui_input.png")
        pg.get_by_role("button", name="Run Optimization").click()
        pg.wait_for_selector("text=Strength Distribution", timeout=900000); pg.set_viewport_size({"width": 1500, "height": 1700}); time.sleep(30)
        pg.screenshot(path=OUT + "fig17_gui_output.png", full_page=True)
        b.close()
    print("screenshots saved")
finally:
    srv.terminate()
