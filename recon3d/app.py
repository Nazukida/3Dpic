from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
VIEWER_SRC = os.path.join(HERE, "viewer", "index.html")
CONSOLE_SRC = os.path.join(HERE, "web", "console.html")
QUALITY_PRESETS = ["fast", "balanced", "high", "ultra"]

class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.lines: list[str] = []
        self.state= "idle"
        self.started_at = 0.0
        self.ended_at = 0.0
        self.out_dir = ""
        self.image_dir = ""
        self.quality = ""
        self.stage = ""
        self.frac = 0.0
        self.result: dict | None = None
        
    def is_running(self) -> bool:
        with self.lock:
            return self.state == "running"
    def start(self, image_dir: str, out_dir: str, quality: str, dense: bool, workers: int, voxel: float) -> tuple[bool, str]:
        with self.lock:
            if self.state == "running":
                return False, "Job is already running"
            self.state = "running"
            self.lines = []
            self.started_at = time.time()
            self.ended_at = 0.0
            self.out_dir = out_dir
            self.image_dir = image_dir
            self.quality = quality
            self.stage = "starting"
            self.frac = 0.0
            self.result = None
        cmd = [sys.executable, "-u", "-m", "recon3d.run", "--images", image_dir, "--out", out_dir, "--quality", quality]
        if not dense:
            cmd.append("--no-dense")
        if workers > 0:
            cmd += ["--workers", str(workers)]
        if voxel > 0:
            cmd += ["--voxel", str(voxel)]
            
        self._append(f"${' '.join(cmd)}")
        try:
            self.proc = subprocess.Popen(cmd, cwd = PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env = {**os.environ, "PYTHONUNBUFFERED": "1"})
        except Exception as e:
            with self.lock:
                self.state = "error"
                self.ended_at = time.time()
                self._append(f"Failed to start job: {e}")
            return False, f"Failed to start job: {e}"
        
        self.thread = threading.Thread(target=self._pump, daemon = True)
        self.thread.start()
        return True, "started"
    
    def stop(self) -> None:
        with self.lock:
            p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
            self._append("[console] stop requested.")
            with self.lock:
                if self.state == "running":
                    self.state = "stopped"
                    self.ended_at = time.time()
    
    def _append(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            
    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            self._append(line)
            self._parse_line(line)
        code = self.proc.wait()
        with self.lock:
            self.ended_at = time.time()
            if self.state == "stopped":
                pass
            elif code == 0:
                self.state = "done"
                self.stage = "done"
                self.frac = 1.0
            else:
                self.state = "error"
        if self.state == "done":
            self.load_result()
            
    _STAGE_ORDER = [
        ("feature", "提取特征 SIFT"),
        ("match", "特征匹配"),
        ("tracks", "构建轨迹"),
        ("sfm", "稀疏重建 sfm"),
        ("dense", "稠密重建 mvs"),
        ("clean", "点云清理"),
        ("pipeline] Done", "完成")
    ]
    
    def _parse_progress(self, line: str) -> None:
        stage = None
        frac = None
        m = re.search(r"\[(features|match)\].*?(\d+)\s*/\s*(\d+)", line)
        if m:
            stage = "features" if m.group(1) == " features" else "match"
            cur, tot = int(m.group(2)), int(m.group(3))
            frac = cur / tot if tot else 0.0
        md = re.search(r"depth\s+(\d+)\s*/\s*(\d+)", line)
        if md:
            stage = "dense"
            cur, tot = int(md.group(1)), int(md.group(2))
            frac = cur / tot if tot else 0.0
        if stage is None:
            for key, _label in self._STAGE_ORDER:
                if key in line:
                    stage = key.split("]")[0] if "]" in key else key
                    break
        if stage is not None:
            with self.lock:
                self.stage = stage
                if frac is not None:
                    self.frac = frac
                else:
                    self.frac = 0.0
                    
    def _load_result(self) -> None:
        info = {}
        sj = os.path.join(self.out_dir, "scene.json")
        if os.path.exists(sj):
            try:
                with open(sj) as f:
                    info = json.load(f)
            except Exception:
                pass
        files = {}
        for name in ("sparse.ply", "dense.ply", "cameras.ply"):
            p = os.path.join(self.out_dir, name)
            files[name] = os.path.getsize(p) if os.path.exists(p) else 0
        with self.lock:
            self.result = {"scene": {k: info.get(k) for k in ("num_cameras", "num_sparse_points")}, "files": files}
        
        def snapshot(self, since: int) -> dict:
            with self.lock:
                new = self.lines[since:]
                return {
                    "state": self.state,
                    "stage": self.stage,
                    "frac": round(self.frac, 3),
                    "total_lines": len(self.lines),
                    "lines": new,
                    "elapsed": round((self.ended_at or time.time()) - self.started_at, 1) if self.started_at else 0.0,
                    "out_dir": self.out_dir,
                    "result": self.result,
                }
                
JOB = Job()

IMAGES_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def _default_browse_root() -> str:
    return PROJECT_ROOT

def _windows_drivers() -> list[str]:
    drives = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        path = f"{letter}:\\"
        if os.path.exists(path):
            drives.append(path)
    return drives

def list_dir(path: str) -> dict:
    path = os.path.abspath(path) if path else _default_browse_root()
    if not os.path.isdir(path):
        path = _default_browse_root()
    entries = []
    n_images = 0
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            try:
                if os.path.isdir(full):
                    entries.append({"name": name, "dir": True})
                elif name.lower().endswith(IMAGES_EXTS):
                    n_images += 1
            except OSError:
                continue
    except PermissionError:
        pass
    parent = os.path.dirname(path.rstrip("\\/")) or path
    drives = _windows_drivers() if os.name == "nt" else []
    return {
        "path": path,
        "parent": parent,
        "dirs": entries,
        "n_images": n_images,
        "drives": drives,
        "sep": os.sep,
    }
    
#HTTP SERVER

class Handler(http.server.SimpleHTTPRequestHandler):
    server_version = "recon3d-console"
    
    def log_message(self, fmt, *args):
        pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
        
    def _send_json(self, obj, code = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        
    def _send_file(self, path, ctype):
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return
        try:
            fs = os.path.getsize(path)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(fs))
            self.end_headers()
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        except (BrokenPipeError, ConnectionResetError):
            pass
        
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        
        if route in ("/", "/index.html", "/console", "/console.html"):
            self._send_file(CONSOLE_SRC, "text/html; charset=utf-8")
            return
        if route == "/api/browse":
            try:
                since = int(qs.get("since", ["0"])[0])
            except (ValueError, TypeError):
                since = 0
            self._send_json(JOB.snapshot(since))
            return
        if route == "/api/presets":
            self._send_json({"presets": QUALITY_PRESETS,
                             "cpu": os.cpu_count() or 4,
                             "root": PROJECT_ROOT})
            return
        
        if route == "/viewer/" or route == "/viewer/index.html":
            self._send_file(VIEWER_SRC, "text/html; charset=utf-8")
            return
        if route.startswith("/viewer/"):
            rel = route[len("/viewer/"):]
            out_dir = JOB.out_dir or os.path.join(PROJECT_ROOT, "output")
            target = os.path.normpath(os.path.join(out_dir, rel))
            if not os.path.abspath(target).startswith(os.path.abspath(out_dir)):
                self.send_error(403, "Forbidden")
                return
            ctype = _content_type(target)
            self._send_file(target, ctype)
            return
        
        self.send_error(404, "Not found")
        
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        length = int(self.headers.get("Content-Length", "0")) \
            if str(self.headers.get("Content-Length", "0")).isdigit() else 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            data = {}
            
        if route == "/api/start":
            image_dir = (data.get("image_dir") or "").strip()
            if not image_dir or not os.path.isdir(image_dir):
                self._send_json({"ok": False, "error": "图片文件夹不存在"}, 400)
                return
            n_imgs = sum(1 for f in os.listdir(image_dir) if f.lower().endswith(IMAGES_EXTS))
            if n_imgs < 2:
                self._send_json({"ok": False, "error": "图片文件夹中至少需要两张图片"}, 400)
                return
            out_dir = (data.get("out_dir") or "").strip() or os.path.join(PROJECT_ROOT, "output")
            quality = data.get("quality", "balanced")
            if quality not in QUALITY_PRESETS:
                quality = "balanced"
            dense = bool(data.get("dense", True))
            try:
                workers = int(data.get("workers", 0) or 0)
            except (ValueError, TypeError):
                workers = 0
            try:
                voxel = float(data.get("voxel", 0.0) or 0.0)
            except (ValueError, TypeError):
                voxel = 0.0
            ok, msg = JOB.start(image_dir, out_dir, quality, dense, workers, voxel)
            self._send_json({"ok": ok, "msg": msg, "n_images": n_imgs, "out_dir": out_dir}, 200 if ok else 409)
            return
        if route == "/api/stop":
            JOB.stop()
            self._send_json({"ok": True})
            return
        self.send_error(404, "Not found")
        
def _content_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".html") or p.endswith(".htm"):
        return "text/html; charset=utf-8"
    if p.endswith(".js"):
        return "text/javascript; charset=utf-8"
    if p.endswith(".json"):
        return "application/json; charset=utf-8"
    if p.endswith(".ply"):
        return "application/octet-stream"
    return "application/octet-stream"

def _prepare() -> None:
    pass

def run(port: int = 8000, open_browser: bool = True) -> None:
    if not os.path.isfile(CONSOLE_SRC):
        raise SystemExit(f"console.html missing at {CONSOLE_SRC}")
    _prepare()
    chosen = None
    httpd = None
    for cand in range(port, port + 50):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", cand), Handler)
            chosen = cand
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"No free port in {port}..{port+50}")
    
    url = f"http://127.0.0.1:{chosen}/"
    print(f"[console] recon3d control panel")
    print(f"[console] open: {url}")
    print(f"[console] project root: {PROJECT_ROOT}")
    print("[console] press Ctrl+C to exit.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[console] exiting...")
            JOB.stop()
            httpd.shutdown()
            
def main(argv = None):
    p = argparse.ArgumentParser(description = "recon3d local web console")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action = "store_true")
    args = p.parse_args(argv)
    run(args.port, not args.no_open)
    
if __name__ == "__main__":
    main()