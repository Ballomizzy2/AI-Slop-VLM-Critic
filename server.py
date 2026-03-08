"""
server.py — Critic Pipeline server (stdlib only)

Endpoints:
  UI / streaming (human-facing):
    POST /api/run          Upload video + audience, get job_id back
    GET  /api/stream/:id   SSE stream of pipeline logs
    GET  /api/job/:id      Poll job status + report

  Agent-facing (machine-to-machine):
    POST /api/evaluate     JSON body { video_path, audience } → returns report synchronously
                           No upload needed — pass a path the server can access

  UI:
    GET  /                 Serve ui.html

Run: python server.py
"""

import http.server
import json
import mimetypes
import os
import re
import subprocess
import sys
import uuid
import queue
import threading
import traceback
import urllib.parse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

JOBS: dict[str, dict] = {}
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _to_public_media_url(path_str: str):
    """Map an on-disk media path to a URL this server can serve."""
    try:
        path = Path(path_str).resolve()
    except Exception:
        return None

    for base_dir, url_prefix in ((UPLOAD_DIR.resolve(), "/uploads"), (OUTPUT_DIR.resolve(), "/output")):
        try:
            rel = path.relative_to(base_dir)
            rel_path = rel.as_posix()
            if not rel_path:
                return None
            return f"{url_prefix}/{urllib.parse.quote(rel_path)}"
        except ValueError:
            continue
    return None


def _resolve_safe_static_path(base_dir: Path, url_tail: str):
    """Resolve /prefix/<tail> to a file under base_dir; block traversal."""
    rel = urllib.parse.unquote(url_tail).lstrip("/")
    if not rel:
        return None

    base = base_dir.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None

    return candidate if candidate.is_file() else None


def run_pipeline_thread(job_id: str, video_path: str, audience: str = "general"):
    """Run pipeline in subprocess to avoid Whisper/numba + print-monkey-patching conflicts."""
    job = JOBS[job_id]
    q: queue.Queue = job["logs"]

    def emit(msg: str, kind: str = "log"):
        q.put(json.dumps({"kind": kind, "text": msg}))

    def classify_kind(text: str) -> str:
        if any(x in text for x in ["Saved", "complete", "[OK]", "done"]):
            return "success"
        if any(x in text for x in ["Error", "failed", "Warning", "Traceback"]):
            return "error"
        if "VERDICT" in text or "===" in text:
            return "verdict"
        if text.startswith("["):
            return "node"
        return "log"

    try:
        job["status"] = "running"
        emit("Critic Pipeline starting...", "info")
        emit(f"Video:    {os.path.basename(video_path)}", "info")
        emit(f"Audience: {audience}", "info")

        pipeline_dir = Path(__file__).parent
        runner = pipeline_dir / "run_pipeline.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(pipeline_dir)

        proc = subprocess.Popen(
            [sys.executable, str(runner), video_path, str(OUTPUT_DIR), audience],
            cwd=str(pipeline_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        report = {}
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("__REPORT__"):
                try:
                    report = json.loads(line[9:])
                except json.JSONDecodeError:
                    pass
                continue
            q.put(json.dumps({"kind": classify_kind(line), "text": line}))

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Pipeline exited with code {proc.returncode}")

        if not report:
            base = os.path.splitext(os.path.basename(video_path))[0]
            report_path = OUTPUT_DIR / f"{base}_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)

        job["report"] = report
        job["status"] = "done"
        q.put(json.dumps({
            "kind": "done",
            "report": report,
            "video_url": _to_public_media_url(video_path),
        }))

    except Exception as e:
        tb = traceback.format_exc()
        emit(f"Pipeline error: {e}", "error")
        job["status"] = "error"
        job["error"] = str(e)
        q.put(json.dumps({"kind": "error", "text": str(e)}))
    finally:
        q.put(None)


class CriticHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path.startswith("/api/stream/"):
            self._handle_stream(path.split("/api/stream/")[1])
        elif path.startswith("/api/job/"):
            job_id = path.split("/api/job/")[1]
            if job_id not in JOBS:
                self.send_json({"error": "not found"}, 404)
            else:
                job = JOBS[job_id]
                self.send_json({"status": job["status"], "report": job.get("report"), "error": job.get("error")})
        elif path in ("/", "/index.html"):
            self._serve_file(Path(__file__).parent / "ui.html", "text/html")
        elif path.startswith("/uploads/"):
            self._serve_media_file(UPLOAD_DIR, path[len("/uploads/"):])
        elif path.startswith("/output/"):
            self._serve_media_file(OUTPUT_DIR, path[len("/output/"):])
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/run":
            self._handle_run()
        elif path == "/api/evaluate":
            self._handle_evaluate()
        else:
            self.send_json({"error": "not found"}, 404)

    def _handle_run(self):
        """Accept multipart video upload, start async pipeline job."""
        content_type   = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))
        body           = self.rfile.read(content_length)
        audience       = "general"

        if "multipart/form-data" in content_type:
            boundary = content_type.split("boundary=")[1].encode()
            video_path, audience = self._parse_multipart(body, boundary)
            if not video_path:
                self.send_json({"error": "Could not parse upload"}, 400)
                return
        else:
            try:
                data       = json.loads(body)
                video_path = data.get("video_path")
                audience   = data.get("audience", "general")
            except Exception:
                self.send_json({"error": "Invalid body"}, 400)
                return

        job_id = str(uuid.uuid4())[:8]
        JOBS[job_id] = {"status": "queued", "logs": queue.Queue(), "report": None, "error": None}
        threading.Thread(target=run_pipeline_thread, args=(job_id, video_path, audience), daemon=True).start()
        self.send_json({"job_id": job_id})

    def _handle_evaluate(self):
        """
        Agent-facing endpoint. Synchronous — blocks until pipeline completes.

        Request:  POST /api/evaluate
                  Content-Type: application/json
                  { "video_path": "/path/to/video.mp4", "audience": "casual" }

        Response: The full critic report JSON.
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except Exception:
            self.send_json({"error": "Invalid JSON body"}, 400)
            return

        video_path = data.get("video_path")
        audience   = data.get("audience", "general")

        if not video_path:
            self.send_json({"error": "video_path is required"}, 400)
            return
        if not os.path.exists(video_path):
            self.send_json({"error": f"File not found: {video_path}"}, 404)
            return

        pipeline_dir = Path(__file__).parent
        runner = pipeline_dir / "run_pipeline.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(pipeline_dir)
        try:
            result = subprocess.run(
                [sys.executable, str(runner), video_path, str(OUTPUT_DIR), audience],
                cwd=str(pipeline_dir),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            if result.returncode != 0:
                self.send_json({"error": result.stdout or result.stderr or "Pipeline failed"}, 500)
                return
            report = {}
            for line in (result.stdout or "").splitlines():
                if line.startswith("__REPORT__"):
                    try:
                        report = json.loads(line[9:])
                        break
                    except json.JSONDecodeError:
                        pass
            if not report:
                base = os.path.splitext(os.path.basename(video_path))[0]
                report_path = OUTPUT_DIR / f"{base}_report.json"
                if report_path.exists():
                    with open(report_path) as f:
                        report = json.load(f)
            self.send_json(report)
        except subprocess.TimeoutExpired:
            self.send_json({"error": "Pipeline timed out"}, 500)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def _handle_stream(self, job_id: str):
        if job_id not in JOBS:
            self.send_response(404); self.end_headers(); return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = JOBS[job_id]["logs"]
        while True:
            try:
                item = q.get(timeout=30)
                if item is None:
                    self.wfile.write(b"data: __END__\n\n")
                    self.wfile.flush()
                    break
                self.wfile.write(f"data: {item}\n\n".encode())
                self.wfile.flush()
            except queue.Empty:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def _parse_multipart(self, body: bytes, boundary: bytes):
        """Parse multipart form — returns (video_path, audience)."""
        video_path = None
        audience   = "general"
        parts      = body.split(b"--" + boundary)

        for part in parts:
            if b"\r\n\r\n" not in part:
                continue
            header_end = part.find(b"\r\n\r\n")
            headers    = part[:header_end].decode(errors="replace")
            content    = part[header_end + 4:]
            if content.endswith(b"\r\n"):
                content = content[:-2]

            if 'name="audience"' in headers:
                audience = content.decode(errors="replace").strip()

            elif 'filename="' in headers:
                fn_start  = headers.find('filename="') + 10
                fn_end    = headers.find('"', fn_start)
                filename  = headers[fn_start:fn_end] or "video.mp4"
                save_path = str(UPLOAD_DIR / filename)
                with open(save_path, "wb") as f:
                    f.write(content)
                video_path = save_path

        return video_path, audience

    def _serve_file(self, path: Path, mime: str):
        if not path.exists():
            self.send_json({"error": "file not found"}, 404); return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_media_file(self, base_dir: Path, url_tail: str):
        target = _resolve_safe_static_path(base_dir, url_tail)
        if not target:
            self.send_json({"error": "file not found"}, 404)
            return

        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        file_size = target.stat().st_size
        start = 0
        end = file_size - 1
        status_code = 200

        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
            if not m:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

            start_str, end_str = m.groups()
            if start_str:
                start = int(start_str)
                end = int(end_str) if end_str else file_size - 1
            elif end_str:
                suffix_len = int(end_str)
                if suffix_len <= 0:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                start = max(file_size - suffix_len, 0)
                end = file_size - 1
            else:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

            if start >= file_size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

            end = min(end, file_size - 1)
            status_code = 206

        chunk_size = end - start + 1
        self.send_response(status_code)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(chunk_size))
        if status_code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with open(target, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                data = f.read(min(64 * 1024, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7474))
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), CriticHandler)
    print("\n[Critic Pipeline]")
    print(f"   UI:     http://localhost:{PORT}")
    print(f"   Agent:  POST http://localhost:{PORT}/api/evaluate")
    print(f"           {{\"video_path\": \"/path/to/video.mp4\", \"audience\": \"casual\"}}")
    print(f"\n   Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
