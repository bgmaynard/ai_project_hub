import os, sys, json, re, time, importlib.util, subprocess, urllib.request

ROOT   = r"C:\ai_project_hub"
CODE   = os.path.join(ROOT, "store", "code")
OUTDIR = os.path.join(ROOT, "store", "ai_shared")
REPORT_JSON = os.path.join(OUTDIR, "validator_report.json")
REPORT_MD   = os.path.join(OUTDIR, "validator_report.md")

def find_py_files(base):
    for dp,_,files in os.walk(base):
        # skip venvs and site-packages
        if "venv" in dp.lower() or "site-packages" in dp.lower():
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dp,f)

def parse_imports(path):
    imps=set()
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as fh:
            for line in fh:
                m = re.match(r"\s*import\s+([a-zA-Z0-9_\.]+)", line)
                if m: imps.add(m.group(1).split(".")[0])
                m = re.match(r"\s*from\s+([a-zA-Z0-9_\.]+)\s+import", line)
                if m: imps.add(m.group(1).split(".")[0])
    except Exception: pass
    return sorted(imps)

def check_import_available(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False

def try_pytest(code_dir):
    try:
        # run quietly; capture output
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", code_dir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120
        )
        return {"returncode": proc.returncode, "output": proc.stdout[-2000:]}
    except Exception as e:
        return {"error": str(e)}

def ping_ibkr_api():
    url = "http://localhost:5000/api/ibkr/test"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return {"ok": True, "payload": json.loads(r.read().decode("utf-8"))}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files=list(find_py_files(CODE))
    summary=[]
    missing=set()
    for p in files:
        imps=parse_imports(p)
        missing_local=[m for m in imps if m not in ("__future__",) and not check_import_available(m)]
        if missing_local:
            missing.update(missing_local)
        summary.append({"file": os.path.relpath(p, ROOT), "imports": imps, "missing_here": missing_local})

    pytest_result = try_pytest(CODE)
    ibkr = ping_ibkr_api()

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": ROOT,
        "scanned_files": len(files),
        "missing_modules_overall": sorted(missing),
        "pytest_result": pytest_result,
        "ibkr_api_probe": ibkr,
        "files": summary[:200]  # cap for size
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # quick human-readable md
    lines = []
    lines.append("# AI Validator Report")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Files scanned: {report['scanned_files']}")
    lines.append("")
    if report["missing_modules_overall"]:
        lines.append("## Missing Python modules (import not found)")
        for m in report["missing_modules_overall"]:
            lines.append(f"- `{m}` (pip install {m})")
        lines.append("")
    lines.append("## Pytest (summary)")
    lines.append(f"```\n{return_or_err:=report['pytest_result']}\n```" if "error" in report["pytest_result"] else
                 f"- returncode: {report['pytest_result'].get('returncode')}\n```\n{report['pytest_result'].get('output','')}\n```")
    lines.append("")
    lines.append("## IBKR API Probe")
    if report["ibkr_api_probe"].get("ok"):
        lines.append("- ✅ `/api/ibkr/test` reachable")
        lines.append(f"```\n{json.dumps(report['ibkr_api_probe']['payload'], indent=2)}\n```")
    else:
        lines.append(f"- ❌ Not reachable: {report['ibkr_api_probe'].get('error')}")
    with open(REPORT_MD,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Wrote:\n  {REPORT_JSON}\n  {REPORT_MD}")

if __name__ == "__main__":
    main()
