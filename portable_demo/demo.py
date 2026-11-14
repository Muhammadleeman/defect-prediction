"""Portable defect-risk backtest demo. Needs only Python 3.8+ and git. No pip installs.

It rewinds a Git repository to about a year ago, scores every source file with the saved model,
then checks which files really received bug fixes afterwards.

Usage:
    python portable_demo/demo.py PATH_TO_A_CLONED_REPO
    python portable_demo/demo.py PATH_TO_A_CLONED_REPO --cutoff 2025-11-01 --end 2026-10-31
"""
import argparse, ast, json, math, os, re, subprocess, sys
from collections import defaultdict
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))          # bundled copy of the 'radon' metrics code
from radon.complexity import cc_visit                      # noqa: E402
from radon.raw import analyze                              # noqa: E402

KEYWORD_RE = re.compile(r"\b(fix(e[sd]?)?|bugs?|bugfix)\b", re.I)
NOISE_RE = re.compile(
    r"\b(typos?|docs?|documentation|readme|changelog|ci|lint\w*|pre-commit|"
    r"dependabot|bump|workflow|actions?|release|spelling|flake8|mypy|ruff|format\w*)\b", re.I)
EXCLUDE_RE = re.compile(
    r"(^|/)(tests?|testing|docs?|examples?|benchmarks?|_vendor|vendor|build|scripts|asv_bench)(/|$)"
    r"|(^|/)(setup|conftest)\.py$|(^|/)test_[^/]*\.py$|_test\.py$")


def is_bug_fix(subject):
    return bool(KEYWORD_RE.search(subject)) and not NOISE_RE.search(subject)


def git(path, *args):
    return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout


def parse_log(text):
    commits = []
    for chunk in text.split("\x01")[1:]:
        head, _, rest = chunk.partition("\x03")
        h, author, msg = head.split("\x02", 2)
        files, toks, i = [], rest.split("\0"), 0
        while i < len(toks):
            parts = toks[i].lstrip("\n").split("\t")
            if len(parts) == 3:
                a, d, p = parts
                a = int(a) if a.isdigit() else 0
                d = int(d) if d.isdigit() else 0
                if p == "":                                  # rename: next two tokens are old, new
                    files.append((a, d, toks[i + 2], toks[i + 1]))
                    i += 3
                    continue
                files.append((a, d, p, None))
            i += 1
        commits.append({"hash": h, "author": author, "msg": msg.strip(), "files": files})
    return commits


def get_log(path, *args, newest_first=True):
    fmt = "--format=%x01%H%x02%an%x02%B%x03"
    extra = [] if newest_first else ["--reverse"]
    commits = parse_log(git(path, "log", "HEAD", "--no-merges", "-M", "-z", "--numstat", fmt, *extra, *args))
    alias = {}
    for c in commits:                                        # follow renames
        out = []
        for a, d, p, old in c["files"]:
            if newest_first:
                cur = alias.get(p, p)
                if old:
                    alias[old] = cur
            else:
                cur = alias.get(old, old) if old else alias.get(p, p)
                if old:
                    alias[p] = cur
            out.append((a, d, cur))
        c["files"] = out
    return commits


def fan_out(tree, tops):
    deps = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in tops:
                    deps.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            dots = "." * n.level
            if n.level > 0:
                if n.module:
                    deps.add(dots + n.module)
                else:
                    deps.update(dots + a.name for a in n.names)
            elif n.module and n.module.split(".")[0] in tops:
                deps.add(n.module)
    return len(deps)


def load_model(name):
    with open(os.path.join(HERE, "models", name), encoding="utf-8") as f:
        return json.load(f)


def predict(model, row):
    """Same maths as the saved scikit-learn pipeline: log1p, standardise, logistic regression."""
    z = 0.0
    for feat, m, s, c in zip(model["features"], model["mean"], model["scale"], model["coef"]):
        z += c * ((math.log1p(row[feat]) - m) / s)
    z += model["intercept"]
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def auc(labels, scores):
    """Probability that a random fixed file scores higher than a random unfixed one (ties count half)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks, i = [0.0] * len(scores), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(labels)
    n0 = len(labels) - n1
    return (sum(r for r, l in zip(ranks, labels) if l) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_path")
    ap.add_argument("--cutoff", help="YYYY-MM-DD (default: 12 months before the last commit)")
    ap.add_argument("--end", help="YYYY-MM-DD (default: date of the last commit)")
    ap.add_argument("--top", type=float, default=0.20, help="share of files to review (default 0.20)")
    a = ap.parse_args()

    full, size = load_model("reduced.json"), load_model("size_only.json")
    repo = os.path.basename(os.path.normpath(a.repo_path))
    if repo in {x.split("/")[-1] for x in full.get("trained_on", [])}:
        print("WARNING: %s was part of the training data, so this result is NOT a fair test.\n" % repo)

    last = date.fromisoformat(git(a.repo_path, "log", "-1", "--format=%cs", "HEAD").strip())
    end = date.fromisoformat(a.end) if a.end else last
    cutoff = date.fromisoformat(a.cutoff) if a.cutoff else last - timedelta(days=365)
    end_incl = (end + timedelta(days=1)).isoformat()
    commit = git(a.repo_path, "rev-list", "-1", "--before=" + cutoff.isoformat(), "HEAD").strip()
    if not commit:
        sys.exit("No commit before the cutoff date. Choose a later --cutoff.")

    files = [p for p in git(a.repo_path, "ls-tree", "-r", "--name-only", commit).splitlines()
             if p.endswith(".py") and not EXCLUDE_RE.search(p)]
    tops = set()
    for p in files:
        parts = p.split("/")
        if parts[0] in ("src", "lib"):
            parts = parts[1:]
        tops.add(parts[0][:-3] if parts[0].endswith(".py") else parts[0])

    stats = defaultdict(lambda: {"added": 0, "authors": set(), "fix": 0})
    for c in get_log(a.repo_path, "--before=" + cutoff.isoformat(), "--", "*.py"):   # history before cutoff
        fix = is_bug_fix(c["msg"].split("\n")[0])
        for added, _, p in c["files"]:
            s = stats[p]
            s["added"] += added
            s["authors"].add(c["author"])
            s["fix"] += int(fix)

    fixed, n_fix = set(), 0
    for c in get_log(a.repo_path, "--after=" + cutoff.isoformat(), "--before=" + end_incl, "--", "*.py",
                     newest_first=False):                                            # answer key
        if is_bug_fix(c["msg"].split("\n")[0]):
            n_fix += 1
            fixed.update(p for _, _, p in c["files"])

    rows, skipped = [], 0
    for p in files:
        src = git(a.repo_path, "show", commit + ":" + p)
        if len(src) > 120000:                       # huge generated files make the analysis crawl
            skipped += 1
            continue
        try:
            tree, raw, blocks = ast.parse(src), analyze(src), cc_visit(src)
        except Exception:
            continue
        s = stats.get(p)
        rows.append({"file": p, "loc": raw.loc, "lloc": raw.lloc, "comments": raw.comments,
                     "cc_max": max([b.complexity for b in blocks] or [0]),
                     "fan_out": fan_out(tree, tops),
                     "lines_added": s["added"] if s else 0,
                     "n_authors": len(s["authors"]) if s else 0,
                     "prior_defects": s["fix"] if s else 0})
    if skipped:
        print("(skipped %d very large generated file(s))" % skipped)
    for r in rows:
        r["fixed"] = int(r["file"] in fixed)
        r["risk_model"] = predict(full, r)
        r["risk_size"] = predict(size, r)

    n, pos = len(rows), sum(r["fixed"] for r in rows)
    k = max(1, round(a.top * n))
    print("Repository: %s" % repo)
    print("Scored as it was on %s (commit %s): %d source files" % (cutoff, commit[:8], n))
    print("Then checked fixes from %s to %s: %d bug-fix commits touched %d of those files (%.0f%%)."
          % (cutoff, end, n_fix, pos, 100.0 * pos / n))
    if pos == 0 or pos == n:
        sys.exit("Not enough variation to evaluate (no or all files fixed). Try a longer window.")
    print("Review budget: top %.0f%% = %d files.\n" % (a.top * 100, k))

    labels = [r["fixed"] for r in rows]
    print("%-26s%20s%20s%7s" % ("ranking method", "fixed files caught", "share of all fixed", "AUC"))
    print("%-26s%14.1f of %-3d%18.0f%%%8.2f" % ("random picking (expected)", a.top * pos, pos, a.top * 100, 0.5))
    for name, key in (("size only (biggest first)", "risk_size"), ("model (7 metrics)", "risk_model")):
        scores = [r[key] for r in rows]
        order = sorted(range(n), key=lambda i: -scores[i])          # stable, like the full version
        hit = sum(labels[i] for i in order[:k])
        print("%-26s%14d of %-3d%18.0f%%%8.2f" % (name, hit, pos, 100.0 * hit / pos, auc(labels, scores)))

    print("\nTop 10 files by model risk, and whether they really got a bug fix afterwards:")
    for r in sorted(rows, key=lambda r: -r["risk_model"])[:10]:
        print("  %-55s risk %.2f  %s" % (r["file"], r["risk_model"], "FIXED LATER" if r["fixed"] else "-"))
    print("\nNote: the answer key is a commit-message rule (about 84% precise in the audit), "
          "so treat this as a demo, not a proof.")


if __name__ == "__main__":
    main()
