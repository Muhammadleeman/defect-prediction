# Portable defect-risk demo

Shows how well a file-level defect-prediction model would have ranked the files of a public Python
project, using only information available a year earlier.

**You need:** Python 3.8 or newer and `git`. **No `pip install`, no virtual environment.**

## Run it

1. Clone this repository and open a terminal in its folder.
2. Clone any public Python project to test on. Projects that were NOT used to train the model
   give a fair test, for example:

   ```
   git clone https://github.com/psf/black.git
   git clone https://github.com/pallets/flask.git
   ```

3. Run the demo (Windows `python`, macOS/Linux `python3`):

   ```
   python portable_demo/demo.py black --cutoff 2025-11-01 --end 2026-10-31
   ```

   Replace `black` with the path of the cloned project. A run takes about a minute.
   Without `--cutoff` and `--end` it rewinds 12 months from the last commit.

## What it does

1. Rewinds the project to the cutoff date and measures every source file as it was then.
2. Gives each file a risk score with the saved model (`models/*.json`).
3. Looks at which files really received bug-fix commits afterwards.
4. Prints how many of those files each method found among its top 20%:
   random picking, a size-only ranking (biggest files first) and the model.

## How to read the result

* **Random picking** finds about 20% of the fixed files by chance.
* **Size only** and **model** usually find much more, and usually about the same as each other.
  That is the main finding of the study: most of the model's power comes from file size.
* Small projects have few fixed files, so a difference of one file moves the percentages a lot.
  Treat it as an illustration, not as proof.

The "answer key" is a commit-message rule (a commit whose subject says fix/bug, excluding docs,
CI and similar), measured at about 84% precise in a manual audit.

## Notes

* Do not test on the projects the model was trained on (requests, aiohttp, scikit-learn, scrapy,
  matplotlib, pytest, pip, poetry). The demo prints a warning if you do.
* Numbers change if you clone at a different time, because newer commits are included. Use
  `--cutoff` and `--end` to keep runs comparable.
* Very large generated files (over 120 KB) are skipped, because they make the metric code slow.
* `vendor/radon` is a bundled copy of the stdlib-only parts of the MIT-licensed `radon` library
  (version 6.0.1), so nothing has to be installed. Its licence is in `vendor/radon/LICENSE`.

## For the project owner (one-time setup)

The model files are exported from the main project's scikit-learn models. In the project's
virtual environment, from the project root:

```
python portable_demo/export_portable_models.py
git add portable_demo
git commit -m "Add portable demo"
git push
```

The export script prints the maximum difference from the scikit-learn models, which should be
about 1e-15.
