#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/dlewis3/Desktop/AI/DWARF-v2')
RUN_ROOT = ROOT / 'runs/dsqg_w_reset_quality_20260704_105504_20k'
SESSION = 'dsqg_w_quality_20k'
VARIANT = 'w_typed_aux0'
LOG = RUN_ROOT / 'pretrain' / VARIANT / 'trainer.stdout.log'
SEM_ROOT = RUN_ROOT / 'semantic_transfer'
STEP_RE = re.compile(r'\[ep\d+ step (\d+)/(\d+)\] ce=([0-9.]+).*? ([0-9]+) tok/s(.*)')
BENCH_RE = re.compile(r'\[BENCH\].*?steady_tok_s=([0-9]+).*')
PPL_RE = re.compile(r'ep\d+: ppl=([0-9.]+).*')
MILESTONES = [250, 500, 750, 1000, 1250]

def tmux_alive() -> bool:
    return subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

def parse_log():
    text = LOG.read_text(errors='ignore') if LOG.exists() else ''
    steps = STEP_RE.findall(text)
    bench = BENCH_RE.findall(text)
    ppl = PPL_RE.findall(text)
    fatal = ('[FATAL]' in text) or ('Traceback' in text) or ('OutOfMemoryError' in text)
    if not steps:
        return {'step': 0, 'total': 1250, 'ce': None, 'tok_s': None, 'tail': '', 'bench': bench[-1] if bench else None, 'ppl': ppl[-1] if ppl else None, 'fatal': fatal}
    step, total, ce, tok_s, tail = steps[-1]
    return {'step': int(step), 'total': int(total), 'ce': float(ce), 'tok_s': int(tok_s), 'tail': tail.strip(), 'bench': int(bench[-1]) if bench else None, 'ppl': float(ppl[-1]) if ppl else None, 'fatal': fatal}

def sem_files():
    if not SEM_ROOT.exists():
        return []
    return sorted(str(p.relative_to(RUN_ROOT)) for p in SEM_ROOT.rglob('*') if p.is_file())

last_reported = 0
reported_done = False
reported_sem = False
print(f'PROGRESS_UPDATE watcher started for {RUN_ROOT}', flush=True)
while True:
    state = parse_log()
    step = state['step']
    for m in MILESTONES:
        if last_reported < m <= step:
            print(f"PROGRESS_UPDATE {VARIANT} step {step}/{state['total']} ce={state['ce']} tok_s={state['tok_s']} run_root={RUN_ROOT}", flush=True)
            last_reported = m
    if state['fatal']:
        print(f"PROGRESS_UPDATE {VARIANT} fatal/traceback detected at step {step}; inspect {LOG}", flush=True)
        break
    if not reported_done and (state.get('ppl') is not None or step >= state['total']):
        print(f"PROGRESS_UPDATE {VARIANT} training complete step={step}/{state['total']} ce={state['ce']} tok_s={state['tok_s']} ppl={state.get('ppl')} bench={state.get('bench')} run_root={RUN_ROOT}", flush=True)
        reported_done = True
    files = sem_files()
    if files and not reported_sem:
        print('PROGRESS_UPDATE semantic-transfer files detected: ' + json.dumps(files[-8:]), flush=True)
        reported_sem = True
    if reported_done and reported_sem and not tmux_alive():
        print('PROGRESS_UPDATE watcher complete: tmux session ended and semantic files exist', flush=True)
        break
    if reported_done and not tmux_alive():
        print('PROGRESS_UPDATE watcher complete: tmux session ended; semantic files not detected yet', flush=True)
        break
    time.sleep(60)
