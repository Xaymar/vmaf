#!/usr/bin/env python3
# compare.py CPU_DIR VULKAN_DIR: the per-frame features of every JSON log in
# CPU_DIR against the one of the same name in VULKAN_DIR; exits 1 unless they
# are all the same.
import glob
import json
import os
import sys

a, b = sys.argv[1], sys.argv[2]
bad = 0
for fa in sorted(glob.glob(os.path.join(a, '*.json'))):
    n = os.path.basename(fa)
    fb = os.path.join(b, n)
    if not os.path.exists(fb):
        print(n, 'missing')
        bad += 1
        continue
    A = json.load(open(fa))['frames']
    B = json.load(open(fb))['frames']
    worst = {}
    if len(A) != len(B):
        worst['(frames)'] = (len(A), len(B))
    for x, y in zip(A, B):
        for k, v in x['metrics'].items():
            if k not in y['metrics']:
                worst[k] = ('missing',)
                continue
            d = abs(v - y['metrics'][k])
            if d > 0 and d > worst.get(k, (0,))[0]:
                worst[k] = (d, x['frameNum'], v, y['metrics'][k])
    if worst:
        bad += 1
        print(n, 'DIFFERS')
        for k, w in sorted(worst.items()):
            print('   ', k, w)
    else:
        print(n, 'identical,', len(A), 'frames,', len(A[0]['metrics']), 'features')
sys.exit(1 if bad else 0)
