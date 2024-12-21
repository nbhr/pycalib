# embed JS into HTML

import sys
import os

JS='./qrcodeborder.js'
SRC = ['qrtimecode.html']

for f in SRC:
    basename, ext = os.path.splitext(f)
    dst = basename + "+js.html"

    with open(f, 'r') as fin, open(dst, 'w') as fout:
        for line in fin:
            line = line.rstrip()
            if f'<script src="{JS}"></script>' in line:
                print(f'<!-- {JS} -->', file=fout)
                print('<script>', file=fout)
                print(open(JS, 'r').read(), file=fout)
                print('</script>', file=fout)
            else:
                print(line, file=fout)
