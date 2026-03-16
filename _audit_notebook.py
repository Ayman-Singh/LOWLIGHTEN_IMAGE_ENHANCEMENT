"""Audit the Kaggle notebook JSON for correctness."""
import json, sys

path = r'd:\Low-lighten image\ELIEI_Implementation\ELIEI_Kaggle_Training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []

# Check nbformat
nf = nb.get('nbformat')
print(f'nbformat: {nf}')
if nf != 4:
    errors.append(f'nbformat is {nf}, expected 4')

# Check kernelspec
meta = nb.get('metadata', {})
ks = meta.get('kernelspec', {})
print(f'kernelspec: {json.dumps(ks)}')
if ks.get('name') != 'python3':
    errors.append(f'kernelspec.name = {ks.get("name")}, expected python3')
if ks.get('language') != 'python':
    errors.append(f'kernelspec.language = {ks.get("language")}, expected python')
if 'display_name' not in ks:
    errors.append('kernelspec.display_name missing')

li = meta.get('language_info', {})
print(f'language_info: {json.dumps(li)}')

# Check all cells
cells = nb.get('cells', [])
print(f'\nTotal cells: {len(cells)}')
for i, cell in enumerate(cells):
    ct = cell.get('cell_type', 'MISSING')
    src = ''.join(cell.get('source', []))
    has_outputs = 'outputs' in cell
    has_exec = 'execution_count' in cell
    
    if ct == 'code':
        if not has_outputs:
            errors.append(f'Cell {i} (code): missing "outputs" key')
        if not has_exec:
            errors.append(f'Cell {i} (code): missing "execution_count" key')
    
    first_line = src.split('\n')[0][:70] if src else '(empty)'
    print(f'  [{i}] {ct:8s} outputs={has_outputs} exec_count={has_exec}  {first_line}')

if errors:
    print(f'\n=== ERRORS ({len(errors)}) ===')
    for e in errors:
        print(f'  !! {e}')
    sys.exit(1)
else:
    print('\n=== ALL CHECKS PASSED ===')
