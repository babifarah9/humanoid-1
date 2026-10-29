from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
for rel in ['humanoid1/balance.py','humanoid1/benchmark.py']:
    ast.parse((root/rel).read_text(), filename=rel)
print('PASS: v0.2 controller and diagnostic modules parse')
