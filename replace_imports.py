# replace_imports.py
import pathlib, re
root = pathlib.Path('.').resolve()

# exclude directories
exclude = {'.git', '.venv', 'venv', '__pycache__', 'node_modules'}

pattern_from = re.compile(r'(^|\n)(\s*)from\s+fuzzyai\b')
pattern_import = re.compile(r'(^|\n)(\s*)import\s+fuzzyai\b')
pattern_string = re.compile(r'["\']fuzzyai[./\w-]*["\']')

pyfiles = [p for p in root.rglob('*.py') if not any(part in exclude for part in p.parts)]
for f in pyfiles:
    s = f.read_text(encoding='utf-8')
    s_new = pattern_from.sub(lambda m: m.group(1) + m.group(2) + 'from zynq', s)
    s_new = pattern_import.sub(lambda m: m.group(1) + m.group(2) + 'import zynq', s_new)
    # also update occurrences inside strings like "zynq/..." -> "zynq/..."
    s_new = pattern_string.sub(lambda m: m.group(0).replace('zynq', 'zynq'), s_new)
    if s_new != s:
        f.write_text(s_new, encoding='utf-8')
        print("updated", f)
print("done")
