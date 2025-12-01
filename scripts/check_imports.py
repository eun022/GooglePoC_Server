#!/usr/bin/env python3
"""Scan project .py files for imports and report any modules that fail to import.

Run this from the repository root; the script assumes its working directory
is the `DotStem-api` folder (we add it to sys.path to resolve local packages).
"""
import ast
import importlib
import importlib.util
import os
import sys
from collections import defaultdict


def find_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip caches
        if '__pycache__' in dirpath:
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                yield os.path.join(dirpath, fn)


def module_name_from_path(root, path):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    # drop filename extension
    parts[-1] = parts[-1][:-3]
    # ignore files that start with a dot
    if any(p.startswith('.') for p in parts):
        return None
    return '.'.join(parts)


def collect_imports(pyfile, project_root):
    with open(pyfile, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=pyfile)
        except Exception:
            return []

    imports = []
    module_name = module_name_from_path(project_root, pyfile)
    package = ''
    if module_name and '.' in module_name:
        package = module_name.rpartition('.')[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, pyfile, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # Resolve relative imports to absolute when possible
            mod = node.module
            if node.level and module_name is not None:
                try:
                    name_to_resolve = mod or ''
                    resolved = importlib.util.resolve_name('.' * node.level + name_to_resolve, package)
                    imports.append((resolved, pyfile, node.lineno))
                except Exception:
                    # fallback: record as-is with indication of relative
                    imports.append((('.' * node.level) + (mod or ''), pyfile, node.lineno))
            else:
                if mod:
                    imports.append((mod, pyfile, node.lineno))
    return imports


def try_import(name):
    try:
        importlib.import_module(name)
        return True, None
    except Exception as e:
        return False, e


def main():
    project_root = os.getcwd()
    # ensure local package imports resolve (when running inside DotStem-api)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # scan files
    all_imports = defaultdict(list)
    for py in find_py_files(project_root):
        imports = collect_imports(py, project_root)
        for name, file, lineno in imports:
            all_imports[name].append((file, lineno))

    missing = {}
    checked = set()
    for name in sorted(all_imports.keys()):
        # skip empty names
        if not name:
            continue
        # For dotted names starting with dots (unresolved relative) mark as unresolved
        if name.startswith('.'):
            missing[name] = {
                'error': 'relative import could not be resolved statically',
                'locations': all_imports[name],
            }
            continue

        # Avoid re-checking the same root module repeatedly
        root_mod = name.split('.')[0]
        if name in checked:
            continue
        ok, err = try_import(name)
        if not ok:
            # also try root module if full dotted failed
            if name != root_mod:
                ok_root, err_root = try_import(root_mod)
                if ok_root:
                    # root exists, but submodule import failed; record that
                    missing[name] = {
                        'error': f'submodule import failed: {err}',
                        'locations': all_imports[name],
                    }
                else:
                    missing[name] = {
                        'error': f'module not found: {err_root}',
                        'locations': all_imports[name],
                    }
            else:
                missing[name] = {'error': f'module not found: {err}', 'locations': all_imports[name]}
        checked.add(name)

    # Print a concise report
    if not missing:
        print('No missing imports detected (static import checks passed).')
        return 0

    print('Missing or unresolved imports detected:\n')
    for mod, info in missing.items():
        print(f"Module: {mod}")
        print(f"  Issue: {info['error']}")
        print('  Locations:')
        for fp, ln in info['locations']:
            print(f"    - {fp}:{ln}")
        print('')

    print('Summary: %d unresolved imports.' % len(missing))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
