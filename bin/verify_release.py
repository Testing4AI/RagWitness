#!/usr/bin/env python3
"""Verify file inventory, integrity and scoped privacy guards without dependencies."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', 'verification-output'}
SECRET = re.compile(rb'(?<![A-Za-z0-9])(?:ak_[A-Za-z0-9]{24,}|sk[-_][A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{20,})')


def main():
    manifest = ROOT/'MANIFEST.sha256'
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in expected or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid manifest path')
        expected[name] = digest
    files = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*') if p.is_file()
             and not set(p.relative_to(ROOT).parts) & IGNORED and p.name not in ('MANIFEST.sha256', '.DS_Store')}
    if set(files) != set(expected):
        raise ValueError(f'Inventory mismatch: missing={sorted(set(expected)-set(files))[:5]}, extra={sorted(set(files)-set(expected))[:5]}')
    private = [b'/'+b'Users/', b'@'+b'alipay.com', b'@'+b'antgroup.com', b'chatgpt'+b'.site/']
    for name, path in files.items():
        if path.is_symlink():
            raise ValueError('Symlinks are not part of this release')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected[name]:
            raise ValueError('Hash mismatch: '+name)
        if name != 'bin/verify_release.py' and (SECRET.search(payload) or any(p in payload for p in private)):
            raise ValueError('Privacy guard: '+name)
    print(json.dumps({'valid': True, 'files': len(files), 'known_identity_or_credential_hits': 0,
                      'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
