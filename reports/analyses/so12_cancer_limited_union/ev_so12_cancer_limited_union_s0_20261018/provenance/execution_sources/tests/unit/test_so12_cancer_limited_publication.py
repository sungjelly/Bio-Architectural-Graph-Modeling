"""Immutable publication must detect modified or omitted scientific payloads."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
from run_so12_cancer_limited_union import record, verify_inventory, write


def bundle(tmp_path):
    (tmp_path / 'scores.csv').write_text('tp,fp,fn\n10,3,2\n')
    write(tmp_path / 'manifest.json', {'files': {'scores.csv': record(tmp_path / 'scores.csv')}})
    write(tmp_path / '_SUCCESS', {'manifest_sha256': record(tmp_path / 'manifest.json')['sha256']})
    return tmp_path


def test_inventory_detects_modified_scores(tmp_path):
    root = bundle(tmp_path)
    assert len(verify_inventory(root)['files']) == 1
    (root / 'scores.csv').write_text('tp,fp,fn\n10,0,0\n')
    with pytest.raises(RuntimeError, match='Immutable source changed'):
        verify_inventory(root)


def test_inventory_detects_uninventoried_output_and_missing_file(tmp_path):
    root = bundle(tmp_path)
    extra = root / 'unreviewed.csv'
    extra.write_text('1\n')
    with pytest.raises(RuntimeError, match='Incomplete immutable inventory'):
        verify_inventory(root)
    extra.unlink()
    (root / 'scores.csv').unlink()
    with pytest.raises(RuntimeError, match='Incomplete immutable inventory'):
        verify_inventory(root)


def test_parent_identity_and_symlink_guard(tmp_path):
    root = bundle(tmp_path)
    with pytest.raises(RuntimeError, match='Source manifest identity changed'):
        verify_inventory(root, '0' * 64)
    (root / 'alias.csv').symlink_to(root / 'scores.csv')
    with pytest.raises(RuntimeError, match='Symlinks'):
        verify_inventory(root)
