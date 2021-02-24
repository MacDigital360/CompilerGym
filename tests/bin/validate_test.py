# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/bin:validate."""
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from absl import flags

from compiler_gym.bin.validate import main
from compiler_gym.util.capture_output import capture_output
from tests.pytest_plugins.common import skip_on_ci
from tests.test_main import main as _test_main


def test_okay_llvm_result(monkeypatch):
    stdin = """
benchmark,reward,commandline,walltime
benchmark://cBench-v0/dijkstra,0,opt  input.bc -o output.bc,0.3
""".strip()
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert "✅  cBench-v0/dijkstra " in out.stdout
    assert not out.stderr


def test_okay_llvm_result_file_input():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "test.csv"
        with open(str(path), "w") as f:
            f.write(
                """
benchmark,reward,commandline,walltime
benchmark://cBench-v0/dijkstra,0,opt  input.bc -o output.bc,0.3
""".strip()
            )
        flags.FLAGS.unparse_flags()
        flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])

        with capture_output() as out:
            main(["argv0", str(path)])

    assert "✅  cBench-v0/dijkstra " in out.stdout
    assert not out.stderr


def test_no_input(monkeypatch):
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(""))

    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert "No inputs to validate" in out.stderr


def test_invalid_reward_llvm_result(monkeypatch):
    stdin = """
benchmark,reward,commandline,walltime
benchmark://cBench-v0/dijkstra,0.5,opt  input.bc -o output.bc,0.3
""".strip()
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))
    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert (
        "❌  cBench-v0/dijkstra  Expected reward 0.5000 but received reward 0.0000\n"
        in out.stdout
    )
    assert not out.stderr


def test_invalid_csv_format(monkeypatch):
    stdin = "invalid\ncsv\nformat"
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-ic-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        with pytest.raises(SystemExit):
            main(["argv0", "-"])

    assert "Failed to parse input:" in out.stderr


def test_multiple_valid_inputs(monkeypatch):
    stdin = """
benchmark,reward,walltime,commandline
benchmark://cBench-v0/crc32,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/crc32,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/crc32,,0,opt  input.bc -o output.bc
""".strip()
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert not out.stderr
    assert out.stdout.count("✅") == 3  # Every benchmark passed.


@skip_on_ci
def test_validate_cBench_null_options(monkeypatch):
    stdin = """
benchmark,reward,walltime,commandline
benchmark://cBench-v0/gsm,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/lame,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/stringsearch,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/ghostscript,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/qsort,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/sha,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/ispell,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/blowfish,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/adpcm,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/tiffdither,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/bzip2,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/stringsearch2,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/bitcount,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/jpeg-d,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/jpeg-c,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/dijkstra,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/rijndael,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/patricia,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/tiff2rgba,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/crc32,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/tiff2bw,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/tiffmedian,,0,opt  input.bc -o output.bc
benchmark://cBench-v0/susan,,0,opt  input.bc -o output.bc
""".strip()
    flags.FLAGS.unparse_flags()
    flags.FLAGS(["argv0", "--env=llvm-v0", "--dataset=cBench-v0"])
    monkeypatch.setattr("sys.stdin", StringIO(stdin))

    with capture_output() as out:
        main(["argv0", "-"])

    assert not out.stderr
    assert out.stdout.count("✅") == 23  # Every benchmark passed.


if __name__ == "__main__":
    _test_main()
