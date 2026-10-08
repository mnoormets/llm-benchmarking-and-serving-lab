import io,tarfile
from pathlib import Path
import pytest,zstandard
from serving.ollama_install import extract_package

def package(path,entries):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode="w") as tar:
        for name,contents in entries:
            member=tarfile.TarInfo(name);member.size=len(contents)
            tar.addfile(member,io.BytesIO(contents))
    path.write_bytes(zstandard.ZstdCompressor().compress(data.getvalue()))

def test_extracts_executable_and_relative_libraries(tmp_path):
    archive=tmp_path/"package.zst"
    package(archive,[("bin/ollama",b"test fixture, not executable code"),("lib/ollama/libtest.so",b"fixture")])
    out=tmp_path/"runtime";report=extract_package(archive,out)
    assert report["members"]==2 and (out/"lib/ollama/libtest.so").read_bytes()==b"fixture"
    assert (out/"bin/ollama").is_file()

@pytest.mark.parametrize("name",["../escape","/outside","lib/../escape"])
def test_rejects_archive_path_escape(tmp_path,name):
    archive=tmp_path/"bad.zst";package(archive,[(name,b"x")])
    with pytest.raises(ValueError):extract_package(archive,tmp_path/"runtime")

def test_missing_executable_rejected(tmp_path):
    archive=tmp_path/"bad.zst";package(archive,[("lib/ollama/library",b"fixture")])
    with pytest.raises(ValueError,match="executable"):extract_package(archive,tmp_path/"runtime")
