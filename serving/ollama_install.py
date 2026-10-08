"""Install the official Ollama Linux archive under a local runtime prefix."""
import hashlib,json,platform,stat,tarfile,time,urllib.request
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit,urlunsplit
URL="https://ollama.com/download/ollama-linux-amd64.tar.zst"

def public_download_url(url):
    """Remove temporary signed query tokens from public receipts."""
    parts=urlsplit(url)
    return urlunsplit((parts.scheme,parts.netloc,parts.path,"",""))

def extract_package(archive,destination):
    import zstandard
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    count=total=0
    with Path(archive).open("rb") as source,zstandard.ZstdDecompressor().stream_reader(source) as decoded:
        with tarfile.open(fileobj=decoded,mode="r|") as tar:
            for member in tar:
                name=PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                    raise ValueError("Unsafe archive member")
                if member.isdev() or member.isfifo():raise ValueError("Special archive member")
                count+=1;total+=member.size
                if count>100000 or total>10*1024**3:raise ValueError("Unexpected package size")
                tar.extract(member,path=destination,filter="data")
    executable=destination/"bin/ollama"
    if not executable.is_file() or executable.is_symlink():raise ValueError("Package lacks a regular bin/ollama executable")
    executable.chmod(executable.stat().st_mode|stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)
    return {"members":count,"uncompressed_member_bytes":total}

def install(destination,log_directory):
    if platform.system()!="Linux" or platform.machine().lower() not in ("x86_64","amd64"):
        raise RuntimeError("Linux x86_64 required for Colab package")
    destination=Path(destination);logs=Path(log_directory);logs.mkdir(parents=True,exist_ok=True)
    receipt=destination/"install-receipt.json"
    executable=destination/"bin/ollama"
    if receipt.exists() and executable.is_file():
        (logs/"install-receipt.json").write_bytes(receipt.read_bytes())
        print("Reusing previously extracted local Ollama package",flush=True)
        return str(executable)
    archive=destination.parent/"ollama-linux-amd64.tar.zst"
    archive.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();digest=hashlib.sha256();downloaded=0
    with (logs/"install.log").open("w") as log:
        try:
            log.write("Official package: "+URL+"\n");log.flush()
            print("Downloading official Ollama package; this can take several minutes...",flush=True)
            req=urllib.request.Request(URL,headers={"User-Agent":"llm-serving-colab-lab/1.0"})
            with urllib.request.urlopen(req,timeout=120) as response,archive.open("wb") as out:
                final_url=public_download_url(response.geturl())
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    downloaded+=len(chunk)
                    if downloaded>5*1024**3:raise ValueError("Unexpected download size")
                    out.write(chunk);digest.update(chunk)
            print("Extracting package without systemd or sudo...",flush=True)
            details=extract_package(archive,destination)
            record={"source_url":URL,"resolved_download_url":final_url,"archive_sha256":digest.hexdigest(),"download_bytes":downloaded,"install_seconds":time.perf_counter()-started,"method":"local official tar.zst archive, Python zstandard; no shell installer/system service",**details}
            receipt.write_text(json.dumps(record,indent=2))
            (logs/"install-receipt.json").write_bytes(receipt.read_bytes())
            log.write(json.dumps(record,indent=2)+"\n")
        except Exception as error:
            message=f"{type(error).__name__}: {error}"
            log.write(message+"\n");print("Installation failed:",message,flush=True);raise
    return str(executable)
