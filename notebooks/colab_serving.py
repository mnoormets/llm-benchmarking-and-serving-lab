"""Free Colab T4 runner. Execute in a fresh runtime; no paid APIs or tunnels."""
import hashlib,json,os,platform,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path("/content/llm-benchmarking-and-serving-lab")
if not Path("/content").exists():raise RuntimeError("Run in Colab")
gpus=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,memory.total,driver_version","--format=csv,noheader"],text=True).strip()
if not gpus:raise RuntimeError("Select a free GPU first")
free_mib=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip().splitlines()[0])
if free_mib<8192:raise RuntimeError("Less than 8 GiB free GPU memory. Restart the runtime to release the previous QLoRA model before serving.")
if not ROOT.exists():subprocess.run(["git","clone","https://github.com/mnoormets/llm-benchmarking-and-serving-lab.git",str(ROOT)],check=True)
else:subprocess.run(["git","-C",str(ROOT),"pull","--ff-only"],check=True)
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
subprocess.run([sys.executable,"-m","pip","install","-q","matplotlib","zstandard"],check=True)
output=ROOT/"runs"/time.strftime("colab-serving-%Y%m%d-%H%M%S");output.mkdir(parents=True)
from serving.ollama_install import install
try:
    ollama=install(ROOT/"runs/ollama-runtime",output)
except Exception:
    archive=shutil.make_archive("/content/serving-install-failure","zip",root_dir=output)
    from google.colab import files
    files.download(archive)
    raise
from serving.benchmark import request_one
from serving.sweep import sweep,plot
URL="http://127.0.0.1:11434";MODEL="mistral:7b-instruct-v0.3-q4_K_M"
def api(path,payload=None,timeout=60):
    req=urllib.request.Request(URL+path,data=None if payload is None else json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
# Do not hijack an existing runtime's server/settings.
try:api("/api/version")
except Exception:pass
else:raise RuntimeError("An Ollama server is already using this port. Restart Colab runtime before this experiment.")
env=os.environ.copy();env.update(OLLAMA_HOST="127.0.0.1:11434",OLLAMA_NUM_PARALLEL="4",OLLAMA_MAX_QUEUE="512",OLLAMA_MAX_LOADED_MODELS="1",OLLAMA_CONTEXT_LENGTH="2048",OLLAMA_NO_CLOUD="1",OLLAMA_MODELS=str(ROOT/"runs/ollama-models"))
log=(output/"server.log").open("w")
process=subprocess.Popen([ollama,"serve"],env=env,stdout=log,stderr=subprocess.STDOUT)
try:
    for _ in range(120):
        if process.poll() is not None:raise RuntimeError("Server exited; inspect server.log")
        try:version=api("/api/version");break
        except Exception:time.sleep(1)
    else:raise RuntimeError("Server startup timeout")
    with (output/"pull.log").open("w") as pull_log:subprocess.run([ollama,"pull",MODEL],env=env,stdout=pull_log,stderr=subprocess.STDOUT,check=True)
    cold=request_one(URL+"/api/generate","ollama",MODEL,"Cold-start measurement: explain a software test.",32,600,{"num_ctx":2048,"seed":73})
    (output/"cold-start.json").write_text(json.dumps(cold,indent=2))
    placement=api("/api/ps")
    if not placement.get("models") or not all(m.get("size_vram",0)>0 for m in placement["models"]):raise RuntimeError("No verified GPU model placement; refusing CPU-as-GPU benchmark")
    metadata={"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"hardware":gpus,"python":platform.python_version(),"ollama":version,"model_tags":api("/api/tags"),"model_show":api("/api/show",{"model":MODEL}),"model_placement":placement,"server_parallel_limit":4,"server_max_queue":512,"gpu_memory_before":subprocess.check_output(["nvidia-smi"],text=True),"quantization":"Q4_K_M; no AWQ conversion performed","installation":json.loads((output/"install-receipt.json").read_text())}
    report=sweep(URL+"/api/generate",MODEL,output,metadata=metadata)
    plot(report,output)
finally:
    process.terminate()
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:process.kill();process.wait()
    log.close()
    archive=shutil.make_archive("/content/serving-experiment","zip",root_dir=output)
    from google.colab import files
    files.download(archive)
