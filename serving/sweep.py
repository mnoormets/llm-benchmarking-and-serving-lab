"""Warm, repeatable localhost concurrency sweep, with partial failure evidence."""
import hashlib,json,time
from pathlib import Path
from .benchmark import run,request_one

def prompts(count=100):
    return [f"Case {i:04}: Explain in detail how payment reconciliation should handle a missing invoice amount, duplicate webhook and conflicting delivery status. Use numbered steps." for i in range(count)]

def sweep(url,model,output,levels=(1,10,50,100),repeats=2,requests=100,max_tokens=32,metadata=None):
    if not levels or max(levels)>requests or not 1<=repeats<=3:
        raise ValueError("Enough requests for every concurrency; repeats 1..3")
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    workload=prompts(requests)
    (directory/"prompts.json").write_text(json.dumps(workload,indent=2))
    report={"status":"running","metadata":metadata or {},"levels":list(levels),"repeats":repeats,
            "max_output_tokens":max_tokens,"num_ctx":2048,"prompt_sha256":hashlib.sha256(json.dumps(workload).encode()).hexdigest(),
            "runs":[],"scope":"Closed-loop client concurrency; server parallelism and queueing recorded separately. No backend speedup comparison."}
    def save(): (directory/"report.json").write_text(json.dumps(report,indent=2))
    save()
    try:
        for i in range(3):request_one(url,"ollama",model,f"Warmup {i}: Describe a useful software test.",max_tokens,600,{"num_ctx":2048,"seed":73})
        for concurrency in levels:
            for repeat in range(repeats):
                result=run(url,"ollama",model,workload,concurrency,max_tokens,600,{"num_ctx":2048,"seed":73})
                result.update(repeat=repeat,concurrency=concurrency)
                (directory/f"c{concurrency}-r{repeat}.json").write_text(json.dumps(result,indent=2))
                report["runs"].append({k:v for k,v in result.items() if k!="rows"});save()
                print(json.dumps(report["runs"][-1]),flush=True)
                if result["failures"]:raise RuntimeError("Failed requests; retained partial evidence and stopped higher load")
        report["status"]="completed"
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error));save();raise
    save();return report

def plot(report,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for repeat in range(report["repeats"]):
        rows=[r for r in report["runs"] if r["repeat"]==repeat]
        axes[0].plot([r["concurrency"] for r in rows],[r["ttft_p95_seconds"] for r in rows],"o-",label=f"Repeat {repeat+1}")
        axes[1].plot([r["output_tokens_per_second"] for r in rows],[r["end_to_end_p95_seconds"] for r in rows],"o-")
        for row in rows:axes[1].annotate(f"c={row['concurrency']}",(row["output_tokens_per_second"],row["end_to_end_p95_seconds"]),fontsize=8)
    axes[0].set(xlabel="Client concurrency",ylabel="p95 TTFT (seconds)")
    axes[0].legend();axes[1].set(xlabel="Aggregate output tokens/s",ylabel="p95 end-to-end latency (seconds)")
    fig.suptitle("Warm Mistral-7B / Ollama: server queueing included")
    fig.savefig(Path(directory)/"latency-throughput.png",dpi=150);plt.close(fig)
