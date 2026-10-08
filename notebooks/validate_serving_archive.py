"""Validate exported benchmark receipts without executing archive code."""
import argparse,hashlib,json,math,re,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serving.benchmark import percentile
from serving.ollama_install import public_download_url

def sanitized(value):
    if isinstance(value,dict):
        return {k:public_download_url(v) if k=='resolved_download_url' else sanitized(v) for k,v in value.items()}
    if isinstance(value,list):return [sanitized(v) for v in value]
    return value

def close(actual,expected):
    assert math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9),(actual,expected)

def validate(archive,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())<30 and all(i.file_size<25*1024**2 for i in z.infolist())
        def read(name):return json.loads(z.read(name))
        report=read('report.json');prompts=read('prompts.json')
        assert report['status']=='completed' and report['levels']==[1,10,50,100] and report['repeats']==2
        assert len(prompts)==100 and hashlib.sha256(json.dumps(prompts).encode()).hexdigest()==report['prompt_sha256']
        assert report['metadata']['server_parallel_limit']==4
        placement=report['metadata']['model_placement']['models'][0]
        assert placement['size_vram']==placement['size']>0
        assert placement['details']['quantization_level']=='Q4_K_M'
        total=0;metrics=[];exports=['report.json','prompts.json','cold-start.json','install-receipt.json']
        for summary in report['runs']:
            name=f"c{summary['concurrency']}-r{summary['repeat']}.json";raw=read(name);exports.append(name)
            for key,value in summary.items():assert raw[key]==value
            rows=raw['rows'];assert len(rows)==raw['requests']==100
            assert raw['model']==placement['name'] and raw['protocol']=='ollama'
            good=[r for r in rows if r['success']];assert len(good)==raw['successes']==100 and raw['failures']==0
            for row in good:
                assert 0<=row['ttft_seconds']<=row['end_to_end_seconds']
                assert 0<=row['stream_span_seconds']<=row['end_to_end_seconds']
                assert row['output_tokens']==report['max_output_tokens']==32
                for k in ['ttft_seconds','end_to_end_seconds','stream_span_seconds']:assert math.isfinite(row[k])
            close(raw['output_tokens_per_second'],sum(r['output_tokens'] for r in good)/raw['elapsed_seconds'])
            close(raw['successful_requests_per_second'],len(good)/raw['elapsed_seconds'])
            for field,key in [('ttft','ttft_seconds'),('end_to_end','end_to_end_seconds')]:
                for suffix,p in [('p50',.5),('p95',.95)]:close(raw[f'{field}_{suffix}_seconds'],percentile([r[key] for r in good],p))
            total+=len(good);metrics.append(summary)
        assert len(metrics)==8 and total==800
        server=z.read('server.log').decode('utf-8',errors='replace')
        statuses=re.findall(r'\|\s*(\d{3})\s*\|[^\n]*POST\s+"/api/generate"',server)
        assert len(statuses)==804 and set(statuses)=={'200'}
        lines=[line for line in server.splitlines() if any(x in line for x in ['offloaded 33/33','CUDA0 model buffer','CUDA0 KV buffer','OLLAMA_NUM_PARALLEL:4'])]
        assert any('offloaded 33/33' in line for line in lines)
        (out/'gpu-server-evidence.txt').write_text('\n'.join(lines)+'\n804 /api/generate HTTP 200 responses (800 measured + 3 warmups + 1 cold).\n',encoding='utf-8')
        for name in exports:(out/name).write_text(json.dumps(sanitized(read(name)),indent=2),encoding='utf-8')
        (out/'latency-throughput.png').write_bytes(z.read('latency-throughput.png'))
        result={'verified':True,'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),'source_commit':report['metadata']['source_commit'],'warm_successes':total,'warm_failures':0,'server_http_200':804,'summary_metrics_recomputed':True,'model_gpu_offload_confirmed':True,'local_gpu_replay':False,'redaction':'Only temporary resolved download URL query/fragment removed; measurements unchanged.','limitations':['Exported timings are not independently replayed wall clocks.','Raw engine eval durations and generated text were not exported.','Warm fixed prompt reuse permits cache effects; 32-token responses only.','Client concurrency up to 100; server decode slots fixed at four.','Two repetitions, one GPU/backend; no vLLM or TensorRT comparison.'],'runs':metrics,'cold_start':read('cold-start.json')}
        (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ['runs','cold_start']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',default='results/colab-t4');a=p.parse_args();validate(a.archive,a.output)
