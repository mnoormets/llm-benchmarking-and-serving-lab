"""Bounded localhost streaming benchmark. Tokens come from server usage, never chunk counts."""
import argparse,json,time,math
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from urllib.parse import urlparse
from pathlib import Path

def endpoint_ok(url):
    p=urlparse(url)
    if p.scheme not in ('http','https') or p.hostname not in ('localhost','127.0.0.1','::1') or p.username or p.password:raise ValueError('Only local endpoints permitted; no paid APIs or embedded credentials')

def request_one(url,protocol,model,prompt,max_tokens=64,timeout=120,options=None):
    endpoint_ok(url);start=time.perf_counter();first=None;last=None;chunks=0;tokens=None;input_tokens=None;done=False;engine_eval_ns=None
    if protocol=='openai':payload={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':max_tokens,'stream':True,'stream_options':{'include_usage':True}}
    elif protocol=='ollama':payload={'model':model,'prompt':prompt,'stream':True,'options':{'temperature':0,'num_predict':max_tokens}}
    else:raise ValueError('Unknown protocol')
    if protocol=='ollama':
        payload['options'].update(options or {});payload['keep_alive']='30m'
    req=Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=timeout) as response:
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line or line.startswith(':'):continue
            if protocol=='openai':
                if not line.startswith('data:'):continue
                data=line[5:].strip()
                if data=='[DONE]':done=True;break
                obj=json.loads(data)
                if obj.get('error'):raise RuntimeError(str(obj['error']))
                usage=obj.get('usage')
                if usage:tokens=usage.get('completion_tokens');input_tokens=usage.get('prompt_tokens')
                content=''.join(c.get('delta',{}).get('content') or '' for c in obj.get('choices',[]))
            else:
                obj=json.loads(line)
                if obj.get('error'):raise RuntimeError(str(obj['error']))
                content=obj.get('response','')
                if obj.get('done'):done=True;tokens=obj.get('eval_count');input_tokens=obj.get('prompt_eval_count');engine_eval_ns=obj.get('eval_duration')
            if content:
                tick=time.perf_counter();first=first or tick;last=tick;chunks+=1
    if not done:raise RuntimeError('Truncated stream: no completion marker')
    if tokens is not None and (type(tokens) is not int or tokens<0):raise ValueError('Invalid server token count')
    return {'success':True,'engine_decode_tokens_per_second':tokens/(engine_eval_ns/1e9) if tokens is not None and isinstance(engine_eval_ns,(int,float)) and engine_eval_ns>0 else None,'ttft_seconds':None if first is None else first-start,'end_to_end_seconds':time.perf_counter()-start,'output_tokens':tokens,'input_tokens':input_tokens,'content_chunks':chunks,'stream_span_seconds':None if first is None else last-first,'ttft_definition':'first nonempty content chunk, not first byte; a chunk may contain multiple tokens'}

def percentile(values,p):
    values=sorted(values)
    if not values:return None
    return values[max(0,math.ceil(p*len(values))-1)]

def run(url,protocol,model,prompts,concurrency=1,max_tokens=64,timeout=120,options=None):
    endpoint_ok(url)
    if not prompts or not 1<=concurrency<=100 or len(prompts)>1000 or not 1<=max_tokens<=2048:raise ValueError('Invalid bounded load')
    def one(prompt):
        try:return request_one(url,protocol,model,prompt,max_tokens,timeout,options)
        except Exception as e:return {'success':False,'error_type':type(e).__name__,'error':str(e)}
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(one,prompts))
    duration=time.perf_counter()-start;good=[r for r in rows if r['success']];ttft=[r['ttft_seconds'] for r in good if r['ttft_seconds'] is not None]
    tokens_complete=bool(good) and all(r['output_tokens'] is not None for r in good)
    return {'protocol':protocol,'endpoint':url,'model':model,'requests':len(prompts),'concurrency':concurrency,'elapsed_seconds':duration,'successes':len(good),'failures':len(rows)-len(good),'successful_requests_per_second':len(good)/duration,'output_tokens_per_second':sum(r['output_tokens'] for r in good)/duration if tokens_complete else None,'token_count_source':'server usage/eval_count, not streaming chunks','ttft_p50_seconds':percentile(ttft,.5),'ttft_p95_seconds':percentile(ttft,.95),'end_to_end_p50_seconds':percentile([r['end_to_end_seconds'] for r in good],.5),'end_to_end_p95_seconds':percentile([r['end_to_end_seconds'] for r in good],.95),'rows':rows,'load_model':'closed-loop fixed thread concurrency; queue waits before worker start excluded from per-request TTFT, included in wall throughput'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--protocol',choices=['openai','ollama'],required=True);p.add_argument('--model',required=True);p.add_argument('--requests',type=int,default=20);p.add_argument('--concurrency',type=int,default=1);p.add_argument('--max-tokens',type=int,default=64);p.add_argument('--output',default='runs/benchmark.json');p.add_argument('--label',required=True);a=p.parse_args()
    if not 1<=a.requests<=1000:p.error('Requests must be 1..1000')
    prompts=['Explain why a missing invoice field must remain null.']*a.requests
    result=run(a.url,a.protocol,a.model,prompts,a.concurrency,a.max_tokens);result['experiment_label']=a.label
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
