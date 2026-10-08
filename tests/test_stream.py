import json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from contextlib import contextmanager
import pytest
from serving.benchmark import run,endpoint_ok,percentile

@contextmanager
def server(protocol,usage=True,complete=True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert body['stream'] is True
            self.send_response(200);self.send_header('Content-Type','text/event-stream' if protocol=='openai' else 'application/x-ndjson');self.end_headers()
            if protocol=='openai':
                lines=['data: '+json.dumps({'choices':[{'delta':{'content':'hello world'}}]})+'\n\n']
                if usage:lines+=['data: '+json.dumps({'choices':[],'usage':{'completion_tokens':7,'prompt_tokens':9}})+'\n\n']
                if complete:lines+=['data: [DONE]\n\n']
            else:
                lines=[json.dumps({'response':'hello world','done':False})+'\n']
                if complete:lines+=[json.dumps({'done':True,'eval_count':7,'prompt_eval_count':9,'eval_duration':2000000000})+'\n']
            for line in lines:self.wfile.write(line.encode());self.wfile.flush()
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:yield 'http://127.0.0.1:'+str(http.server_port)
    finally:http.shutdown();http.server_close();thread.join()

@pytest.mark.parametrize('protocol',['openai','ollama'])
def test_tokens_not_chunks_and_parallel_requests(protocol):
    with server(protocol) as url:
        result=run(url,protocol,'protocol-fixture',['prompt']*8,4)
    assert result['successes']==8 and result['failures']==0
    assert all(r['output_tokens']==7 and r['content_chunks']==1 for r in result['rows'])
    assert result['output_tokens_per_second']==56/result['elapsed_seconds']
    assert result['ttft_p95_seconds']>=0

def test_missing_usage_is_unknown_not_fake_token_rate():
    with server('openai',usage=False) as url:result=run(url,'openai','fixture',['p'])
    assert result['successes']==1 and result['output_tokens_per_second'] is None

def test_incomplete_stream_is_failure():
    with server('openai',complete=False) as url:result=run(url,'openai','fixture',['p'])
    assert result['failures']==1 and result['successes']==0

@pytest.mark.parametrize('url',['https://api.example.com','http://user:pass@localhost','file:///tmp/model'])
def test_no_external_or_paid_endpoint(url):
    with pytest.raises(ValueError):endpoint_ok(url)

def test_tail_percentile():assert percentile([1,2,3,100],.95)==100


def test_engine_tps_uses_nanoseconds_and_differs_from_wall_throughput():
    with server('ollama') as url:
        result=run(url,'ollama','fixture',['p'],options={'num_ctx':2048})
    assert result['rows'][0]['engine_decode_tokens_per_second']==3.5
    assert result['output_tokens_per_second']==7/result['elapsed_seconds']
    assert result['end_to_end_p50_seconds']>=0

def test_sweep_writes_real_protocol_receipts(tmp_path):
    from serving.sweep import sweep
    with server('ollama') as url:
        report=sweep(url,'fixture',tmp_path,levels=(1,4),repeats=1,requests=4,max_tokens=8)
    assert report['status']=='completed' and len(report['runs'])==2
    assert all(r['successes']==4 and r['failures']==0 for r in report['runs'])
    assert json.loads((tmp_path/'c4-r0.json').read_text())['concurrency']==4
    assert report['prompt_sha256']
