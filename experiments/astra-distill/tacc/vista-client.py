"""Query a vLLM server: one guided-JSON completion and one fixed-length unguided decode for tokens/s. stdlib only.
  python vista-client.py URL MODELNAME"""
import json, sys, time, urllib.request
url, name = sys.argv[1], sys.argv[2]
def post(body):
    t = time.time(); r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900)); return d, time.time() - t
schema = {"type": "object", "properties": {"space": {"enum": ["MNI", "TAL"]}, "analyses": {"type": "array", "items": {
    "type": "object", "properties": {"name": {"type": "string"}, "points": {"type": "array", "items": {"type": "array",
    "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}, "required": ["name", "points"]}}},
    "required": ["space", "analyses"]}
doc = ("Table 2. Peak activations (MNI) for faces > houses. Left fusiform gyrus -42 -54 -18 T=6.31; Right inferior occipital "
       "gyrus 38 -82 -10 T=5.77; Right amygdala 24 -4 -20 T=4.10.")
msg = [{"role": "user", "content": f"Extract the coordinate space and the analyses with [x,y,z,T] points as JSON.\n\n{doc}"}]
base = {"model": name, "messages": msg, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
d, dt = post({**base, "max_tokens": 512, "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}})
txt = d["choices"][0]["message"]["content"]; u = d["usage"]
try: json.loads(txt); valid = True
except Exception: valid = False
print("GUIDED_JSON valid=%s completion_tokens=%d wall=%.2fs (%.1f tok/s incl. prefill)" % (valid, u["completion_tokens"], dt, u["completion_tokens"] / dt))
print("GUIDED_OUT", txt[:600])
d, dt = post({**base, "max_tokens": 384, "ignore_eos": True, "messages": [{"role": "user", "content": "Write a long essay about the history of neuroimaging."}]})
u = d["usage"]; print("DECODE completion_tokens=%d wall=%.2fs -> %.1f tok/s (single stream)" % (u["completion_tokens"], dt, u["completion_tokens"] / dt))
