import hashlib, json, importlib
MEASURED, ERROR, NULL = "measured", "error", "null"
SEEDS = (42, 123, 456)
def sha(obj,n=16):
 s=obj if isinstance(obj,str) else json.dumps(obj,sort_keys=True,ensure_ascii=False)
 return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]
def existing(name):
 return importlib.import_module("."+name,__package__)
