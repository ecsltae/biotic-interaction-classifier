import sqlite3, re, functools
DB='/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite'
_c=sqlite3.connect(DB, check_same_thread=False)
_c.execute('PRAGMA query_only=1')
def norm(s):
    s=str(s).lower().strip()
    s=re.sub(r'\(.*?\)',' ',s)
    s=re.sub(r'[^a-z0-9 .]',' ',s); return re.sub(r'\s+',' ',s).strip()
@functools.lru_cache(maxsize=400000)
def resolve(name):
    n=norm(name)
    if not n: return None
    for cand in [n, ' '.join(n.split()[:2]), n.split()[0] if n.split() else '']:
        if not cand: continue
        r=_c.execute('select ott_id,is_syn from names where name_norm=? order by is_syn limit 1',(cand,)).fetchone()
        if r: return r[0]
        if not cand.endswith('s'):
            r=_c.execute('select ott_id,is_syn from names where name_norm=? order by is_syn limit 1',(cand+'s',)).fetchone()
            if r: return r[0]
        if cand.endswith('s'):
            r=_c.execute('select ott_id,is_syn from names where name_norm=? order by is_syn limit 1',(cand[:-1],)).fetchone()
            if r: return r[0]
    return None
@functools.lru_cache(maxsize=400000)
def lineage(ott):
    out=[]; cur=str(ott); seen=set()
    while cur and cur not in seen and len(out)<60:
        seen.add(cur)
        r=_c.execute('select pref,parent from concepts where ott_id=?',(cur,)).fetchone()
        if not r: break
        out.append(r[0]); cur=r[1]
    return tuple(out)
