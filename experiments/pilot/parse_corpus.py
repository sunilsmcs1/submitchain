import json, glob, re, sys, os
files = sorted(glob.glob(sys.argv[1] + '/mcp-SciSpace-search-*.txt'), key=os.path.getmtime)
topics = ['intrusion','blockchain','federated','recsys','gnn','nlp','medseg','edge-rl']
# map by modification order is unreliable; infer topic by query-free keyword vote instead
kw = {'intrusion':['intrusion','attack','anomal'],'blockchain':['blockchain','ledger','smart contract'],
 'federated':['federated'],'recsys':['recommend'],'gnn':['graph'],'nlp':['sentiment','text classification','language model','bert'],
 'medseg':['segment','lesion','medical'],'edge-rl':['reinforcement','edge','offload']}
seen=set(); out=[]
for fi,f in enumerate(files):
    for r in json.load(open(f))['results']:
        t = r['text']; m = re.search(r'Abstract:\s*(.*)', t, re.S)
        if not m: continue
        ab = m.group(1).replace('…','').strip()
        ab = re.sub(r'^(Background and objectives:|Abstract[:.]?)\s*','',ab)
        head = t.split('\n')[0]; parts=[p.strip() for p in head.split('·')]
        year = parts[0] if parts and parts[0].isdigit() else ''
        authors = [a.strip() for a in parts[1].split(',')] if len(parts)>1 else []
        key = re.sub(r'\W+','',r['title'].lower())
        if key in seen or len(ab.split())<50: continue
        seen.add(key)
        out.append(dict(id=f'p{len(out):04d}', file=fi, title=r['title'].strip(), abstract=ab, year=year, authors=authors, url=r['url'].split('?')[0]))
print(len(out))
json.dump(out, open('data/corpus.json','w'), indent=0)
import collections; print(collections.Counter(o['file'] for o in out))
print(sum(len(o['abstract'].split()) for o in out)/len(out))
