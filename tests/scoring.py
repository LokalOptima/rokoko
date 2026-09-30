"""Literal word scoring: case/punctuation only; numbers/negation remain significant."""
import random
import re

def words(text): return re.findall(r"\w+(?:['’]\w+)*",text.lower().replace('’',"'"))
def distance(ref,hyp):
    prev=list(range(len(hyp)+1))
    for i,a in enumerate(ref,1):
        cur=[i]
        for j,b in enumerate(hyp,1):cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(a!=b)))
        prev=cur
    return prev[-1]
def score(reference,hypothesis):
    ref=words(reference);assert ref
    return dict(errors=distance(ref,words(hypothesis)),words=len(ref))
def paired_interval(rows_a,rows_b,seed=9401,draws=5000):
    assert len(rows_a)==len(rows_b) and rows_a
    assert all(a['words']==b['words'] and a.get('id')==b.get('id') for a,b in zip(rows_a,rows_b)), 'unmatched bootstrap pairs'
    rng=random.Random(seed);values=[];n=len(rows_a)
    for _ in range(draws):
        chosen=[rng.randrange(n) for _ in range(n)];den=sum(rows_a[i]['words'] for i in chosen)
        values.append(sum(rows_a[i]['errors']-rows_b[i]['errors'] for i in chosen)/den)
    values.sort();return [values[int(.025*draws)],values[int(.975*draws)]]
