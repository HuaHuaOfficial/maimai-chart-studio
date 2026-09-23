"""Optional, bounded, structure-preserving full-chart WHERE selector.

The returned events have NO publication authority; service re-evaluates their
immutable payload with the production Harness before obtaining a new permit.
"""
from __future__ import annotations
import json,time,re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..io.factors import factor_event,factor_note,parse_slide_tracks
from ..io.durations import hold_seconds,slide_seconds
from ..io.symmetry import transform_event_group,canonical_event_window,semantic_notes
from ..io.timing import ticks_to_seconds
from ..io.codec import Codec

BEAM=2
RADIUS=8
WIDTH=17
W=17
R=8
WHAT=26
def keys(text):
    out=[]
    for raw in semantic_notes(text):
        n=factor_note(raw)
        if n.family=='touch':raise ValueError('touch outside scope')
        # Keep every route segment and '*' binding, omit timing/modifiers only.
        geometry=re.sub(r'\[[^]]*\]','',raw)
        geometry=re.sub(r'[bxf$?!@]','',geometry)
        out.append(n.family+':'+geometry)
    return out

def encode_base(text,bpm):
    notes=factor_event(text)['notes']
    w=np.zeros(WHAT,np.float32);g=np.zeros(16,np.float32)
    w[0]=len(notes)/4
    supported=True
    for n in notes:
        family=n['family'];w[1+('tap','hold','slide','touch').index(family)]+=1/4
        for j,k in enumerate(('is_break','is_ex','is_star','is_firework','is_headless')):w[5+j]+=int(n[k])/4
        if n['duration'] and family in ('hold','touch'):w[10]+=min(8,hold_seconds(n['duration'],bpm))/8
        tracks=parse_slide_tracks(n['raw']) if family=='slide' else []
        for tr in tracks:
            wait,movement=slide_seconds(tr['duration'],bpm)
            w[11]+=min(8,wait)/8;w[12]+=min(8,movement)/8
            # Route family retained, ALL numeric route geometry excluded.
            for j,s in enumerate(('-','<','>','^','v','p','q','s','z','V','w','pp','qq')):
                w[13+j]+=int(s in n['shapes'])/4
        if not n['start'].isdigit():supported=False
        else:g[int(n['start'])-1]=1
        for tr in tracks:
            import re
            ends=re.findall('[1-8]',tr['route'])
            if ends:g[8+int(ends[-1])-1]=1
    return w,g,supported


def load_assets(root,device):
    vocab_path=Path(root)/'models/experimental/where_structure_vocab.json'
    checkpoint=Path(root)/'models/experimental/where_structure_causal.pt'
    vocab=json.loads(vocab_path.read_text(encoding='utf8'))['keys']
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    if state['n']!=len(vocab) or state['mode']!='causal':
        raise RuntimeError('WHERE structure asset mismatch')
    model=Model(len(vocab),'causal').to(device).eval()
    model.load_state_dict(state['model'])
    return model,{key:i for i,key in enumerate(vocab)}

class Model(nn.Module):
    def __init__(self,n,mode):
        super().__init__();self.n=n;self.mode=mode
        self.proj=nn.Linear(n+31,96);self.pos=nn.Parameter(torch.randn(1,W,96)*.02)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(96,4,192,.1,batch_first=True),2,enable_nested_tensor=False)
        self.out=nn.Linear(96,n)
    def forward(self,x):
        x=x.clone()
        if self.mode=='local':x[:,:R,26:26+self.n]=0;x[:,R+1:,26:26+self.n]=0
        if self.mode=='causal':x=x[:,:R+1]
        z=self.encoder(self.proj(x)+self.pos[:,:x.shape[1]])
        return self.out(z[:,R])

def objective(logits,y):
    return -(y/y.sum(1,keepdim=True)*logits.log_softmax(-1)).sum(1)

@torch.no_grad()
def evaluate(m,x,y):
    m.eval();p=torch.cat([m(b) for b in x.split(192)])
    return {'cross_entropy':float(objective(p,y).mean()),'top1_in_target':float((y.gather(1,p.argmax(1)[:,None])>0).float().mean())}


def structural_signature(events,bt,bv):
    # Freeze timing, family, modifiers, and every per-track duration.
    from ..io.symmetry import semantic_notes
    rows=[]
    for tick,text in sorted(events.items()):
        notes=[]
        for raw in semantic_notes(text):
            n=factor_event(raw)['notes'][0]
            durations=tuple(t['duration'] for t in parse_slide_tracks(raw)) if n['family']=='slide' else (n['duration'],)
            notes.append((n['family'],n['start'] if n['family']=='touch' else None,*(bool(n[k]) for k in ('is_break','is_ex','is_star','is_firework','is_headless')),durations))
        rows.append((int(tick),tuple(sorted(notes,key=repr))))
    return repr((rows,list(map(int,bt)),list(map(float,bv))))

def select(events,bt,bv,*,root,codec,harness,slot,ds,bpm,version):
    started=time.perf_counter()
    model,index=load_assets(root,codec.device)
    n=len(index)
    N=n
    kernel=harness.kernel
    def encode_bound(text,bpm):
        geometry=np.zeros(n,np.float32)
        try:
            w,_,ok=encode_base(text,bpm);names=keys(text)
            ok=ok and all(name in index for name in names)
            if ok:
                for name in names:geometry[index[name]]+=1
            return w,geometry,ok
        except (ValueError,KeyError,TypeError):
            return np.zeros(WHAT,np.float32),geometry,False
    events=dict(events);bt=np.asarray(bt);bv=np.asarray(bv)
    ticks=np.asarray(sorted(events),np.int64);times=ticks_to_seconds(ticks,bt,bv)
    local_bpm=np.asarray([float(bv[max(0,np.searchsorted(bt,t,side='right')-1)]) for t in ticks])
    rows=[];valid=[];ends=[]
    for tick,at,b in zip(ticks,times,local_bpm):
        text=events[int(tick)];w,g,ok=encode_bound(text,b);rows.append(np.r_[w,g,0]);valid.append(ok)
        end=float(at)
        for note in codec.parse_event(text):
            if note['family']=='slide':
                for track in parse_slide_tracks(note['raw']):end=max(end,float(at)+sum(slide_seconds(track['duration'],b)))
            elif note['duration']:end=max(end,float(at)+hold_seconds(note['duration'],b))
        ends.append(end)
    rows=np.asarray(rows,np.float32);valid=np.asarray(valid,bool);ends=np.asarray(ends)
    max_bar=int(ticks[-1]//384)+1;segments=[];bar=0
    while bar<max_bar:
        end=min(bar+2,max_bar)
        while end<max_bar:
            boundary=float(ticks_to_seconds(np.asarray([end*384]),bt,bv)[0])
            if not np.any((times<boundary-1e-9)&(ends>=boundary-1e-9)):break
            end=min(end+2,max_bar)
        idx=np.flatnonzero((ticks>=bar*384)&(ticks<end*384))
        if len(idx):segments.append({'bars':[bar+1,end],'idx':idx,'merged_for_active_duration':end>bar+2})
        bar=end
    signature=structural_signature(events,bt,bv)
    segment_candidates=[]
    for segment in segments:
        idx=segment['idx'];canonical=canonical_event_window([(int(ticks[i]),events[int(ticks[i])]) for i in idx])
        candidates=[];seen=set();supported=bool(valid[idx].all())
        rotations=range(8) if supported else (0,)
        for rotation in rotations:
            texts=tuple(transform_event_group(events[int(ticks[i])],rotation) for i in idx)
            if texts in seen:continue
            seen.add(texts);candidate_rows=[];ok=True
            for i,text in zip(idx,texts):
                w,g,valid_row=encode_bound(text,local_bpm[i]);ok&=valid_row;candidate_rows.append(np.r_[w,g,0])
            if rotation and not ok:continue
            assert canonical_event_window([(int(ticks[i]),text) for i,text in zip(idx,texts)])==canonical
            candidates.append({'rotation':rotation,'texts':texts,'rows':np.asarray(candidate_rows,np.float32),'valid':ok})
        segment_candidates.append(candidates)
        segment['candidate_count']=len(candidates);segment['model_supported']=supported
        locked=None
    # Each state keeps the last eight committed events. All alternatives for
    # one segment are evaluated in batches, event by event, with no future input.
    states=[{'loss':0.0,'count':0,'history':[],'path':[]}]
    for si,(segment,candidates) in enumerate(zip(segments,segment_candidates)):
        idx=segment['idx'];extensions=[]
        for state in states:
            allowed=[(ci,candidate) for ci,candidate in enumerate(candidates) if 'human_locked_rotation' not in segment or candidate['rotation']==segment['human_locked_rotation']]
            for ci,candidate in allowed:
                extensions.append({'loss':state['loss'],'count':state['count'],'history':list(state['history']),'path':state['path']+[ci],'candidate':candidate})
        for position,i in enumerate(idx):
            batch=[];targets=[];owners=[]
            for ei,ext in enumerate(extensions):
                row=ext['candidate']['rows'][position];history=ext['history']
                if ext['candidate']['valid'] and len(history)>=RADIUS and all(item[2] for item in history[-RADIUS:]):
                    context=[item[0] for item in history[-RADIUS:]]+[row.copy()]
                    target=context[-1][WHAT:WHAT+N].copy();context[-1][WHAT:WHAT+N]=0;context[-1][-1]=1
                    rel=[np.clip(item[1]-times[i],-16,16)/16 for item in history[-RADIUS:]]+[0.]
                    x=np.c_[np.asarray(context),np.asarray(rel),np.tile([ds/15,(slot-4)/2,version/30],(RADIUS+1,1))]
                    batch.append(x);targets.append(target);owners.append(ei)
            if batch:
                with torch.no_grad():
                    x=torch.tensor(np.asarray(batch,np.float32),device='cuda');y=torch.tensor(np.asarray(targets,np.float32),device='cuda')
                    values=objective(model(x),y).cpu().numpy()
                for owner,value in zip(owners,values):extensions[owner]['loss']+=float(value);extensions[owner]['count']+=1
            for ext in extensions:
                ext['history'].append((ext['candidate']['rows'][position],float(times[i]),bool(ext['candidate']['valid'])))
                ext['history']=ext['history'][-RADIUS:]
        # An unsupported segment is frozen and resets learned causal memory;
        # carrying zero geometry through it would fabricate context.
        if not segment['model_supported']:
            for ext in extensions:ext['history']=[]
            segment['causal_reset_after']=True
        extensions.sort(key=lambda x:(x['loss']/max(1,x['count']),x['loss'],tuple(x['path'])))
        states=[{k:v for k,v in ext.items() if k!='candidate'} for ext in extensions[:BEAM]]
        segment['beam_survivors']=len(states)
    # Repair the best causal path by reverting whole structure units. This
    # keeps causal ranking while removing jointly invalid combinations.
    cal=harness.calibration
    identity_path=[next(i for i,option in enumerate(options) if option['rotation']==0) for options in segment_candidates]
    def materialize(path):
        candidate=dict(events)
        for segment,options,ci in zip(segments,segment_candidates,path):
            for i,text in zip(segment['idx'],options[ci]['texts']):candidate[int(ticks[i])]=text
        assert structural_signature(candidate,bt,bv)==signature
        return candidate
    def clean_many(charts):
        payloads=[codec.encode(chart,bt,bv) for chart in charts]
        ends=[float(payload.columns['event_time'].max())+10 for payload in payloads]
        result=kernel.evaluate(payloads,versions=[version]*len(payloads),end_seconds=ends,bpms=[bpm]*len(payloads),thresholds=[cal['thresholds'] if cal else None]*len(payloads),tolerances=cal['tolerance'] if cal else None,features=True)
        hard,quality,_=result.summaries();bad=torch.zeros_like(result.event_batch,dtype=torch.bool)
        for mask in result.hard.values():bad|=mask
        bad|=result.quality.any(1)
        return [{'hard':int(hard[i].sum()),'quality':int(quality[i].sum()),'bad_ticks':sorted(set(map(int,result.event_tick[bad&(result.event_batch==i)].cpu().tolist())))} for i in range(len(payloads))]
    path=list(states[0]['path']);refined=materialize(path);verdict=clean_many([refined])[0];harness_checked=1;repair_steps=[]
    while verdict['hard'] or verdict['quality']:
        choices=[]
        bad_ticks=verdict['bad_ticks'];nearby=set()
        for si,segment in enumerate(segments):
            lo=int(ticks[segment['idx'][0]]);hi=int(ticks[segment['idx'][-1]])
            if any(lo-384<=tick<=hi+384 for tick in bad_ticks):nearby.add(si)
        for si,(current,identity) in enumerate(zip(path,identity_path)):
            if 'human_locked_rotation' in segments[si]:continue
            if current==identity:continue
            if nearby and si not in nearby:continue
            trial=list(path);trial[si]=identity;choices.append((si,trial))
        if not choices:
            for si,(current,identity) in enumerate(zip(path,identity_path)):
                if 'human_locked_rotation' not in segments[si] and current!=identity:
                    trial=list(path);trial[si]=identity;choices.append((si,trial))
        if not choices:raise RuntimeError(name+' identity path unexpectedly failed Harness')
        charts=[materialize(trial) for _,trial in choices];results=clean_many(charts);harness_checked+=len(charts)
        best=min(range(len(choices)),key=lambda i:(results[i]['hard']+results[i]['quality'],results[i]['hard'],results[i]['quality'],-choices[i][0]))
        before_verdict=verdict;si,path=choices[best];verdict=results[best];refined=charts[best]
        repair_steps.append({'reverted_segment':si,'bars':segments[si]['bars'],'trigger_ticks':bad_ticks,'candidate_pool':len(choices),'before':before_verdict,'after':verdict})
    contexts=[];targets=[];history=[]
    for segment,options,ci in zip(segments,segment_candidates,path):
        candidate=options[ci]
        for position,i in enumerate(segment['idx']):
            row=candidate['rows'][position]
            if candidate['valid'] and len(history)>=RADIUS and all(item[2] for item in history[-RADIUS:]):
                context=[item[0] for item in history[-RADIUS:]]+[row.copy()]
                target=context[-1][WHAT:WHAT+N].copy();context[-1][WHAT:WHAT+N]=0;context[-1][-1]=1
                rel=[np.clip(item[1]-times[i],-16,16)/16 for item in history[-RADIUS:]]+[0.]
                contexts.append(np.c_[np.asarray(context),np.asarray(rel),np.tile([ds/15,(slot-4)/2,version/30],(RADIUS+1,1))]);targets.append(target)
            history.append((row,float(times[i]),bool(candidate['valid'])));history=history[-RADIUS:]
        if not segment['model_supported']:history=[]
    with torch.no_grad():
        x=torch.tensor(np.asarray(contexts,np.float32),device='cuda');y=torch.tensor(np.asarray(targets,np.float32),device='cuda')
        repaired_losses=[objective(model(x[lo:lo+256]),y[lo:lo+256]) for lo in range(0,len(x),256)]
        repaired_loss=float(torch.cat(repaired_losses).mean())
    selection_seconds=time.perf_counter()-started
    changes=[]
    for segment,options,ci in zip(segments,segment_candidates,path):
        rotation=options[ci]['rotation']
        changed=sum(events[int(ticks[i])]!=refined[int(ticks[i])] for i in segment['idx'])
        if changed:changes.append({'bars':segment['bars'],'rotation':rotation,'changed_events':changed})
        segment['chosen_rotation']=rotation;segment['changed_events']=changed

    diagnostics={'changed_segments':len(changes),'changed_events':sum(x['changed_events'] for x in changes),'scored_events':len(contexts),'harness_repair_steps':len(repair_steps),'selection_seconds':time.perf_counter()-started,'segments':[{k:v for k,v in segment.items() if k!='idx'} for segment in segments]}
    return refined,diagnostics
