import json,glob,os,sys
import numpy as np
L='lbl/labels'
P='/root/.claude/uploads/f8807758-1b15-51ad-8e4d-9965c761ab37/cf14ace7-predictions.json'
preds=json.load(open(P))
files={os.path.basename(f)[:-4]:f for f in glob.glob(L+'/*.txt')}
print('label files',len(files))
pids=set(p['image_id'] for p in preds)
print('pred imgs',len(pids),'in labels',len(pids&set(files)),'labels w/o preds',len(set(files)-pids))
# image size: tiles 640
S=640
gt={};n=0;cls=set()
for k,f in files.items():
    b=[]
    for ln in open(f):
        t=ln.split()
        if len(t)<5: continue
        cls.add(t[0]);x,y,w,h=map(float,t[1:5])
        b.append([(x-w/2)*S,(y-h/2)*S,(x+w/2)*S,(y+h/2)*S])
    gt[k]=np.array(b).reshape(-1,4);n+=len(b)
print('classes',cls,'GT boxes',n,'tiles with gt',sum(1 for v in gt.values() if len(v)))
bb=np.array([p['bbox'] for p in preds]);print('pred bbox max x+w',(bb[:,0]+bb[:,2]).max(),(bb[:,1]+bb[:,3]).max())
byimg={}
for p in preds:
    x,y,w,h=p['bbox'];byimg.setdefault(p['image_id'],[]).append((p['score'],x,y,x+w,y+h))
def iou(a,B):
    ix=np.maximum(0,np.minimum(a[2],B[:,2])-np.maximum(a[0],B[:,0]));iy=np.maximum(0,np.minimum(a[3],B[:,3])-np.maximum(a[1],B[:,1]))
    i=ix*iy;return i/((a[2]-a[0])*(a[3]-a[1])+(B[:,2]-B[:,0])*(B[:,3]-B[:,1])-i+1e-9)
def run(thr_iou,src=None):
    recs=[];ng=0
    for k,g in gt.items():
        if src and not k.startswith(src):continue
        ng+=len(g);ps=sorted(byimg.get(k,[]),reverse=True);used=np.zeros(len(g),bool)
        for s,*b in ps:
            tp=0
            if len(g):
                io=iou(np.array(b),g);io[used]=-1;j=io.argmax()
                if io[j]>=thr_iou: used[j]=True;tp=1
            recs.append((s,tp))
    recs.sort(key=lambda r:-r[0]);sc=np.array([r[0] for r in recs]);tp=np.array([r[1] for r in recs])
    ctp=np.cumsum(tp);cfp=np.cumsum(1-tp);rec=ctp/max(ng,1);prec=ctp/np.maximum(ctp+cfp,1)
    mrec=np.concatenate([[0],rec,[1]]);mpre=np.concatenate([[1],prec,[0]])
    mpre=np.flip(np.maximum.accumulate(np.flip(mpre)))
    i=np.where(mrec[1:]!=mrec[:-1])[0];ap=((mrec[i+1]-mrec[i])*mpre[i+1]).sum()
    return ap,sc,ctp,cfp,ng
for src in [None,'sard','heridal']:
    ap,sc,ctp,cfp,ng=run(0.5,src);print('\n==',src or 'ALL','GT',ng,'AP50 %.4f'%ap)
    f1=2*ctp/(ctp+cfp+ng);i=f1.argmax();print(' best-F1 thr %.3f F1 %.3f P %.3f R %.3f TP %d FP %d FN %d'%(sc[i],f1[i],ctp[i]/(ctp[i]+cfp[i]),ctp[i]/ng,ctp[i],cfp[i],ng-ctp[i]))
    for t in (0.25,0.3,0.34,0.4,0.5):
        j=np.searchsorted(-sc,-t,side='right')-1
        if j>=0:print(' conf>=%.2f: P %.3f R %.3f TP %d FP %d FN %d'%(t,ctp[j]/(ctp[j]+cfp[j]),ctp[j]/ng,ctp[j],cfp[j],ng-ctp[j]))
aps=[run(t)[0] for t in np.arange(0.5,0.96,0.05)];print('\nmAP50-95 %.4f'%np.mean(aps))
