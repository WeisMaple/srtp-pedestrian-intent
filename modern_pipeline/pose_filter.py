"""Causal per-track smoothing; short missing-joint holds expire after two frames."""
import numpy as np

class PoseFilter:
    def __init__(self):self.states={}
    def update(self,track_id,frame,pose):
        p=np.asarray(pose,dtype=np.float32).copy();old=self.states.get(track_id)
        valid=(p[:,2]>=.2)&(p[:,0]>0)&(p[:,1]>0)
        last=np.full(17,-100,dtype=int)
        if old is not None and 0<frame-old[0]<=2:
            _,prev,last=old;last=last.copy()
            smooth=valid & (frame-last<=2)
            p[smooth,:2]=.7*p[smooth,:2]+.3*prev[smooth,:2]
            hold=(~valid)&(frame-last<=2)
            p[hold]=prev[hold];p[hold,2]*=.8
        last[valid]=frame
        self.states[track_id]=(frame,p.copy(),last)
        if frame%100==0:self.states={k:v for k,v in self.states.items() if frame-v[0]<=30}
        return p
