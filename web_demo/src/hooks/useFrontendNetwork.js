import { useEffect, useState } from 'react';
import { ACTIVE_STATUSES, jobUrl } from '../api';

export function useFrontendNetwork(job, sample) {
    const [data,setData] = useState(null);
    const [error,setError] = useState('');
    const [loading,setLoading] = useState(false);
    const [revision,setRevision] = useState(0);
    const ready = Boolean(job?.id && sample?.frame != null && !ACTIVE_STATUSES.has(job?.status));
    const key = `${job?.id}:${sample?.frame}:${ready}`;
    useEffect(()=>{
        setData(null);setError('');setLoading(false);
        if (!ready) return;
        const controller = new AbortController();
        let timer;
        setLoading(true);
        const run = async (attempt=0) => {
            try {
                const response=await fetch(jobUrl(job.id,`/frontend-network?frame=${sample.frame}`),{signal:controller.signal});
                const result=await response.json();
                if(controller.signal.aborted) return;
                if(response.status===409 && attempt<8) { timer=setTimeout(()=>run(attempt+1),Math.min(2000,500*(attempt+1)));return; }
                if(!response.ok) throw new Error(result.detail || '读取视觉特征失败');
                setData({key,result});setLoading(false);
            } catch(cause) { if(!controller.signal.aborted) {setError(cause.message);setLoading(false);} }
        };
        timer=setTimeout(()=>run(),450);
        return ()=>{controller.abort();clearTimeout(timer);};
    },[job?.id,sample?.frame,ready,revision]);
    return {data:data?.key===key?data.result:null,error,loading,retry:()=>setRevision(v=>v+1)};
}
