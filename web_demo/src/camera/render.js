import { trackLabel } from '../api.js';
import { COCO_EDGES } from '../domain/pose.js';

export async function captureCameraFrame(video) {
    if(video.readyState<2 || !video.videoWidth || !video.videoHeight) return null;
    const scale=Math.min(1,960/Math.max(video.videoWidth,video.videoHeight));
    const canvas=document.createElement('canvas');
    canvas.width=Math.max(1,Math.round(video.videoWidth*scale));
    canvas.height=Math.max(1,Math.round(video.videoHeight*scale));
    canvas.getContext('2d').drawImage(video,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',0.82));
    if(!blob) throw new Error('无法采集摄像头画面，请重新开始。');
    return {blob,canvas,width:canvas.width,height:canvas.height};
}

export function drawCameraResult(canvas, frame, result, skeleton=true) {
    canvas.width=frame.width;canvas.height=frame.height;
    const ctx=canvas.getContext('2d'), width=canvas.width, height=canvas.height;
    ctx.drawImage(frame.canvas,0,0);
    const font=Math.max(14,Math.round(width/60));
    ctx.font=`600 ${font}px system-ui, sans-serif`;ctx.lineWidth=Math.max(2,width/450);
    for(const track of result.tracks) {
        const color=['#b6bcff','#6ddfff','#ffbf66'][track.class_id] || '#b9dae5';
        ctx.strokeStyle=color;ctx.fillStyle=color;
        const points=track.keypoints || [];
        const valid=point=>Array.isArray(point)&&point.length===3&&point.every(Number.isFinite)
            &&point[2]>=0.2&&point[0]>0&&point[1]>0&&point[0]<width&&point[1]<height;
        if(skeleton) {
            for(const [a,b] of COCO_EDGES) if(valid(points[a])&&valid(points[b])) {
                ctx.beginPath();ctx.moveTo(points[a][0],points[a][1]);ctx.lineTo(points[b][0],points[b][1]);ctx.stroke();
            }
            for(const point of points) if(valid(point)) {ctx.beginPath();ctx.arc(point[0],point[1],2.4,0,Math.PI*2);ctx.fill();}
        }
        if(!track.bbox?.every(Number.isFinite)||track.bbox.length!==4) continue;
        const [x1,y1,x2,y2]=track.bbox;
        ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        const label=`ID ${track.track_id} · ${trackLabel(track)}${Number.isFinite(track.class_probability)
            ?` ${(track.class_probability*100).toFixed(1)}%`:` ${track.history_length || 0}/50`}`;
        const textWidth=Math.min(width,ctx.measureText(label).width+14);
        const x=Math.max(0,Math.min(x1,width-textWidth)), y=Math.max(font+12,Math.min(y1,height));
        ctx.fillStyle='rgba(5,18,34,.88)';ctx.fillRect(x,y-font-12,textWidth,font+12);
        ctx.fillStyle=color;ctx.fillText(label,x+7,y-7,Math.max(1,width-x-14));
    }
}
