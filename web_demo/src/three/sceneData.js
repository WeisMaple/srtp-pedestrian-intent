/** Data adapters for the spatial inspector. Missing values stay missing. */
export const STAGE_NAMES = ['网络全景', 'YOLO 特征金字塔', '轨迹记忆', 'LSTM 状态流', '未来骨架', '空间图卷积', '时间卷积', '意图分类'];
export const GRAPH_EDGES = [[15,13],[13,11],[16,14],[14,12],[5,11],[6,12],[5,7],[6,8],[7,9],[8,10],[0,1],[0,2],[1,3],[2,4],[0,5],[0,6]];
export const TEMPLATE_POSE = [[0,2.4],[-.16,2.5],[.16,2.5],[-.34,2.4],[.34,2.4],[-.55,1.8],[.55,1.8],[-.8,1],[.8,1],[-.9,.25],[.9,.25],[-.38,.2],[.38,.2],[-.45,-.8],[.45,-.8],[-.5,-1.8],[.5,-1.8]];
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const at = (values, index) => index.reduce((v, i) => Array.isArray(v) ? v[i] : undefined, values);
export function evenlySpacedIndices(length, maximum) {
    if (!Number.isInteger(length) || length <= 0 || maximum <= 0) return [];
    const count = Math.min(length, Math.floor(maximum));
    return Array.from({ length: count }, (_, i) => count === 1 ? 0 : Math.round(i * (length - 1) / (count - 1)));
}
export function nestedShape(values) {
    const shape = []; let level = values;
    while (Array.isArray(level)) { shape.push(level.length); level = level[0]; }
    return shape;
}
export function flattenCells(values, maxCount = 512) {
    const result = [];
    function visit(item, index) {
        if (result.length >= maxCount) return;
        if (Array.isArray(item)) item.forEach((v, i) => visit(v, [...index, i]));
        else result.push({ value: finite(item) ? item : null, index });
    }
    visit(values, []); return result;
}
export function tensorCells(node, limits = [4, 6, 6]) {
    const values = node?.values; const shape = nestedShape(values);
    if (!shape.length) return [];
    const indices = shape.map((n, i) => evenlySpacedIndices(n, limits[i] || 1));
    const result = [];
    function visit(depth, sampleIndex, layoutIndex) {
        if (depth < shape.length) {
            indices[depth].forEach((i, j) => visit(depth + 1, [...sampleIndex, i], [...layoutIndex, j]));
        } else {
            const value = at(values, sampleIndex);
            const nativeIndex = sampleIndex.map((i, axis) => node.indices?.[axis]?.[i] ?? i);
            result.push({ value: finite(value) ? value : null, index: nativeIndex, sampleIndex, layoutIndex });
        }
    }
    visit(0, [], []); return result;
}
/** Assign one instance per lattice slot before filling structural placeholders.
 * Native tensor indices stay on the cell; layout indices only control placement.
 */
export function latticeSlots(cells, { cols, rows, depth = 1, capacity = cols * rows * depth }) {
    const count=Math.min(capacity,cols*rows*depth);
    const slots=Array.from({length:count},()=>null);
    const unpositioned=[];
    cells.forEach(cell=>{
        const layout=cell.layoutIndex;
        if(!layout || ![2,3].includes(layout.length)){unpositioned.push(cell);return;}
        const [z,y,x]=layout.length===3 ? layout : [0,...layout];
        if(![x,y,z].every(Number.isInteger) || x<0 || x>=cols || y<0 || y>=rows || z<0 || z>=depth)return;
        const slot=z*cols*rows+y*cols+x;
        if(slot<count && slots[slot]===null)slots[slot]=cell;
    });
    let next=0;
    unpositioned.forEach(cell=>{
        while(next<count && slots[next]!==null)next++;
        if(next<count)slots[next++]=cell;
    });
    return slots.map((cell,slot)=>({
        ...(cell || {value:null,index:[]}),
        layoutIndex:[Math.floor(slot/(cols*rows)),Math.floor(slot/cols)%rows,slot%cols],
    }));
}
export function asPose(value) {
    const v = value?.keypoints || value?.pose || value;
    if (!Array.isArray(v)) return null;
    if (v.length === 51 && !Array.isArray(v[0])) return asPose(Array.from({length:17},(_,i)=>v.slice(i*3,i*3+3)));
    return v.length === 17 && v.every(p => Array.isArray(p) && finite(p[0]) && finite(p[1])) ? v : null;
}
export function poseSampleIndex(node, requestedTime, future = false) {
    const length=node?.values?.length || 0;
    if (!length) return -1;
    const indices=node?.indices?.[0];
    if (!Array.isArray(indices) || !indices.length) return requestedTime>=0 && requestedTime<length ? requestedTime : -1;
    // Forecast indices are one-based horizons; history indices are native t.
    const requested=requestedTime+(future && indices[0]===1 ? 1 : 0);
    const validIndices=indices.slice(0,length);
    if(requested<Math.min(...validIndices) || requested>Math.max(...validIndices))return -1;
    let best=0;
    for(let i=1;i<Math.min(indices.length,length);i++)if(Math.abs(indices[i]-requested)<Math.abs(indices[best]-requested))best=i;
    return best;
}
export function poseCoordinateFrame(poses) {
    const points=(poses || []).flatMap(p=>asPose(p) || []);
    if (!points.length) return null;
    const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
    return { minX:Math.min(...xs), maxX:Math.max(...xs), minY:Math.min(...ys), maxY:Math.max(...ys) };
}
export function normalizePose(pose, size = 3.6, coordinateFrame = null) {
    const points = asPose(pose);
    if (!points) return TEMPLATE_POSE.map(([x,y])=>[x * size / 4.3, y * size / 4.3, null]);
    const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
    const bounds=coordinateFrame || {minX:Math.min(...xs),maxX:Math.max(...xs),minY:Math.min(...ys),maxY:Math.max(...ys)};
    const midX = (bounds.minX+bounds.maxX)/2, midY = (bounds.minY+bounds.maxY)/2;
    const scale = size / Math.max(bounds.maxY-bounds.minY, bounds.maxX-bounds.minX, 1e-6);
    return points.map(p=>[(p[0]-midX)*scale, -(p[1]-midY)*scale, finite(p[2]) ? p[2] : null]);
}
function snapshotNode(values, shape, source, extra = {}) { return { values, shape, source, ...extra }; }
export function sceneNodes({data, frontend, track, liveNodes, round = 0}) {
    // Callers omit saved records in live mode. Never combine a partially arrived
    // live frame with a previous saved inference.
    if (liveNodes && (Object.keys(liveNodes).length || (!data && !frontend))) return liveNodes;
    const nodes = {};
    const source = '所选记录 · 实际权重重算';
    (frontend?.layers || []).forEach((layer,i)=> {
        nodes[`yolo.${i}`] = snapshotNode(layer.values, layer.shape?.slice(-3), frontend.source?.note || source, { indices:[layer.channels,layer.sample_y,layer.sample_x], label:layer.type });
    });
    if (frontend?.first_conv?.example) {
        const conv = frontend.first_conv.example;
        nodes['yolo.patch'] = snapshotNode(conv.patch,[3,3,3],source);
        nodes['yolo.weights'] = snapshotNode(conv.weights,[3,3,3],source);
        nodes['yolo.conv_output'] = snapshotNode([conv.actual_output],[1],source);
    }
    const r = data?.lstm?.rounds?.[round] || data?.lstm?.rounds?.[0];
    (r?.layers || []).forEach((layer,i)=>nodes[`lstm.layer.${i}`] = snapshotNode(layer.h,[data.lstm.hidden_size],source,{cell:layer.c,round:round+1}));
    if (r?.sequence) nodes['lstm.sequence'] = {...r.sequence,source};
    if (round === 0 && data?.lstm?.gates?.steps?.length) {
        const gates=data.lstm.gates.steps.at(-1);
        nodes['lstm.gates'] = snapshotNode(['i','f','g','o','c','h'].map(k=>gates[k]),[6],source);
    }
    const history = data?.input?.values || track?.history;
    if (history?.length) nodes.history = snapshotNode(history.map(asPose),[history.length,17,3],data ? source : '保存的轨迹观测');
    if (data?.future?.values) nodes.future = snapshotNode(data.future.values,data.future.shape,source);
    (data?.stgcn?.blocks || []).forEach((block,i)=> ['spatial','temporal','output'].forEach(kind=>{
        if (block[kind]) nodes[`stgcn.${i}.${kind}`] = {...block[kind],shape:[block[kind].shape?.[1],block[kind].shape?.[3]],source_shape:block[kind].shape,source};
    }));
    const temporal = data?.stgcn?.blocks?.[0]?.arithmetic?.temporal;
    if (temporal) {
        nodes['stgcn.window'] = snapshotNode(temporal.input_values,[temporal.input_values.length,9],source,{indices:[undefined,temporal.input_timesteps]});
        nodes['stgcn.kernel'] = snapshotNode(temporal.weights,[temporal.input_values.length,9],source);
    }
    const g=data?.stgcn;
    ['pool','logits','probabilities'].forEach(key=>{if(g?.[key])nodes[`stgcn.${key}`]=snapshotNode(g[key],[g[key].length],source);});
    if (g?.head_weights) nodes['stgcn.weights'] = snapshotNode(g.head_weights,[3,256],source);
    return nodes;
}
export function eventStage(event) {
    const node = String(event?.node || '');
    if (node.startsWith('yolo')) return 1;
    if (node.startsWith('lstm')) return 3;
    if (node.includes('spatial')) return 5;
    if (node.includes('temporal') || /^stgcn\.\d+\.output/.test(node)) return 6;
    if (node.startsWith('stgcn.pool') || node.startsWith('stgcn.logits') || node === 'pipeline.output') return 7;
    return null;
}
