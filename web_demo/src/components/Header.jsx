import { BrandMark } from './Icon';

export default function Header({ health }) {
    const ready = health?.models_ready && health?.encoder_ready;
    const label = !health ? '检查本地环境' : health.unavailable ? '本地服务未连接' : ready ? '模型已就绪' : !health.models_ready ? '模型待配置' : '编码器待配置';
    return <header className="app-header">
        <a className="brand" href="#workbench"><BrandMark /><span>深度学习大作业<span className="brand-dot"> · </span>骨架时序分析</span></a>
        <span className="header-workbench-label">INFERENCE LAB <i>/</i> 全流程计算工作台</span>
        <span className={`model-status ${ready ? 'ready' : 'waiting'}`} title={health?.detail || ''}><i />{label}</span>
    </header>;
}
