import { PAGES } from '../app/page.js';
const LABELS = { workbench: '工作台', corridor: '长廊', results: '结果' };

export default function AppShell({ page, onNavigate, status, children }) {
    return <div className="app-shell">
        <header className="app-top">
            <div className="app-brand">看见推理 <span>·</span> 行人意图</div>
            <nav className="app-nav" aria-label="页面">
                {PAGES.map(value => <button key={value} aria-current={page === value ? 'page' : undefined}
                    onClick={() => onNavigate(value)}>{LABELS[value]}</button>)}
            </nav>
            <div className={`app-status ${status?.ready ? 'ready' : ''}`}><i />{status?.label || '本地模型'}</div>
        </header>
        <main className="app-main">{children}</main>
    </div>;
}
