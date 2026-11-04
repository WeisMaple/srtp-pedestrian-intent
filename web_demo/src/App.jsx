import { useEffect, useState } from 'react';
import AppShell from './components/AppShell';
import { hashForPage, pageFromHash } from './app/page';

export default function App() {
    const [page, setPage] = useState(() => pageFromHash(window.location.hash));
    useEffect(() => {
        const onChange = () => setPage(pageFromHash(window.location.hash));
        window.addEventListener('hashchange', onChange);
        return () => window.removeEventListener('hashchange', onChange);
    }, []);
    const navigate = next => { window.location.hash = hashForPage(next); setPage(pageFromHash(hashForPage(next))); };
    return <AppShell page={page} onNavigate={navigate} status={{ ready: false, label: '本地模型' }}>
        <div className="page-placeholder">正在加载 {page}…</div>
    </AppShell>;
}
