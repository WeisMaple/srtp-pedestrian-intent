const paths = {
    upload: <><path d="M8 16H6a4 4 0 0 1-.7-7.94 6.5 6.5 0 0 1 12.5-1.12A4.5 4.5 0 0 1 18 16h-2" /><path d="m8 12 4-4 4 4m-4-4v13" /></>,
    play: <path d="m9 5 11 7-11 7Z" />,
    pause: <><path d="M8 5v14M16 5v14" /></>,
    stop: <rect x="6" y="6" width="12" height="12" rx="1" />,
    download: <><path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5" /></>,
    chevron: <path d="m8 10 4 4 4-4" />,
    arrow: <path d="M4 12h16m-5-5 5 5-5 5" />,
    back: <path d="M20 12H4m5-5-5 5 5 5" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    file: <><path d="M14 3H5v18h14V8Z" /><path d="M14 3v5h5m-11 5h8m-8 4h5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    expand: <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />,
    video: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m10 9 5 3-5 3Z" /></>,
    person: <><circle cx="12" cy="5" r="2" /><path d="m7 11 5-3 5 3m-5-3v7m0-2-5 8m5-8 5 8m-10-10-3 5m13-5 3 5" /></>,
    target: <><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 3v4m8-4v4M4 11h4m8 0h4M8 17h8" /></>,
    stack: <><path d="m12 3 10 5-10 5L2 8Zm-9 9 9 5 9-5m-18 5 9 5 9-5" /></>,
    forecast: <><path d="M3 17c4 0 4-10 8-10s4 10 10 10" /><path d="m17 13 4 4-4 4" /><circle cx="3" cy="17" r="1" /></>,
    graph: <><circle cx="5" cy="5" r="2" /><circle cx="19" cy="6" r="2" /><circle cx="12" cy="19" r="2" /><path d="m7 5 10 1M6 7l5 10m7-9-5 9" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.1" /></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M5 8a8 8 0 0 1 13-3l2 3M4 16l2 3a8 8 0 0 0 13-3" /></>,
};

export default function Icon({ name, size = 20, className = '', ...props }) {
    return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className} {...props}>{paths[name] || paths.info}</svg>;
}

export function BrandMark() {
    return <svg className="brand-mark" viewBox="0 0 36 36" fill="none" aria-hidden="true"><path d="m7 27 14-20 8 23-22-3" stroke="currentColor" strokeWidth="1.6" /><circle cx="21" cy="7" r="5" fill="currentColor" stroke="white" strokeWidth="1.7" /><circle cx="7" cy="27" r="4.5" fill="currentColor" stroke="white" strokeWidth="1.7" /><circle cx="29" cy="30" r="4.5" fill="currentColor" stroke="white" strokeWidth="1.7" /></svg>;
}
