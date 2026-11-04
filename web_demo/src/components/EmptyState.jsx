export default function EmptyState({ title, children }) {
    return <div className="empty-state blueprint-panel"><strong>{title}</strong><p>{children}</p></div>;
}
