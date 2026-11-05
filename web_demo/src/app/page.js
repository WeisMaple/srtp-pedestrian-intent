export const PAGES = ['workbench', 'corridor', 'camera', 'results'];
const DEFAULT_PAGE = 'workbench';

export function pageFromHash(hash) {
    const value = String(hash || '').replace(/^#\/?/, '').split('/')[0];
    return PAGES.includes(value) ? value : DEFAULT_PAGE;
}

export function hashForPage(page) {
    return '#/' + (PAGES.includes(page) ? page : DEFAULT_PAGE);
}
