import { Button } from './ui/Controls';
import { STEPS } from '../domain/timeline';

export default function StepNav({ state, actions }) {
    return <nav className="step-nav" aria-label="六步讲解">
        {STEPS.map(item => <Button key={item.step} aria-current={state.step === item.step ? 'step' : undefined}
            onClick={() => actions.selectStep(item.step)}>
            <span>{String(item.step).padStart(2, '0')}</span>
            <strong>{item.title}</strong>
            <small>{item.kicker}</small>
        </Button>)}
    </nav>;
}
