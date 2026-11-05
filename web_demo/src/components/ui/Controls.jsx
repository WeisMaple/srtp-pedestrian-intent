import { Children, Fragment, isValidElement, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { menuPosition, rangePercent, selectKey, selectionChange, stepNumber } from './controlModel';
import '../../styles/controls.css';

export function Button({className='',type='button',...props}) {
    return <button {...props} type={type} className={`ui-button ${className}`} />;
}
export function Range({className='',min=0,max=100,value,style,...props}) {
    return <input {...props} type="range" min={min} max={max} value={value} className={`ui-range ${className}`}
        style={{...style,'--range-fill':`${rangePercent(value,min,max)}%`}} />;
}
export function Checkbox({className='',...props}) {
    return <span className={`ui-checkbox ${className}`}><input {...props} type="checkbox" />
        <span aria-hidden="true"><svg viewBox="0 0 16 16"><path d="m3 8 3 3 7-7" /></svg></span></span>;
}
export function NumberField({className='',min=-Infinity,max=Infinity,step=1,value,onChange,disabled,...props}) {
    const label=props['aria-label']||'数值';
    const change=direction=>{const next=String(stepNumber(value,direction,min,max,step));onChange?.({target:{value:next},currentTarget:{value:next}});};
    return <span className={`ui-number ${className}`}>
        <input {...props} type="number" min={Number.isFinite(Number(min))?min:undefined} max={Number.isFinite(Number(max))?max:undefined}
            step={step} value={value} onChange={onChange} disabled={disabled} />
        <Button className="ui-number-step is-minus" tabIndex="-1" aria-label={`减少${label}`} disabled={disabled||Number(value)<=Number(min)} onClick={()=>change(-1)}>−</Button>
        <Button className="ui-number-step is-plus" tabIndex="-1" aria-label={`增加${label}`} disabled={disabled||Number(value)>=Number(max)} onClick={()=>change(1)}>+</Button>
    </span>;
}
export function Progress({value=0,max=100,className='',...props}) {
    return <span {...props} className={`ui-progress ${className}`} role="progressbar" aria-valuemin="0" aria-valuemax={max}
        aria-valuenow={Math.max(0,Math.min(Number(max),Number(value)||0))}><i style={{width:`${rangePercent(value,0,max)}%`}} /></span>;
}
export function Details({className='',...props}) {return <details {...props} className={`ui-details ${className}`} />;}
export function Summary({className='',...props}) {return <summary {...props} className={`ui-summary ${className}`} />;}
export function FilePicker({children,onFile,accept,disabled,className='',label='选择本地文件'}) {
    const input=useRef(null);
    return <span className="ui-file-picker"><input ref={input} type="file" hidden accept={accept} disabled={disabled}
        aria-label={label} onChange={event=>onFile?.(event.target.files?.[0])} />
        <Button className={className} disabled={disabled} onClick={()=>{input.current.value='';input.current.click();}}>{children}</Button></span>;
}

function optionRows(children) {
    return Children.toArray(children).flatMap(child=>{
        if(!isValidElement(child)) return [];
        if(child.type===Fragment) return optionRows(child.props.children);
        if(child.type!=='option') return [];
        const text=Children.toArray(child.props.children).join('');
        return [{value:String(child.props.value??text),label:child.props.children,text,disabled:child.props.disabled}];
    });
}

/** Select-only combobox. Focus stays on the trigger; navigation does not commit. */
export function Select({children,value,onChange,disabled,className='',name,id,...props}) {
    const rows=useMemo(()=>optionRows(children),[children]);
    const generated=useId(),listId=`${generated}-options`,trigger=useRef(null),menu=useRef(null);
    const selected=rows.findIndex(row=>row.value===String(value));
    const [state,setState]=useState({open:false,active:selected});
    const [position,setPosition]=useState(null);
    const search=useRef({text:'',time:0});
    const open=state.open&&!disabled&&rows.length>0;
    const active=rows[state.active]&&!rows[state.active].disabled?state.active:Math.max(0,selected);
    const commit=next=>{const event=selectionChange(value,next,name);if(event)onChange?.(event);};
    const close=()=>setState(previous=>({...previous,open:false}));
    useLayoutEffect(()=>{
        if(!open) return undefined;
        const place=()=>{if(trigger.current)setPosition(menuPosition(trigger.current.getBoundingClientRect(),window.innerWidth,window.innerHeight,Math.min(300,rows.length*42+12)));};
        place();
        const outside=event=>{if(!trigger.current?.contains(event.target)&&!menu.current?.contains(event.target))close();};
        window.addEventListener('resize',place);window.addEventListener('scroll',place,true);
        document.addEventListener('pointerdown',outside);
        return ()=>{window.removeEventListener('resize',place);window.removeEventListener('scroll',place,true);document.removeEventListener('pointerdown',outside);};
    },[open,rows.length]);
    useLayoutEffect(()=>{if(open)document.getElementById(`${listId}-${active}`)?.scrollIntoView({block:'nearest'});},[open,active,listId]);
    const onKeyDown=event=>{
        if(disabled)return;
        if(event.ctrlKey||event.metaKey)return;
        const next=selectKey({open,active:open?active:selected},event.key,rows);
        if(next.handled){event.preventDefault();event.stopPropagation();}
        if(event.key.length===1&&event.key!==' ') {
            event.preventDefault();event.stopPropagation();
            const now=Date.now(),prior=now-search.current.time<600?search.current.text:'';
            const text=prior+event.key.toLocaleLowerCase();search.current={text,time:now};
            const query=[...text].every(letter=>letter===text[0])?text[0]:text;
            const start=query.length===1?active+1:0;
            const index=rows.map((_,offset)=>(start+offset)%rows.length).find(i=>!rows[i].disabled&&rows[i].text.toLocaleLowerCase().startsWith(query));
            setState({open:true,active:index??active});return;
        }
        if(next.commit!==undefined)commit(next.commit);
        setState({open:next.open,active:next.active});
    };
    return <span className={`ui-select ${className}`}>
        {name&&<input type="hidden" name={name} value={value??''} />}
        <Button {...props} id={id} ref={trigger} className="ui-select-trigger" role="combobox" aria-haspopup="listbox"
            aria-expanded={open} aria-controls={open?listId:undefined} aria-activedescendant={open?`${listId}-${active}`:undefined}
            disabled={disabled||!rows.some(row=>!row.disabled)} onKeyDown={onKeyDown}
            onBlur={close} onClick={()=>setState({open:!open,active:selected>=0?selected:rows.findIndex(row=>!row.disabled)})}>
            <span>{rows[selected]?.label??'请选择'}</span><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
        </Button>
        {open&&position&&createPortal(<div ref={menu} id={listId} className="ui-select-menu" role="listbox"
            aria-label={props['aria-label']||'选项'} style={position} onMouseDown={event=>event.preventDefault()}>
            {rows.map((row,index)=><div key={`${row.value}-${index}`} id={`${listId}-${index}`} role="option"
                aria-selected={selected===index} aria-disabled={row.disabled||undefined}
                className={`ui-select-option ${active===index?'is-active':''}`} onPointerMove={event=>{if(event.pointerType!=='touch'&&!row.disabled&&active!==index)setState({open:true,active:index});}}
                onClick={()=>{if(!row.disabled){commit(row.value);close();trigger.current?.focus();}}}>
                <span>{row.label}</span><span className="ui-select-check" aria-hidden="true">{selected===index?'✓':''}</span></div>)}
        </div>,document.fullscreenElement||document.body)}
    </span>;
}
