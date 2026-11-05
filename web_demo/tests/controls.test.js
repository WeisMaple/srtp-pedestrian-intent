import test from 'node:test';
import assert from 'node:assert/strict';
import { selectKey, selectionChange, rangePercent, stepNumber, menuPosition } from '../src/components/ui/controlModel.js';

test('reconfirming the selected camera does not emit a change or restart inference',()=>{
    assert.equal(selectionChange('camera-1','camera-1','camera'),null);
    assert.equal(selectionChange(1,'1'),null);
    const event=selectionChange('camera-1','camera-2','camera');
    assert.deepEqual(event.target,{value:'camera-2',name:'camera'});
    assert.equal(event.target,event.currentTarget);
});

const options=[{value:'a',label:'Alpha'},{value:'b',label:'Beta',disabled:true},{value:'c',label:'Charlie'}];
test('select keyboard navigation skips disabled options and commits only on confirmation',()=>{
    let state={open:false,active:0};
    state=selectKey(state,'ArrowDown',options);
    assert.equal(state.open,true);assert.equal(state.active,0);assert.equal(state.commit,undefined);
    state=selectKey(state,'ArrowDown',options);
    assert.equal(state.active,2);assert.equal(state.commit,undefined);
    state=selectKey(state,'Enter',options);
    assert.equal(state.commit,'c');assert.equal(state.open,false);
    const escaped=selectKey({open:true,active:2},'Escape',options);
    assert.equal(escaped.commit,undefined);assert.equal(escaped.open,false);
    assert.equal(selectKey({open:true,active:2},'Home',options).active,0);
    assert.equal(selectKey({open:true,active:0},'End',options).active,2);
    assert.equal(selectKey({open:true,active:2},'Tab',options).commit,'c');
    assert.equal(selectKey({open:false,active:0},'x',[]).active,-1);
});
test('numeric controls preserve bounds and fractional steps without producing NaN',()=>{
    assert.equal(stepNumber(0,-1,0,199,1),0);
    assert.equal(stepNumber(199,1,0,199,1),199);
    assert.equal(stepNumber('',1,3,10,1),4);
    assert.equal(stepNumber(0.2,1,0,1,0.1),0.3);
    assert.equal(rangePercent(4,0,8),50);
    assert.equal(rangePercent(2,2,2),0);
    assert.equal(rangePercent(Infinity,0,10),0);
});
test('select menus flip above near the viewport bottom and fit narrow viewports',()=>{
    const bottom=menuPosition({left:290,top:600,bottom:632,width:180},390,700,240);
    assert.ok(bottom.left>=8);assert.ok(bottom.left+bottom.width<=382);
    assert.ok(bottom.top+bottom.maxHeight<=594);
    const top=menuPosition({left:10,top:20,bottom:52,width:260},320,640,240);
    assert.equal(top.top,58);assert.ok(top.width<=304);
});
