const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const file = path.join(__dirname, '../static/cube-player.js');
const motion = fs.existsSync(file) ? require(file) : {};
function api() { assert.equal(typeof motion.turn, 'function', 'real cube motion engine is required'); return motion; }
function solved() {
  const faces = [[0,1,'R'],[0,-1,'O'],[1,1,'W'],[1,-1,'Y'],[2,1,'G'],[2,-1,'B']];
  const out=[];
  for(const x of [-1,1]) for(const y of [-1,1]) for(const z of [-1,1]) {
    const p=[x,y,z]; for(const [axis,sign,color] of faces) if(p[axis]===sign) {
      const n=[0,0,0]; n[axis]=sign; out.push({position:p.slice(),normal:n,color});
    }
  }
  return out;
}
const scene = {initial:solved(),moves:['R',"R'"]};
test('clockwise R moves the front UFR sticker to top UBR',()=>{
  const r=api().turn(solved(),'R');
  assert.ok(r.some(s=>s.color==='G' && s.position.join()==='1,1,-1' && s.normal.join()==='0,1,0'));
});
test('clockwise U moves the front UFR sticker to left UFL',()=>{
  const r=api().turn(solved(),'U');
  assert.ok(r.some(s=>s.color==='G' && s.position.join()==='-1,1,1' && s.normal.join()==='-1,0,0'));
});
test('all face and whole-cube inverses restore the exact state without input mutation',()=>{
  const a=api(),original=solved(),before=JSON.stringify(original);
  for(const face of ['R','L','U','D','F','B','x','y','z']) {
    assert.deepEqual(a.turn(a.turn(original,face),face+"'"),original);
    let value=original;for(let i=0;i<4;i++)value=a.turn(value,face);
    assert.deepEqual(value,original);
  }
  assert.equal(JSON.stringify(original),before);
});
test('malformed notation and incomplete sticker states fail closed',()=>{
  const a=api(); for(const token of ['', 'Q','R3','R\n','RR',null])assert.throws(()=>a.turn(solved(),token));
  assert.throws(()=>a.poseAtFrame({initial:[],moves:['R']},30));
});
test('mid-turn is deterministic and moves only the selected rigid layer',()=>{
  const a=api();const first=a.poseAtFrame(scene,30),second=a.poseAtFrame(scene,30);
  assert.deepEqual(first,second);assert.equal(first.moveIndex,0);assert.ok(first.progress>0 && first.progress<1);
  for(let i=0;i<scene.initial.length;i++) if(scene.initial[i].position[0]===-1)assert.deepEqual(first.stickers[i],scene.initial[i]);
  assert.ok(first.stickers.some(s=>s.position.some(v=>Math.abs(v)!==1)));
});
test('completed frames snap to exact integer states and inverse restores start',()=>{
  const a=api();assert.deepEqual(a.poseAtFrame(scene,45).stickers,a.turn(scene.initial,'R'));
  assert.deepEqual(a.poseAtFrame(scene,100000).stickers,scene.initial);
  assert.equal(a.poseAtFrame(scene,100000).complete,true);
});
test('seeking backwards reconstructs state rather than accumulating drift',()=>{
  const a=api();a.poseAtFrame(scene,88);assert.deepEqual(a.poseAtFrame(scene,0).stickers,scene.initial);
  assert.deepEqual(a.poseAtFrame(scene,-50).stickers,scene.initial);
});
test('timeline pause, replay and move stepping do not advance while paused',()=>{
  const a=api(),t=new a.Timeline(scene);t.play(0);t.tick(1000);assert.ok(t.frame>0);
  t.pause(1000);const f=t.frame;t.tick(4000);assert.equal(t.frame,f);
  t.seekMove(1);assert.deepEqual(a.poseAtFrame(scene,t.frame).stickers,a.turn(scene.initial,'R'));
  t.reset();assert.equal(t.frame,0);assert.equal(t.playing,false);
  t.seekMove(100);assert.equal(a.poseAtFrame(scene,t.frame).complete,true);
});
test('empty-move teaching scene remains a stable valid cube',()=>{
  const a=api();const r=a.poseAtFrame({initial:solved(),moves:[]},10000);
  assert.deepEqual(r.stickers,solved());assert.equal(r.complete,true);
});

function mountedFixture(t, options={}) {
  const a=api(),callbacks=new Map();let sequence=0;
  const originalRaf=global.requestAnimationFrame,originalCancel=global.cancelAnimationFrame;
  global.requestAnimationFrame=fn=>{callbacks.set(++sequence,fn);return sequence;};
  global.cancelAnimationFrame=id=>callbacks.delete(id);
  t.after(()=>{global.requestAnimationFrame=originalRaf;global.cancelAnimationFrame=originalCancel;});
  const ctx=new Proxy({createLinearGradient:()=>({addColorStop(){}})},{get:(o,k)=>k in o?o[k]:()=>{}});
  const views=Array.from({length:16},()=>{
    const canvas={width:900,height:620,getContext:()=>options.noCanvas?null:ctx},nodes=new Map();let listener;
    return {querySelector:s=>s==='canvas'?canvas:s==='.cube-poster'?null:(nodes.has(s)?nodes.get(s):(nodes.set(s,{textContent:''}),nodes.get(s))),
      addEventListener:(type,fn)=>{listener=fn;},dispatch:action=>listener({target:{closest:()=>({dataset:{cubeAction:action}})}})};
  });
  const player=a.mount({querySelectorAll:()=>views},{scenes:Array.from({length:16},()=>options.scene||scene)});
  const finishFrames=()=>{for(const[id,fn]of [...callbacks]){callbacks.delete(id);fn(performance.now()+100000);}};
  return {player,views,finishFrames};
}
test('pausing an animation does not complete its waiting narration transition',async t=>{
  const {player,views,finishFrames}=mountedFixture(t);player.play(0);views[0].dispatch('play');
  let resolved=false;const waiting=player.finish(0).then(v=>{resolved=true;return v;});
  await Promise.resolve();assert.equal(resolved,false);
  views[0].dispatch('play');finishFrames();assert.equal(await waiting,true);
});

test('missing canvas support keeps the existing poster and narration usable',t=>{
  assert.equal(mountedFixture(t,{noCanvas:true}).player,null);
});

test('authored checkpoints appear beside the face direction and completed state',t=>{
  const authored={...scene,move_labels:['Check the corner','Keep the front facing you']};
  const {views}=mountedFixture(t,{scene:authored});
  assert.match(views[0].querySelector('.cube-move-label').textContent,/Check the corner.*Right face/);
  views[0].dispatch('next');views[0].dispatch('next');
  assert.match(views[0].querySelector('.cube-move-label').textContent,/Example complete.*Keep the front facing you/);
});

test('a new narration run resets a partly completed animation',t=>{
  const {player,views}=mountedFixture(t);
  views[0].dispatch('next');assert.match(views[0].querySelector('.cube-move-count').textContent,/R'/);
  player.begin(0);assert.doesNotMatch(views[0].querySelector('.cube-move-count').textContent,/R'/);
});

test('stale voice promises and events cannot interrupt a newer cube step',async()=>{
  const vm=require('node:vm'),html=fs.readFileSync(path.join(__dirname,'../templates/share.html'),'utf8');
  const begin=html.indexOf('      async function speakCurrent('),end=html.indexOf('\n      function autoAdvance()',begin),rejects=[];
  let mutations=0;
  const ctx={console:{warn(){}},statusEl:{},readBtn:{},tapPrompt:{},autoBtn:{classList:{add(){},remove(){}}},
    cubePlayer:{stop(){mutations++;},play(){mutations++;},begin(){mutations++;},finish:async()=>true},
    audioEl:{pause(){},load(){},play(){return new Promise((yes,no)=>rejects.push(no));}},
    speechRun:0,current:0,rate:.9,total:16,autoMode:false,getAudioUrl:async i=>'audio/'+i};
  ctx.stopAllAudio=()=>{ctx.speechRun++;ctx.cubePlayer.stop();};vm.createContext(ctx);
  vm.runInContext(html.slice(begin,end),ctx);
  ctx.setAuto(true);ctx.speakCurrent();await new Promise(setImmediate);
  const old={play:ctx.audioEl.onplay,error:ctx.audioEl.onerror,end:ctx.audioEl.onended};
  ctx.current=1;ctx.setAuto(true);ctx.speakCurrent();await new Promise(setImmediate);
  const status=ctx.statusEl.textContent,priorMutations=mutations;
  rejects[0](new Error('AbortError'));old.play();old.error();await old.end();await new Promise(setImmediate);
  assert.equal(ctx.autoMode,true);assert.equal(ctx.statusEl.textContent,status);assert.equal(mutations,priorMutations);
});
test('changing slides cancels the old animation completion without advancing it',async t=>{
  const {player}=mountedFixture(t);player.play(0);const waiting=player.finish(0);player.show(1);
  assert.equal(await waiting,false);
});
test('the authored example carries its exact state through all teaching steps5to16',()=>{
  const a=api(),data=JSON.parse(fs.readFileSync(path.join(__dirname,'../lesson_platform/cube_animation_v2.json'),'utf8'));
  const canonical=s=>s.map(v=>v.position.join()+':'+v.normal.join()+':'+v.color).sort();
  for(let i=4;i<15;i++)assert.deepEqual(canonical(a.poseAtFrame(data.scenes[i],1e8).stickers),canonical(data.scenes[i+1].initial));
  const last=a.poseAtFrame(data.scenes[15],1e8).stickers;
  for(const axis of [0,1,2])for(const sign of [-1,1])assert.equal(new Set(last.filter(s=>s.normal[axis]===sign).map(s=>s.color)).size,1);
});
