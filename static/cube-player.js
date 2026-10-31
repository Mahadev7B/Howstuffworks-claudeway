/* Deterministic, dependency-free cube animation. No network or speech synthesis. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CubeMotion = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const FACES={R:[0,1],L:[0,-1],U:[1,1],D:[1,-1],F:[2,1],B:[2,-1],x:[0,1],y:[1,1],z:[2,1]};
  const COLORS={W:'#f8fafc',Y:'#ffdb38',G:'#34bd75',B:'#408de9',R:'#f15358',O:'#ff9c43'};
  const FPS=30,TURN=30,HOLD=15,LEAD=15;
  const clone=s=>s.map(v=>({position:v.position.slice(),normal:v.normal.slice(),color:v.color}));
  function notation(token) {
    if(typeof token!=='string'||!/^([RLUDFBxyz])(['2]?)$/.test(token))throw Error('Invalid cube move');
    const [axis,sign]=FACES[token[0]];
    return {axis,sign,whole:/[xyz]/.test(token[0]),quarter:-sign*(token.endsWith('2')?2:token.endsWith("'")?-1:1),token};
  }
  function rotate(v,axis,angle) {
    const [x,y,z]=v,c=Math.cos(angle),s=Math.sin(angle);
    return axis===0?[x,y*c-z*s,y*s+z*c]:axis===1?[x*c+z*s,y,-x*s+z*c]:[x*c-y*s,x*s+y*c,z];
  }
  function turn(state,token) {
    const m=notation(token),a=m.quarter*Math.PI/2;
    return state.map(v=>m.whole||v.position[m.axis]===m.sign?{
      position:rotate(v.position,m.axis,a).map(x=>Math.round(x)||0),
      normal:rotate(v.normal,m.axis,a).map(x=>Math.round(x)||0),color:v.color
    }:{position:v.position.slice(),normal:v.normal.slice(),color:v.color});
  }
  function validScene(scene) {
    if(!scene||!Array.isArray(scene.initial)||scene.initial.length!==24||!Array.isArray(scene.moves)||scene.moves.length>160)throw Error('Invalid cube scene');
    const keys=new Set(),counts={};
    for(const s of scene.initial) {
      if(!s||!Array.isArray(s.position)||!Array.isArray(s.normal)||s.position.length!==3||s.normal.length!==3||!COLORS[s.color]||
         !s.position.every(x=>x===-1||x===1)||!s.normal.every(x=>[-1,0,1].includes(x))||s.normal.reduce((a,x)=>a+Math.abs(x),0)!==1||
         s.normal.some((x,i)=>x!==0&&s.position[i]!==x))throw Error('Invalid cube sticker');
      keys.add(s.position.join()+':'+s.normal.join());counts[s.color]=(counts[s.color]||0)+1;
    }
    if(keys.size!==24||Object.keys(counts).length!==6||Object.values(counts).some(n=>n!==4))throw Error('Invalid cube colors');
    scene.moves.forEach(notation);
    return scene;
  }
  function duration(scene) {return scene.moves.length?LEAD+scene.moves.length*(TURN+HOLD):LEAD;}
  function poseAtFrame(scene,frame) {
    validScene(scene);if(!Number.isFinite(frame))throw Error('Invalid frame');
    frame=Math.max(0,frame);let state=clone(scene.initial);
    const elapsed=Math.max(0,frame-LEAD),index=Math.min(scene.moves.length,Math.floor(elapsed/(TURN+HOLD)));
    for(let i=0;i<index;i++)state=turn(state,scene.moves[i]);
    if(index===scene.moves.length)return {stickers:state,base:clone(state),moveIndex:index,progress:1,rotation:null,complete:true};
    const t=frame<LEAD?0:Math.min(1,(elapsed-index*(TURN+HOLD))/TURN),progress=t*t*(3-2*t),m=notation(scene.moves[index]);
    if(t===1)return {stickers:turn(state,m.token),base:turn(state,m.token),moveIndex:index,progress:1,rotation:null,complete:false};
    const angle=m.quarter*Math.PI/2*progress;
    const stickers=state.map(v=>m.whole||v.position[m.axis]===m.sign?{
      position:rotate(v.position,m.axis,angle),normal:rotate(v.normal,m.axis,angle),color:v.color
    }:{position:v.position.slice(),normal:v.normal.slice(),color:v.color});
    return {stickers,base:state,moveIndex:index,progress,rotation:{...m,angle},complete:false};
  }
  class Timeline {
    constructor(scene){this.scene=validScene(scene);this.frame=0;this.playing=false;this.rate=0.9;this.anchor=0;this.base=0;}
    play(now){if(this.frame>=duration(this.scene))this.frame=0;this.base=this.frame;this.anchor=now;this.playing=true;}
    tick(now){if(this.playing){this.frame=Math.min(duration(this.scene),this.base+Math.max(0,now-this.anchor)*FPS*this.rate/1000);if(this.frame>=duration(this.scene))this.playing=false;}return this.frame;}
    pause(now){this.tick(now);this.playing=false;}
    reset(){this.frame=0;this.base=0;this.playing=false;}
    seekMove(index){index=Math.max(0,Math.min(this.scene.moves.length,Math.trunc(index)));this.playing=false;this.frame=index===0?0:LEAD+index*(TURN+HOLD);this.base=this.frame;}
    setRate(rate,now){if(!Number.isFinite(rate)||rate<0.3||rate>2)throw Error('Invalid playback rate');this.tick(now);this.rate=rate;this.base=this.frame;this.anchor=now;}
  }
  const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0),add=(a,b)=>a.map((v,i)=>v+b[i]),mul=(a,k)=>a.map(x=>x*k);
  const unit=v=>mul(v,1/Math.hypot(...v));
  const camera=unit([4,3.1,5]),right=unit([5,0,-4]);
  const up=unit([-camera[1]*4,camera[2]*5+camera[0]*4,-camera[1]*5]);
  function moveLabel(token) {
    if(!token)return 'Example complete';
    const names={R:'Right face',L:'Left face',U:'Top face',D:'Bottom face',F:'Front face',B:'Back face',x:'Whole cube',y:'Whole cube',z:'Whole cube'};
    const kind=/[xyz]/.test(token[0])?(token==='x2'?' — turn it over':' — change your hold'):token.endsWith('2')?' — half turn':token.endsWith("'")?' — counterclockwise':' — clockwise';
    return names[token[0]]+kind;
  }
  function draw(canvas,scene,frame) {
    const ctx=canvas.getContext('2d');if(!ctx)return;
    const w=canvas.width,h=canvas.height,pose=poseAtFrame(scene,frame),r=pose.rotation;
    const bg=ctx.createLinearGradient(0,0,w,h);bg.addColorStop(0,'#172b42');bg.addColorStop(1,'#091625');ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
    const project=p=>[w*.43+dot(p,right)*h*.28,h*.53-dot(p,up)*h*.28];
    const polys=[],selected=p=>r&&(r.whole||p[r.axis]===r.sign);
    function polygon(points,n,color,active){if(dot(n,camera)<=0.005)return;polys.push({points,depth:points.reduce((a,p)=>a+dot(p,camera),0)/4,color,active});}
    function face(center,normal,size,color,moving,active) {
      const axis=normal.findIndex(v=>v!==0),others=[0,1,2].filter(i=>i!==axis),u=[0,0,0],v=[0,0,0];u[others[0]]=size;v[others[1]]=size;
      let points=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>add(add(center,mul(u,a)),mul(v,b)));
      let n=normal;if(moving){points=points.map(p=>rotate(p,r.axis,r.angle));n=rotate(n,r.axis,r.angle);}polygon(points,n,color,active);
    }
    const corners=new Map();pose.base.forEach(s=>corners.set(s.position.join(),s.position));
    for(const p of corners.values())for(const [axis,sign]of [[0,1],[0,-1],[1,1],[1,-1],[2,1],[2,-1]]){
      const n=[0,0,0];n[axis]=sign;face(add(mul(p,.5),mul(n,.475)),n,.475,'#111c2b',selected(p),false);
    }
    for(const s of pose.base){const moving=!!selected(s.position),focus=scene.focus?.corner;
      const active=moving||(Array.isArray(focus)&&focus.every((v,i)=>v===s.position[i]));
      face(add(mul(s.position,.5),mul(s.normal,.486)),s.normal,.394,COLORS[s.color],moving,active);}
    polys.sort((a,b)=>a.depth-b.depth);
    for(const q of polys){const pts=q.points.map(project);ctx.beginPath();pts.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=q.color;ctx.fill();ctx.strokeStyle=q.active?'#91fff0':'#08101c';ctx.lineWidth=q.active?3:2;ctx.lineJoin='round';ctx.stroke();}
    // A small face-on cue removes the camera-dependent meaning of clockwise.
    const token=scene.moves[pose.moveIndex],m=token?notation(token):null;
    if(m&&!m.whole){const x=w*.82,y=h*.24,radius=h*.075;
      ctx.fillStyle='#203e56';ctx.beginPath();ctx.roundRect(x-radius-18,y-radius-24,radius*2+36,radius*2+77,14);ctx.fill();
      ctx.strokeStyle='#91fff0';ctx.lineWidth=5;const ccw=token.endsWith("'");const start=-Math.PI*.7,end=start+(ccw?-1:1)*Math.PI*1.55;
      ctx.beginPath();ctx.arc(x,y,radius,start,end,ccw);ctx.stroke();
      const px=x+radius*Math.cos(end),py=y+radius*Math.sin(end),tangent=end+(ccw?-1:1)*Math.PI/2;
      ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px-14*Math.cos(tangent-.55),py-14*Math.sin(tangent-.55));ctx.lineTo(px-14*Math.cos(tangent+.55),py-14*Math.sin(tangent+.55));ctx.closePath();ctx.fillStyle='#91fff0';ctx.fill();
      ctx.fillStyle='#fff';ctx.textAlign='center';ctx.font='bold '+Math.round(h*.055)+'px sans-serif';ctx.fillText(token[0],x,y+10);ctx.font=Math.round(h*.024)+'px sans-serif';ctx.fillText('Look at this face',x,y+radius+36);
    }
    ctx.fillStyle='#acc8d9';ctx.font=Math.round(h*.027)+'px sans-serif';ctx.textAlign='left';
    ctx.fillText(m?.whole?'Whole cube: change your hold, not just one layer':'FRONT stays facing you during face turns',w*.065,h*.92);
    return pose;
  }
  function mount(doc,data) {
    if(!data||!Array.isArray(data.scenes)||data.scenes.length!==16)return null;
    const views=Array.from(doc.querySelectorAll('.cube-motion'));if(views.length!==16)return null;
    if(views.some(v=>!v.querySelector('canvas')?.getContext('2d')))return null;
    const timelines=data.scenes.map(s=>new Timeline(s));let active=0,raf=0,pending=null;
    function resolvePending(completed){if(pending){const done=pending;pending=null;done(completed);}}
    function paint(){const view=views[active],t=timelines[active],scene=data.scenes[active],pose=draw(view.querySelector('canvas'),scene,t.frame);
      view.querySelector('canvas').hidden=false;const poster=view.querySelector('.cube-poster');if(poster)poster.hidden=true;
      const checkpoint=scene.move_labels?.[pose.complete?scene.moves.length-1:pose.moveIndex];
      view.querySelector('.cube-move-label').textContent=scene.moves.length?(pose.complete?'Example complete'+(checkpoint?' - '+checkpoint:''):(checkpoint?checkpoint+' - ':'')+moveLabel(scene.moves[pose.moveIndex])):scene.focus?.corner?'Look at the highlighted corner':'Look at this example';
      view.querySelector('.cube-move-count').textContent=scene.moves.length?(scene.moves[pose.moveIndex]||'Done')+' · Move '+Math.min(pose.moveIndex+1,scene.moves.length)+' of '+scene.moves.length:'Look and compare';
      view.querySelector('[data-cube-action="play"]').textContent=t.playing?'Pause animation':'Play animation';
      view.querySelector('.cube-caption').textContent=scene.caption||'';
    }
    function loop(now){raf=0;timelines[active].tick(now);paint();if(timelines[active].playing)raf=requestAnimationFrame(loop);else if(timelines[active].frame>=duration(data.scenes[active]))resolvePending(true);}
    function start(){timelines[active].play(performance.now());if(!raf)raf=requestAnimationFrame(loop);paint();}
    function halt(){if(raf)cancelAnimationFrame(raf);raf=0;timelines[active].pause(performance.now());paint();}
    function stop(){halt();resolvePending(false);}
    function show(index){stop();active=Math.max(0,Math.min(15,index));timelines[active].reset();paint();}
    views.forEach((v,index)=>v.addEventListener('click',e=>{const b=e.target.closest('[data-cube-action]');if(!b)return;
      if(active!==index)show(index);const t=timelines[active],action=b.dataset.cubeAction;
      if(action==='play'){if(t.playing)halt();else start();}
      else if(action==='replay'){halt();t.reset();start();}
      else {halt();const p=poseAtFrame(data.scenes[active],t.frame);t.seekMove(p.moveIndex+(action==='next'?1:-1));paint();if(t.frame>=duration(data.scenes[active]))resolvePending(true);}
    }));
    show(0);
    return {show,stop,setRate:r=>timelines.forEach(t=>t.setRate(r,performance.now())),
      begin:index=>{show(index);start();},
      play:index=>{if(active!==index)show(index);start();},
      finish:index=>{if(active!==index)return Promise.resolve(false);const t=timelines[active];if(t.frame>=duration(data.scenes[active]))return Promise.resolve(true);return new Promise(resolve=>{resolvePending(false);pending=resolve;});}};
  }
  return {turn,poseAtFrame,Timeline,duration,draw,mount,moveLabel,validScene};
});
