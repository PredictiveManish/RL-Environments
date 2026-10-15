"""
export.py
=========
Two jobs:
  1. run_rollout()  -- replay a policy in the environment and capture the full
     physical state of the episode (trajectory, goal, disruptions, score).
  2. write_viewer() -- bake one or more rollouts, plus the campus geometry, into
     a single self-contained HTML file that renders the campus in 3D and
     animates every rollout. No internet, no server -- just open the file.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(os.path.dirname(_HERE), "viewer", "vendor")


def run_rollout(env, policy, task_id=0, seed=0, max_steps=None, label="rollout"):
    obs, info = env.reset(seed=seed, options={"task_id": task_id})
    max_steps = max_steps or env.max_steps
    done = False
    steps = 0
    while not done and steps < max_steps:
        obs, r, term, trunc, info = env.step(policy(env))
        done = term or trunc
        steps += 1
    st = env.get_state()
    st.update({
        "label": label,
        "success": bool(info.get("success", False)),
        "distance": float(info.get("distance", 0.0)),
        "steps": steps,
    })
    return st


def _campus_public(raw):
    """Trim the campus dict to exactly what the viewer needs."""
    return {
        "meta": raw["meta"],
        "boundary": raw["boundary"],
        "buildings": [{"name": b["name"], "kind": b["kind"], "height": b["height"],
                       "polygon": b["polygon"]} for b in raw["buildings"]],
        "paths": raw["paths"],
        "pois": raw["pois"],
    }


def build_viewer_html(campus_raw, rollouts, title="IKGPTU Campus RL Environment"):
    with open(os.path.join(_VENDOR, "three.min.js")) as f:
        three_js = f.read()
    with open(os.path.join(_VENDOR, "OrbitControls.js")) as f:
        orbit_js = f.read()

    html = _TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__THREE__", three_js)
    html = html.replace("__ORBIT__", orbit_js)
    html = html.replace("__CAMPUS_JSON__", json.dumps(_campus_public(campus_raw)))
    html = html.replace("__ROLLOUTS_JSON__", json.dumps(rollouts))
    return html


def write_viewer(path, campus_raw, rollouts, title="IKGPTU Campus RL Environment"):
    html = build_viewer_html(campus_raw, rollouts, title)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#0b1016;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#e8eef5}
  #app{position:fixed;inset:0}
  .panel{position:absolute;background:rgba(12,18,26,.82);backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,.09);border-radius:12px;padding:12px 14px;font-size:13px;line-height:1.5}
  #hud{top:14px;left:14px;min-width:250px}
  #hud h1{margin:0 0 2px;font-size:14px;letter-spacing:.02em}
  #hud .sub{color:#8fa3b8;font-size:11px;margin-bottom:8px}
  #hud .row{display:flex;justify-content:space-between;gap:16px}
  #hud .k{color:#8fa3b8}
  #hud .v{font-variant-numeric:tabular-nums;font-weight:600}
  .ok{color:#3ddc84}.bad{color:#ff6b6b}.warn{color:#ffcf5c}
  #controls{bottom:14px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:center;max-width:94vw}
  button{background:#1b2634;color:#e8eef5;border:1px solid rgba(255,255,255,.14);
    border-radius:8px;padding:7px 12px;font-size:12px;cursor:pointer}
  button:hover{background:#243346}
  button.on{background:#2d6cdf;border-color:#2d6cdf}
  #legend{top:14px;right:14px;font-size:12px}
  #legend .dot{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:7px;vertical-align:middle}
  #legend div{margin:3px 0}
  #err{position:absolute;inset:0;display:none;place-items:center;text-align:center;padding:30px;color:#ff8f8f}
  input[type=range]{width:120px}
</style>
</head>
<body>
<div id="app"></div>
<div id="hud" class="panel">
  <h1 id="hud-title">IKGPTU Campus RL Environment</h1>
  <div class="sub" id="hud-sub">Main Campus, Kapurthala</div>
  <div class="row"><span class="k">Rollout</span><span class="v" id="h-roll">-</span></div>
  <div class="row"><span class="k">Task</span><span class="v" id="h-task" style="text-align:right;max-width:150px">-</span></div>
  <div class="row"><span class="k">Step</span><span class="v" id="h-step">0</span></div>
  <div class="row"><span class="k">Distance to goal</span><span class="v" id="h-dist">-</span></div>
  <div class="row"><span class="k">Episode reward</span><span class="v" id="h-rew">0</span></div>
  <div class="row"><span class="k">Collisions</span><span class="v" id="h-col">0</span></div>
  <div class="row"><span class="k">Status</span><span class="v" id="h-status">-</span></div>
</div>
<div id="legend" class="panel">
  <div><span class="dot" style="background:#3ddc84"></span>Goal / entrance</div>
  <div><span class="dot" style="background:#2d6cdf"></span>Start</div>
  <div><span class="dot" style="background:#ff9d2e"></span>Shuttle (agent)</div>
  <div><span class="dot" style="background:#ff6b6b"></span>Disruption zone</div>
  <div><span class="dot" style="background:#9aa7b4"></span>Road / walkway</div>
</div>
<div id="controls" class="panel">
  <button id="btn-play">Pause</button>
  <button id="btn-restart">Restart</button>
  <button id="btn-prev">&larr; Rollout</button>
  <button id="btn-next">Rollout &rarr;</button>
  <button id="btn-top">Top view</button>
  <span style="color:#8fa3b8">Speed</span><input id="speed" type="range" min="1" max="12" value="4">
</div>
<div id="err">Could not start the 3D view. Please open this file in a modern browser with WebGL enabled.</div>

<script>__THREE__</script>
<script>__ORBIT__</script>
<script>
const CAMPUS = __CAMPUS_JSON__;
const ROLLOUTS = __ROLLOUTS_JSON__;

if (typeof THREE === "undefined") { document.getElementById("err").style.display = "grid"; }
else {
const KIND_COLOR = {
  admin:0x5b7cfa, library:0x7d5bfa, auditorium:0xfa5b8f, academic:0x3fb6d8,
  hostel:0x39c07a, cafeteria:0xf2a93b, workshop:0x9aa7b4, sports:0x2fd4a7, guest:0xc98b5b
};
const app = document.getElementById("app");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1016);
scene.fog = new THREE.Fog(0x0b1016, 1200, 2600);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
app.appendChild(renderer.domElement);

const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 1, 6000);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI/2.05;

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x24303c, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.15);
sun.position.set(600, 900, 400); sun.castShadow = true;
sun.shadow.mapSize.set(2048,2048);
const sc = sun.shadow.camera; sc.left=-1000; sc.right=1000; sc.top=1000; sc.bottom=-1000; sc.far=3000;
scene.add(sun);

// local ENU (x east, y north) -> three (x, height, z=-y)
const P = (x,y,h)=>({x:x, y:h||0, z:-y});
const V = (x,y,h)=> new THREE.Vector3(x, h||0, -y);

// ---- campus footprint ----
const cx = CAMPUS.boundary.reduce((a,p)=>a+p[0],0)/CAMPUS.boundary.length;
const cy = CAMPUS.boundary.reduce((a,p)=>a+p[1],0)/CAMPUS.boundary.length;
const ccenter = V(cx, cy, 0);

const ground = new THREE.Mesh(new THREE.PlaneGeometry(6000,6000),
  new THREE.MeshStandardMaterial({color:0x16202a, roughness:1, metalness:0}));
ground.rotation.x = -Math.PI/2; ground.receiveShadow = true; ground.position.y = -0.05;
scene.add(ground);

const bshape = new THREE.Shape();
CAMPUS.boundary.forEach((p,i)=>{ i?bshape.lineTo(p[0],p[1]):bshape.moveTo(p[0],p[1]); });
const boundaryMesh = new THREE.Mesh(new THREE.ShapeGeometry(bshape),
  new THREE.MeshStandardMaterial({color:0x1d3a2a, roughness:1, metalness:0, side:THREE.DoubleSide}));
boundaryMesh.rotation.x = -Math.PI/2; boundaryMesh.position.y = 0.02; boundaryMesh.receiveShadow = true;
scene.add(boundaryMesh);

const bpts = CAMPUS.boundary.map(p=>V(p[0],p[1],0.06)); bpts.push(bpts[0].clone());
scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(bpts),
  new THREE.LineBasicMaterial({color:0x4f7f63})));

// ---- roads & walkways ----
CAMPUS.paths.forEach(path=>{
  const pts = path.points.map(p=>V(p[0],p[1],0.3));
  const col = path.kind==="road" ? 0x6b7783 : (path.kind==="walk" ? 0x8a7f66 : 0x55606b);
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({color:col})));
});

// ---- building labels ----
function label(text){
  const c=document.createElement("canvas"); const s=2; const pad=16;
  const ctx=c.getContext("2d"); ctx.font=(28*s)+"px sans-serif";
  const w=ctx.measureText(text).width+pad*2; c.width=w; c.height=44*s;
  const x=c.getContext("2d"); x.font=(28*s)+"px sans-serif"; x.fillStyle="rgba(232,238,245,0.95)";
  x.fillText(text, pad, 32*s);
  const t=new THREE.CanvasTexture(c); t.minFilter=THREE.LinearFilter;
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:t, transparent:true, depthTest:false}));
  sp.scale.set(c.width*0.16, c.height*0.16, 1);
  return sp;
}

// ---- buildings ----
CAMPUS.buildings.forEach(b=>{
  const shape=new THREE.Shape();
  b.polygon.forEach((p,i)=>{ i?shape.lineTo(p[0],p[1]):shape.moveTo(p[0],p[1]); });
  const geo=new THREE.ExtrudeGeometry(shape,{depth:b.height,bevelEnabled:false});
  geo.rotateX(-Math.PI/2);
  const color=KIND_COLOR[b.kind]||0x8899aa;
  const mesh=new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:color, roughness:0.62, metalness:0.12, side:THREE.DoubleSide}));
  mesh.castShadow=true; mesh.receiveShadow=true; scene.add(mesh);
  const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geo,25),
    new THREE.LineBasicMaterial({color:0x0a0f14}));
  scene.add(edges);
  const c=b.polygon.reduce((a,p)=>[a[0]+p[0],a[1]+p[1]],[0,0]).map(v=>v/b.polygon.length);
  const lb=label(b.name); lb.position.copy(V(c[0],c[1],b.height+26)); scene.add(lb);
});

// ---- POI markers ----
CAMPUS.pois.forEach(p=>{
  const m=new THREE.Mesh(new THREE.SphereGeometry(5,12,12),
    new THREE.MeshStandardMaterial({color:0x3ddc84, emissive:0x1f7a45, emissiveIntensity:.7}));
  m.position.copy(V(p.xy[0],p.xy[1],4)); scene.add(m);
});

// ---- per-rollout dynamic objects ----
const agent=new THREE.Mesh(new THREE.SphereGeometry(7,20,20),
  new THREE.MeshStandardMaterial({color:0xff9d2e, emissive:0x7a4a00, emissiveIntensity:.6}));
agent.castShadow=true; scene.add(agent);
const goalRing=new THREE.Mesh(new THREE.RingGeometry(9,13,32),
  new THREE.MeshBasicMaterial({color:0x3ddc84, side:THREE.DoubleSide}));
goalRing.rotation.x=-Math.PI/2; goalRing.position.y=1.2; scene.add(goalRing);
const startRing=new THREE.Mesh(new THREE.RingGeometry(7,10,32),
  new THREE.MeshBasicMaterial({color:0x2d6cdf, side:THREE.DoubleSide}));
startRing.rotation.x=-Math.PI/2; startRing.position.y=1.2; scene.add(startRing);
const trailMat=new THREE.LineBasicMaterial({color:0xffd166});
let trail=new THREE.Line(new THREE.BufferGeometry(), trailMat); scene.add(trail);
let zoneGroup=new THREE.Group(); scene.add(zoneGroup);

let ri=0, t=0, playing=true, speed=4;
function loadRollout(i){
  ri=(i+ROLLOUTS.length)%ROLLOUTS.length; t=0;
  const r=ROLLOUTS[ri];
  goalRing.position.copy(V(r.goal[0], r.goal[1], 1.2));
  startRing.position.copy(V(r.start[0], r.start[1], 1.2));
  zoneGroup.clear();
  (r.dynamic_obstacles||[]).forEach(z=>{
    const ring=new THREE.Mesh(new THREE.RingGeometry(z[2]-2, z[2], 48),
      new THREE.MeshBasicMaterial({color:0xff6b6b, side:THREE.DoubleSide, transparent:true, opacity:.5}));
    ring.rotation.x=-Math.PI/2; ring.position.copy(V(z[0],z[1],0.8)); zoneGroup.add(ring);
  });
  document.getElementById("h-roll").textContent=(ri+1)+" / "+ROLLOUTS.length;
  document.getElementById("h-task").textContent=r.start_name+" \u2192 "+r.goal_name;
  document.getElementById("h-status").textContent="";
  update(0);
}
function update(idx){
  const r=ROLLOUTS[ri];
  const pos=r.trajectory[Math.min(idx, r.trajectory.length-1)];
  agent.position.copy(V(pos[0],pos[1],7));
  const seg=r.trajectory.slice(0, idx+1).map(p=>V(p[0],p[1],7));
  trail.geometry.dispose(); trail.geometry=new THREE.BufferGeometry().setFromPoints(seg);
  const g=V(r.goal[0],r.goal[1],7);
  const d=Math.hypot(pos[0]-r.goal[0], pos[1]-r.goal[1]);
  document.getElementById("h-step").textContent=idx+" / "+(r.trajectory.length-1);
  document.getElementById("h-dist").textContent=d.toFixed(1)+" m";
  document.getElementById("h-rew").textContent=r.cum_reward.toFixed(1);
  document.getElementById("h-col").textContent=r.n_collisions;
  const st=document.getElementById("h-status");
  if(idx>=r.trajectory.length-1){
    st.textContent=r.success?"REACHED GOAL":"timed out";
    st.className="v "+(r.success?"ok":"warn");
  } else { st.textContent="driving"; st.className="v"; }
}
function frame(){
  requestAnimationFrame(frame);
  const r=ROLLOUTS[ri];
  if(playing && t < r.trajectory.length-1){ t+=speed*0.35; update(Math.floor(t)); }
  else if(playing && t>=r.trajectory.length-1){ update(r.trajectory.length-1); }
  controls.update(); renderer.render(scene,camera);
}

// ---- camera framing ----
function frameCamera(top){
  const b=CAMPUS.boundary; let minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;
  b.forEach(p=>{minx=Math.min(minx,p[0]);maxx=Math.max(maxx,p[0]);miny=Math.min(miny,p[1]);maxy=Math.max(maxy,p[1]);});
  const span=Math.max(maxx-minx, maxy-miny);
  if(top){ camera.position.set(cx, span*0.92, -cy+0.01); controls.target.set(cx,0,-cy); }
  else { camera.position.set(cx-span*0.06, span*0.40, -cy+span*0.50); controls.target.set(cx,0,-cy); }
  controls.update();
}
frameCamera(false);

document.getElementById("btn-play").onclick=e=>{playing=!playing; e.target.textContent=playing?"Pause":"Play"; e.target.classList.toggle("on",!playing);};
document.getElementById("btn-restart").onclick=()=>{t=0; playing=true; document.getElementById("btn-play").textContent="Pause";};
document.getElementById("btn-prev").onclick=()=>loadRollout(ri-1);
document.getElementById("btn-next").onclick=()=>loadRollout(ri+1);
document.getElementById("btn-top").onclick=()=>frameCamera(true);
document.getElementById("speed").oninput=e=>speed=+e.target.value;
addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth,innerHeight);});

loadRollout(0); frame();
}
</script>
</body>
</html>
"""
