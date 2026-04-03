import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 32, 32);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const ambient = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambient);
const directional = new THREE.DirectionalLight(0xffffff, 0.8);
directional.position.set(10, 20, 10);
scene.add(directional);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(50, 50),
  new THREE.MeshStandardMaterial({ color: 0x1f2937 })
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const depot = new THREE.Mesh(
  new THREE.BoxGeometry(2.5, 2.5, 2.5),
  new THREE.MeshStandardMaterial({ color: 0x3b82f6 })
);
depot.position.set(0, 1.25, 0);
scene.add(depot);

const vehicle = new THREE.Mesh(
  new THREE.BoxGeometry(1.3, 0.8, 0.8),
  new THREE.MeshStandardMaterial({ color: 0x93c5fd })
);
vehicle.position.set(0, 0.4, 0);
scene.add(vehicle);

const schoolMeshes = new Map();
let currentState = null;

function mapCoord(v) {
  return (v - 0.5) * 32;
}

function stockColor(stockRatio) {
  if (stockRatio < 0.2) return 0xef4444;
  if (stockRatio < 0.6) return 0xf59e0b;
  return 0x22c55e;
}

function ensureSchoolMesh(school) {
  if (schoolMeshes.has(school.id)) return schoolMeshes.get(school.id);

  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(1.6, 1.6, 1.6),
    new THREE.MeshStandardMaterial({ color: stockColor(school.stock_ratio) })
  );
  mesh.position.set(mapCoord(school.x), 0.8, mapCoord(school.y));
  scene.add(mesh);
  schoolMeshes.set(school.id, mesh);
  return mesh;
}

function updateFromState(state) {
  currentState = state;
  state.schools.forEach((school) => {
    const mesh = ensureSchoolMesh(school);
    mesh.material.color.setHex(stockColor(school.stock_ratio));

    const scale = 0.8 + school.vulnerability * 1.2;
    mesh.scale.set(scale, scale * 1.4, scale);
    mesh.position.y = mesh.scale.y * 0.5;
  });

  if (state.vehicle) {
    const tx = mapCoord(state.vehicle.x);
    const tz = mapCoord(state.vehicle.y);
    const t = 0.25;
    vehicle.position.x += (tx - vehicle.position.x) * t;
    vehicle.position.z += (tz - vehicle.position.z) * t;
  }

  const stats = document.getElementById("stats");
  stats.innerHTML = `
    <div>Episode: <b>${state.episode}</b></div>
    <div>Step: <b>${state.step}</b> | Week: <b>${state.week}</b></div>
    <div>Depot stock: <b>${state.depot_stock.toFixed(1)}</b></div>
    <div>Total reward: <b>${state.total_reward.toFixed(2)}</b></div>
    <div>Last action: <b>${state.last_action_name ?? "None"}</b></div>
    <div>Status: <b>${state.done ? "Done" : "Running"}</b></div>
    <div style="margin-top:6px;font-size:11px;opacity:0.85">Live: GET /state every 500ms</div>
  `;
}

async function fetchState() {
  const res = await fetch("/state");
  if (!res.ok) throw new Error(`state ${res.status}`);
  return await res.json();
}

async function pollState() {
  try {
    const state = await fetchState();
    updateFromState(state);
  } catch (e) {
    console.error(e);
  }
}

async function stepOnce() {
  const res = await fetch("/step", { method: "POST" });
  const data = await res.json();
  updateFromState(data);
}

async function resetEnv() {
  const res = await fetch("/reset", { method: "POST" });
  const data = await res.json();
  updateFromState(data);
}

document.getElementById("stepBtn").addEventListener("click", async () => {
  await stepOnce();
});

document.getElementById("resetBtn").addEventListener("click", async () => {
  await resetEnv();
});

async function init() {
  await pollState();
  setInterval(pollState, 500);
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init();
animate();
