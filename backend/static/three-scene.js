/**
 * Career PathFinder — 3D WebGL Constellation & Quantum Core Engine
 * ================================================================
 * Built with Three.js for a high-fidelity, interactive 3D computing experience:
 * 1. Deep Space Neural Constellation (280 particle nodes with synaptic filaments)
 * 2. Floating 3D Quantum Polyhedral Core (wireframe icosahedron + inner glow + orbital rings)
 * 3. 3D Floating Geometry Nodes (torus rings, octahedrons) drifting in zero-gravity
 * 4. Interactive mouse parallax & kinetic inertia
 */

class WebGLSceneManager {
  constructor() {
    this.canvas = document.getElementById("canvas-3d");
    if (!this.canvas || typeof THREE === "undefined") {
      console.warn("Three.js not loaded or canvas missing.");
      return;
    }

    try {
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
      this.camera.position.z = 180;

      this.renderer = new THREE.WebGLRenderer({
        canvas: this.canvas,
        alpha: true,
        antialias: true
      });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    } catch (err) {
      console.warn("WebGL initialization failed; gracefully degrading to CSS ambient backdrop.", err);
      if (this.canvas) this.canvas.style.display = "none";
      return;
    }

    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this.clock = new THREE.Clock();

    // Accessibility + performance: honour reduced-motion, and idle when tab hidden.
    this.reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.paused = false;
    this._rendered = false;
    document.addEventListener("visibilitychange", () => { this.paused = document.hidden; });

    try {
      this.initConstellation();
      this.initQuantumCore();
      this.initFloatingArtifacts();
      this.bindEvents();
      this.animate();
    } catch (err) {
      console.warn("Three.js object setup failed; degrading gracefully.", err);
    }
  }

  initConstellation() {
    // Neural Skill Constellation (Particle Cloud + Dynamic Lines)
    const particleCount = 280;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const darkColors = new Float32Array(particleCount * 3);
    const lightColors = new Float32Array(particleCount * 3);

    // Dark Mode: glowing neon gold, cyan, mint
    const goldDark = new THREE.Color(0xE0A64C);
    const tealDark = new THREE.Color(0x00F0FF);
    const mintDark = new THREE.Color(0x4ADE80);

    // Light Mode: rich high-contrast dark amber, lapis blue, deep teal
    const amberLight = new THREE.Color(0xB45309);
    const lapisLight = new THREE.Color(0x0369A1);
    const tealLight = new THREE.Color(0x0F766E);

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * 600;
      positions[i3 + 1] = (Math.random() - 0.5) * 500;
      positions[i3 + 2] = (Math.random() - 0.5) * 400;

      // Color variation across nodes
      const rand = Math.random();
      const cDark = rand < 0.5 ? goldDark : (rand < 0.8 ? tealDark : mintDark);
      const cLight = rand < 0.5 ? amberLight : (rand < 0.8 ? lapisLight : tealLight);

      darkColors[i3] = cDark.r;
      darkColors[i3 + 1] = cDark.g;
      darkColors[i3 + 2] = cDark.b;

      lightColors[i3] = cLight.r;
      lightColors[i3 + 1] = cLight.g;
      lightColors[i3 + 2] = cLight.b;
    }

    this.darkColors = darkColors;
    this.lightColors = lightColors;

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(darkColors.slice(), 3));

    // Particle sprite
    const pMaterial = new THREE.PointsMaterial({
      size: 3.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending
    });

    this.particles = new THREE.Points(geometry, pMaterial);
    this.scene.add(this.particles);

    // Particle connectivity filaments
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xE0A64C,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending
    });

    // Build static connection mesh for nearby stars
    const linePositions = [];
    const threshold = 65;
    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const dx = positions[i*3] - positions[j*3];
        const dy = positions[i*3+1] - positions[j*3+1];
        const dz = positions[i*3+2] - positions[j*3+2];
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < threshold) {
          linePositions.push(positions[i*3], positions[i*3+1], positions[i*3+2]);
          linePositions.push(positions[j*3], positions[j*3+1], positions[j*3+2]);
        }
      }
    }

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(linePositions, 3));
    this.lines = new THREE.LineSegments(lineGeo, lineMat);
    this.scene.add(this.lines);
  }

  initQuantumCore() {
    this.coreGroup = new THREE.Group();

    // 1. Outer Hologram Wireframe Icosahedron
    const icoGeo = new THREE.IcosahedronGeometry(28, 1);
    const icoMat = new THREE.MeshBasicMaterial({
      color: 0xE0A64C,
      wireframe: true,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending
    });
    this.outerIco = new THREE.Mesh(icoGeo, icoMat);
    this.coreGroup.add(this.outerIco);

    // 2. Middle Dodecahedron Shell
    const dodGeo = new THREE.DodecahedronGeometry(18, 0);
    const dodMat = new THREE.MeshBasicMaterial({
      color: 0x00F0FF,
      wireframe: true,
      transparent: true,
      opacity: 0.55,
      blending: THREE.AdditiveBlending
    });
    this.midDod = new THREE.Mesh(dodGeo, dodMat);
    this.coreGroup.add(this.midDod);

    // 3. Inner Pulsating Energy Sphere
    const sphereGeo = new THREE.SphereGeometry(10, 24, 24);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0xFFF0D0,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });
    this.innerSphere = new THREE.Mesh(sphereGeo, sphereMat);
    this.coreGroup.add(this.innerSphere);

    // 4. Orbital Ring 1
    const ring1Geo = new THREE.TorusGeometry(38, 0.4, 16, 100);
    const ring1Mat = new THREE.MeshBasicMaterial({
      color: 0xE0A64C,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending
    });
    this.ring1 = new THREE.Mesh(ring1Geo, ring1Mat);
    this.ring1.rotation.x = Math.PI / 3;
    this.coreGroup.add(this.ring1);

    // 5. Orbital Ring 2
    const ring2Geo = new THREE.TorusGeometry(44, 0.3, 16, 100);
    const ring2Mat = new THREE.MeshBasicMaterial({
      color: 0x4ADE80,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending
    });
    this.ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
    this.ring2.rotation.y = Math.PI / 4;
    this.coreGroup.add(this.ring2);

    // Initial position: Top right background
    this.coreGroup.position.set(160, 40, -40);
    this.scene.add(this.coreGroup);
  }

  initFloatingArtifacts() {
    this.artifacts = [];
    
    // Create 4 floating geometric artifacts in deep space
    const shapes = [
      new THREE.OctahedronGeometry(12, 0),
      new THREE.TetrahedronGeometry(14, 0),
      new THREE.TorusGeometry(10, 2.5, 12, 48),
      new THREE.IcosahedronGeometry(8, 0)
    ];

    const colors = [0xE0A64C, 0x00F0FF, 0x4ADE80, 0x93C5FD];

    shapes.forEach((geo, i) => {
      const mat = new THREE.MeshBasicMaterial({
        color: colors[i],
        wireframe: true,
        transparent: true,
        opacity: 0.35,
        blending: THREE.AdditiveBlending
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(
        (i % 2 === 0 ? -1 : 1) * (140 + i * 30),
        (i < 2 ? -1 : 1) * (70 + i * 20),
        -60 - i * 20
      );
      mesh.rotSpeed = {
        x: (Math.random() - 0.5) * 0.015,
        y: (Math.random() - 0.5) * 0.015,
        z: (Math.random() - 0.5) * 0.015
      };
      this.artifacts.push(mesh);
      this.scene.add(mesh);
    });
  }

  bindEvents() {
    window.addEventListener("resize", () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    window.addEventListener("mousemove", (e) => {
      this.mouse.targetX = (e.clientX / window.innerWidth - 0.5) * 2;
      this.mouse.targetY = -(e.clientY / window.innerHeight - 0.5) * 2;
    });

    // Subtle scroll tilt
    window.addEventListener("scroll", () => {
      const scrollY = window.scrollY || document.documentElement.scrollTop;
      this.camera.position.y = -scrollY * 0.08;
    });
  }

  animate() {
    this._raf = requestAnimationFrame(() => this.animate());

    // Reduced-motion: draw a single static frame, then stay idle.
    if (this.reduceMotion) {
      if (!this._rendered) { this.renderer.render(this.scene, this.camera); this._rendered = true; }
      return;
    }
    // Tab hidden: skip all work to save CPU/GPU.
    if (this.paused) return;

    const delta = this.clock.getDelta();
    const time = this.clock.getElapsedTime();

    // Smooth mouse interpolation
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.05;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.05;

    // Rotate particle universe
    if (this.particles) {
      this.particles.rotation.y = time * 0.03 + this.mouse.x * 0.15;
      this.particles.rotation.x = time * 0.015 + this.mouse.y * 0.15;
    }
    if (this.lines) {
      this.lines.rotation.y = time * 0.03 + this.mouse.x * 0.15;
      this.lines.rotation.x = time * 0.015 + this.mouse.y * 0.15;
    }

    // Animate Quantum Core
    if (this.coreGroup) {
      this.outerIco.rotation.x = time * 0.25;
      this.outerIco.rotation.y = time * 0.35;
      this.midDod.rotation.x = -time * 0.3;
      this.midDod.rotation.z = time * 0.2;
      this.ring1.rotation.z = time * 0.4;
      this.ring2.rotation.x = time * 0.3;

      // Pulse inner sphere
      const pulse = 1 + Math.sin(time * 3) * 0.12;
      this.innerSphere.scale.set(pulse, pulse, pulse);

      // Core gentle float
      this.coreGroup.position.y = 40 + Math.sin(time * 1.5) * 8 + this.mouse.y * 12;
      this.coreGroup.position.x = 160 + this.mouse.x * 15;
    }

    // Animate floating artifacts
    this.artifacts.forEach((mesh, idx) => {
      mesh.rotation.x += mesh.rotSpeed.x;
      mesh.rotation.y += mesh.rotSpeed.y;
      mesh.rotation.z += mesh.rotSpeed.z;
      mesh.position.y += Math.sin(time * 1.2 + idx) * 0.15;
    });

    this.renderer.render(this.scene, this.camera);
  }

  setTheme(theme) {
    const isLight = theme === "light";
    const blendingMode = isLight ? THREE.NormalBlending : THREE.AdditiveBlending;

    // 1. Neural Particles (Constellation)
    if (this.particles && this.particles.material) {
      this.particles.material.blending = blendingMode;
      this.particles.material.opacity = isLight ? 0.85 : 0.75;
      this.particles.material.size = isLight ? 4.5 : 3.5;
      this.particles.material.needsUpdate = true;

      const colAttr = this.particles.geometry.getAttribute("color");
      if (colAttr && this.lightColors && this.darkColors) {
        colAttr.array.set(isLight ? this.lightColors : this.darkColors);
        colAttr.needsUpdate = true;
      }
    }

    // 2. Synaptic Filaments (Lines)
    if (this.lines && this.lines.material) {
      this.lines.material.blending = blendingMode;
      this.lines.material.opacity = isLight ? 0.35 : 0.12;
      this.lines.material.color.setHex(isLight ? 0xC06A0A : 0xE0A64C);
      this.lines.material.needsUpdate = true;
    }

    // 3. Quantum Polyhedral Core (3D Geometry in Upper Right)
    if (this.outerIco && this.outerIco.material) {
      this.outerIco.material.blending = blendingMode;
      this.outerIco.material.color.setHex(isLight ? 0xC06A0A : 0xE0A64C);
      this.outerIco.material.opacity = isLight ? 0.75 : 0.45;
      this.outerIco.material.needsUpdate = true;
    }
    if (this.midDod && this.midDod.material) {
      this.midDod.material.blending = blendingMode;
      this.midDod.material.color.setHex(isLight ? 0x0369A1 : 0x00F0FF);
      this.midDod.material.opacity = isLight ? 0.75 : 0.55;
      this.midDod.material.needsUpdate = true;
    }
    if (this.innerSphere && this.innerSphere.material) {
      this.innerSphere.material.blending = blendingMode;
      this.innerSphere.material.color.setHex(isLight ? 0xD97706 : 0xFFF0D0);
      this.innerSphere.material.opacity = isLight ? 0.90 : 0.8;
      this.innerSphere.material.needsUpdate = true;
    }
    if (this.ring1 && this.ring1.material) {
      this.ring1.material.blending = blendingMode;
      this.ring1.material.color.setHex(isLight ? 0xC06A0A : 0xE0A64C);
      this.ring1.material.opacity = isLight ? 0.70 : 0.4;
      this.ring1.material.needsUpdate = true;
    }
    if (this.ring2 && this.ring2.material) {
      this.ring2.material.blending = blendingMode;
      this.ring2.material.color.setHex(isLight ? 0x0F766E : 0x4ADE80);
      this.ring2.material.opacity = isLight ? 0.65 : 0.35;
      this.ring2.material.needsUpdate = true;
    }

    // 4. Floating 3D Geometric Artifacts
    if (this.artifacts && Array.isArray(this.artifacts)) {
      const lightArtifactColors = [0xC06A0A, 0x0369A1, 0x0F766E, 0x9333EA];
      const darkArtifactColors = [0xE0A64C, 0x00F0FF, 0x4ADE80, 0x93C5FD];
      this.artifacts.forEach((mesh, idx) => {
        if (mesh.material) {
          mesh.material.blending = blendingMode;
          mesh.material.color.setHex(isLight ? lightArtifactColors[idx % 4] : darkArtifactColors[idx % 4]);
          mesh.material.opacity = isLight ? 0.65 : 0.35;
          mesh.material.needsUpdate = true;
        }
      });
    }
  }
}

// ----------------- 3D Card Tilt Engine (DISABLED) -----------------
// Disabled: User feedback indicated all card movement on cursor hover is unwanted.
// The 3D WebGL background constellation remains fully active.
class Vanilla3DTilt {
  static init() {
    // No-op: Card tilt disabled by design decision.
  }
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  window.sceneManager = new WebGLSceneManager();
  window.webglScene = window.sceneManager;
  const savedTheme = localStorage.getItem("career_pathfinder_theme") || "dark";
  window.sceneManager.setTheme(savedTheme);
  Vanilla3DTilt.init();
});

window.Vanilla3DTilt = Vanilla3DTilt;
