// 3D view of the room: region boxes, the source cell and the textured slice plane.
// Axes: three.js x = width (w), y = height (h), z = depth (d), all in metres.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const TYPE_STYLE = {
  wall: { color: '--wall', opacity: 0.55 },
  source: { color: '--source', opacity: 0.35 },
  listener: { color: '--listener', opacity: 0.16 },
};

export class RoomView {
  constructor(container, hoverLabel) {
    this.container = container;
    this.hoverLabel = hoverLabel;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.addEventListener('change', () => this.requestRender());

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const light = new THREE.DirectionalLight(0xffffff, 1.4);
    light.position.set(1, 2, 1.5);
    this.scene.add(light);

    this.room = new THREE.Group();
    this.scene.add(this.room);
    this.layers = { wall: new THREE.Group(), source: new THREE.Group(), listener: new THREE.Group() };
    Object.values(this.layers).forEach((g) => this.room.add(g));
    this.pickables = [];

    this.texture = null;
    this.plane = null;
    this.planeOutline = null;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.renderPending = false;

    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.domElement.addEventListener('pointermove', (e) => this.hover(e));
    this.renderer.domElement.addEventListener('pointerleave', () => { this.hoverLabel.hidden = true; });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.restyle());
  }

  setScene(info, imageCanvas) {
    this.room.clear();
    this.layers = { wall: new THREE.Group(), source: new THREE.Group(), listener: new THREE.Group() };
    Object.values(this.layers).forEach((g) => this.room.add(g));
    this.pickables = [];
    this.imageCanvas = imageCanvas;
    this.dx = info.grid.dx;
    this.shape = info.grid.shape;
    const [W, H, D] = info.scene.extent;
    this.extent = [W, H, D];

    // room outline and floor grid
    const roomBox = new THREE.BoxGeometry(W, H, D);
    this.roomEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(roomBox),
      new THREE.LineBasicMaterial({ color: cssVar('--ink-2') }));
    this.roomEdges.position.set(W / 2, H / 2, D / 2);
    this.room.add(this.roomEdges);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(W, D),
      new THREE.MeshBasicMaterial({ color: cssVar('--surface-2'), side: THREE.DoubleSide, transparent: true, opacity: 0.6, depthWrite: false }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(W / 2, 0, D / 2);
    floor.renderOrder = -1;
    this.floor = floor;
    this.room.add(floor);

    for (const region of info.scene.regions) {
      const size = [0, 1, 2].map((i) => Math.max(1e-3, region.max[i] - region.min[i]));
      const geometry = new THREE.BoxGeometry(...size);
      const style = TYPE_STYLE[region.type];
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: cssVar(style.color),
        transparent: true,
        opacity: style.opacity,
        depthWrite: false,
        roughness: 0.9,
      }));
      mesh.position.set(...[0, 1, 2].map((i) => (region.min[i] + region.max[i]) / 2));
      mesh.userData.region = region;
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({ color: cssVar(style.color), transparent: true, opacity: 0.8 }));
      edges.position.copy(mesh.position);
      edges.userData.styleKey = style.color;
      mesh.userData.styleKey = style.color;
      this.layers[region.type].add(mesh, edges);
      this.pickables.push(mesh);
    }

    // the active source cell
    const dx = this.dx;
    const radius = Math.max(dx * 0.6, Math.min(W, H, D) * 0.015);
    this.sourceMarker = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 20, 14),
      new THREE.MeshStandardMaterial({ color: cssVar('--source'), emissive: cssVar('--source'), emissiveIntensity: 0.4 }));
    this.sourceMarker.position.set(...info.source.position);
    this.sourceMarker.userData.region = { name: 'active source', type: 'source' };
    this.room.add(this.sourceMarker);
    this.pickables.push(this.sourceMarker);

    // slice plane, textured with the 2D slice image
    this.plane = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, transparent: true, opacity: 0.95 }));
    this.planeOutline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1)),
      new THREE.LineBasicMaterial({ color: cssVar('--accent') }));
    this.room.add(this.plane, this.planeOutline);

    // camera: three-quarter view from the front-right, above
    const span = Math.max(W, H, D);
    this.controls.target.set(W / 2, H / 2, D / 2);
    this.camera.position.set(W / 2 + span * 1.25, H / 2 + span * 0.95, D / 2 + span * 1.45);
    this.camera.near = span / 200;
    this.camera.far = span * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.resize();
  }

  setLayerVisible(type, visible) {
    if (this.layers[type]) this.layers[type].visible = visible;
    this.requestRender();
  }

  setSlice(axis, index) {
    if (!this.plane) return;
    const [W, H, D] = this.extent;
    const at = (index + 0.5) * this.dx;
    const plane = this.plane;
    plane.rotation.set(0, 0, 0);
    // PlaneGeometry lies in xy; texture u -> local x, v (image top) -> local +y.
    if (axis === 'h') {
      // plan: image right = +w, image down = +d
      plane.scale.set(W, D, 1);
      plane.rotation.x = -Math.PI / 2;
      plane.position.set(W / 2, at, D / 2);
    } else if (axis === 'w') {
      // side: image right = +d, image up = +h
      plane.scale.set(D, H, 1);
      plane.rotation.y = -Math.PI / 2;
      plane.position.set(at, H / 2, D / 2);
    } else {
      // front: image right = +w, image up = +h
      plane.scale.set(W, H, 1);
      plane.position.set(W / 2, H / 2, at);
    }
    this.planeOutline.position.copy(plane.position);
    this.planeOutline.rotation.copy(plane.rotation);
    this.planeOutline.scale.copy(plane.scale);
    this.updateTexture();
  }

  updateTexture() {
    if (!this.plane || !this.imageCanvas || !this.imageCanvas.width) return;
    const canvas = this.imageCanvas;
    if (!this.texture || this.texture.image !== canvas ||
        this.textureSize !== `${canvas.width}x${canvas.height}`) {
      this.texture?.dispose();
      this.texture = new THREE.CanvasTexture(canvas);
      this.texture.magFilter = THREE.NearestFilter;
      this.texture.minFilter = THREE.LinearFilter;
      this.texture.generateMipmaps = false;
      this.texture.colorSpace = THREE.SRGBColorSpace;
      this.textureSize = `${canvas.width}x${canvas.height}`;
      this.plane.material.map = this.texture;
      this.plane.material.needsUpdate = true;
    }
    this.texture.needsUpdate = true;
    this.requestRender();
  }

  restyle() {
    this.room.traverse((object) => {
      const key = object.userData.styleKey;
      if (key) object.material.color.set(cssVar(key));
    });
    this.roomEdges?.material.color.set(cssVar('--ink-2'));
    this.floor?.material.color.set(cssVar('--surface-2'));
    this.planeOutline?.material.color.set(cssVar('--accent'));
    this.requestRender();
  }

  hover(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const visible = this.pickables.filter((m) => m.parent?.visible !== false);
    const hit = this.raycaster.intersectObjects(visible, false)[0];
    if (!hit) {
      this.hoverLabel.hidden = true;
      return;
    }
    const region = hit.object.userData.region;
    let text = `${region.name} · ${region.type}`;
    if (region.type === 'wall') {
      text += region.material ? ` · ${Array.isArray(region.material) ? region.material.join(' + ') : region.material}` : '';
      text += ` · β ${region.beta.toFixed(3)}`;
    }
    this.hoverLabel.textContent = text;
    this.hoverLabel.hidden = false;
    this.hoverLabel.style.left = `${event.clientX - rect.left + 12}px`;
    this.hoverLabel.style.top = `${event.clientY - rect.top + 12}px`;
  }

  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  requestRender() {
    if (this.renderPending) return;
    this.renderPending = true;
    requestAnimationFrame(() => {
      this.renderPending = false;
      // keep animating while orbit damping settles
      if (this.controls.update()) this.requestRender();
      this.renderer.render(this.scene, this.camera);
    });
  }
}
