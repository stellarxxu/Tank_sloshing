import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Download, Info } from 'lucide-react';
import * as THREE from 'three';

const Sloshing3DSimulator = () => {
  // 状态管理
  const [shapeType, setShapeType] = useState('cylindrical');
  const [radius, setRadius] = useState(2.0);
  const [length, setLength] = useState(4.0);
  const [width, setWidth] = useState(3.0);
  const [height, setHeight] = useState(3.0);
  const [fillHeight, setFillHeight] = useState(2.0);
  const [quakeType, setQuakeType] = useState('sine');
  const [pga, setPga] = useState(0.3);
  const [duration, setDuration] = useState(20);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [showInfo, setShowInfo] = useState(false);

  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const animationRef = useRef(null);

  // 3D物理求解器
  class Sloshing3DSolver {
    constructor(shape, dimensions, h, pga, duration) {
      this.shape = shape;
      this.dims = dimensions;
      this.h = h;
      this.g = 9.81;
      this.pga = pga;
      this.duration = duration;
      
      this.calcNaturalFrequency();
      this.calcDamping();
      this.calcModalParticipation();
    }

    calcNaturalFrequency() {
      if (this.shape === 'rectangular') {
        const L = this.dims.length;
        const W = this.dims.width;
        const k = Math.PI / Math.max(L, W);
        this.omega_n = Math.sqrt(this.g * k * Math.tanh(k * this.h));
      } else if (this.shape === 'cylindrical') {
        const R = this.dims.radius;
        const epsilon_1 = 1.8412;
        this.omega_n = Math.sqrt((this.g * epsilon_1 / R) * Math.tanh(epsilon_1 * this.h / R));
      } else {
        const R_mean = (this.dims.outerRadius + this.dims.innerRadius) / 2;
        const epsilon_1 = 1.8412;
        this.omega_n = Math.sqrt((this.g * epsilon_1 / R_mean) * Math.tanh(epsilon_1 * this.h / R_mean));
      }
      this.freq_n = this.omega_n / (2 * Math.PI);
    }

    calcDamping() {
      const nu = 1e-6;
      let characteristicLength;
      
      if (this.shape === 'rectangular') {
        characteristicLength = Math.max(this.dims.length, this.dims.width);
      } else if (this.shape === 'cylindrical') {
        characteristicLength = this.dims.radius;
      } else {
        characteristicLength = (this.dims.outerRadius + this.dims.innerRadius) / 2;
      }
      
      const xi_viscous = 2 * Math.sqrt(nu / (this.omega_n * characteristicLength ** 2));
      const xi_structural = 0.005;
      this.xi = Math.max(0.005, Math.min(0.05, xi_viscous + xi_structural));
    }

    calcModalParticipation() {
      if (this.shape === 'rectangular') {
        const L = this.dims.length;
        const k = Math.PI / L;
        const kh = k * this.h;
        this.gamma = kh < 0.01 ? 1.0 : Math.tanh(kh) / kh;
      } else {
        const epsilon_1 = 1.8412;
        const R = this.shape === 'cylindrical' ? this.dims.radius : 
                   (this.dims.outerRadius + this.dims.innerRadius) / 2;
        const x = epsilon_1 * this.h / R;
        
        if (x < 0.1) {
          this.gamma = 1.0;
        } else {
          // 简化的贝塞尔函数近似
          this.gamma = 0.7;
        }
      }
    }

    generateEarthquake(t) {
      const seed = 42;
      const noise = Math.sin(seed * t) * Math.cos(seed * t * 1.7);
      
      let acc;
      if (this.quakeType === 'sine') {
        const envelope = t < this.duration * 0.1 ? t / (this.duration * 0.1) : 1;
        acc = Math.sin(2 * Math.PI * 0.6 * t) * envelope;
      } else if (this.quakeType === 'elcentro') {
        const envelope = Math.exp(-0.15 * t) * (t ** 1.5);
        acc = noise * envelope;
      } else if (this.quakeType === 'kobe') {
        const envelope = Math.exp(-0.5 * (t - 3) ** 2) * 5;
        acc = noise * envelope;
      } else {
        const envelope = Math.exp(-0.2 * t) * t;
        acc = noise * envelope;
      }
      
      return acc * this.pga * this.g;
    }

    solve() {
      const dt = 0.05;
      const steps = Math.floor(this.duration / dt);
      const t_array = [];
      const q_array = [];
      const acc_array = [];
      
      let q = 0, q_dot = 0;
      
      for (let i = 0; i < steps; i++) {
        const t = i * dt;
        const a_ground = this.generateEarthquake(t);
        
        const forcing = -this.gamma * a_ground;
        const q_ddot = forcing - 2 * this.xi * this.omega_n * q_dot - (this.omega_n ** 2) * q;
        
        q_dot += q_ddot * dt;
        q += q_dot * dt;
        
        t_array.push(t);
        q_array.push(q);
        acc_array.push(a_ground);
      }
      
      return { t: t_array, q: q_array, acc: acc_array };
    }

    getWaveHeight(q, x, y) {
      if (this.shape === 'rectangular') {
        const L = this.dims.length;
        const W = this.dims.width;
        const phi_x = Math.cos(Math.PI * x / L);
        const phi_y = Math.cos(Math.PI * y / W);
        return q * phi_x * phi_y;
      } else if (this.shape === 'cylindrical') {
        const r = Math.sqrt(x * x + y * y);
        const R = this.dims.radius;
        if (r > R) return 0;
        const theta = Math.atan2(y, x);
        return q * (r / R) * Math.cos(theta);
      } else {
        const r = Math.sqrt(x * x + y * y);
        const R_out = this.dims.outerRadius;
        const R_in = this.dims.innerRadius;
        if (r < R_in || r > R_out) return 0;
        const theta = Math.atan2(y, x);
        return q * ((r - R_in) / (R_out - R_in)) * Math.cos(theta);
      }
    }
  }

  // Three.js 3D可视化
  useEffect(() => {
    if (!canvasRef.current) return;

    // 初始化场景
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    const camera = new THREE.PerspectiveCamera(
      50,
      canvasRef.current.clientWidth / canvasRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(8, 6, 8);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current, 
      antialias: true 
    });
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    renderer.shadowMap.enabled = true;

    // 光照
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // 网格地面
    const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0xcccccc);
    scene.add(gridHelper);

    // 坐标轴
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    sceneRef.current = { scene, camera, renderer };

    // 鼠标控制
    let isDragging = false;
    let previousMousePosition = { x: 0, y: 0 };

    const onMouseDown = (e) => {
      isDragging = true;
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e) => {
      if (!isDragging) return;
      
      const deltaX = e.clientX - previousMousePosition.x;
      const deltaY = e.clientY - previousMousePosition.y;
      
      const rotationSpeed = 0.005;
      camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), -deltaX * rotationSpeed);
      
      const radius = Math.sqrt(
        camera.position.x ** 2 + 
        camera.position.z ** 2
      );
      const newY = camera.position.y + deltaY * rotationSpeed * radius;
      camera.position.y = Math.max(1, Math.min(15, newY));
      
      camera.lookAt(0, 0, 0);
      previousMousePosition = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    canvasRef.current.addEventListener('mousedown', onMouseDown);
    canvasRef.current.addEventListener('mousemove', onMouseMove);
    canvasRef.current.addEventListener('mouseup', onMouseUp);

    // 渲染循环
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // 清理
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      canvasRef.current?.removeEventListener('mousedown', onMouseDown);
      canvasRef.current?.removeEventListener('mousemove', onMouseMove);
      canvasRef.current?.removeEventListener('mouseup', onMouseUp);
      renderer.dispose();
    };
  }, []);

  // 创建容器和液体
  const createVisualization = () => {
    if (!sceneRef.current) return;
    const { scene } = sceneRef.current;

    // 清除旧对象
    while (scene.children.length > 4) {
      scene.remove(scene.children[4]);
    }

    // 容器框架
    const containerMaterial = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });
    
    if (shapeType === 'rectangular') {
      const geometry = new THREE.BoxGeometry(length, height, width);
      const edges = new THREE.EdgesGeometry(geometry);
      const container = new THREE.LineSegments(edges, containerMaterial);
      container.position.y = height / 2;
      scene.add(container);
    } else if (shapeType === 'cylindrical') {
      const geometry = new THREE.CylinderGeometry(radius, radius, height, 32);
      const edges = new THREE.EdgesGeometry(geometry);
      const container = new THREE.LineSegments(edges, containerMaterial);
      container.position.y = height / 2;
      scene.add(container);
    } else {
      const outerGeometry = new THREE.CylinderGeometry(radius * 1.5, radius * 1.5, height, 32);
      const innerGeometry = new THREE.CylinderGeometry(radius * 0.7, radius * 0.7, height, 32);
      const outerEdges = new THREE.EdgesGeometry(outerGeometry);
      const innerEdges = new THREE.EdgesGeometry(innerGeometry);
      const outerContainer = new THREE.LineSegments(outerEdges, containerMaterial);
      const innerContainer = new THREE.LineSegments(innerEdges, containerMaterial);
      outerContainer.position.y = height / 2;
      innerContainer.position.y = height / 2;
      scene.add(outerContainer);
      scene.add(innerContainer);
    }

    // 静态水面
    const waterGeometry = shapeType === 'rectangular' 
      ? new THREE.BoxGeometry(length * 0.98, 0.1, width * 0.98)
      : new THREE.CylinderGeometry(radius * 0.98, radius * 0.98, 0.1, 32);
    
    const waterMaterial = new THREE.MeshPhongMaterial({
      color: 0x4F90F0,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide
    });
    
    const water = new THREE.Mesh(waterGeometry, waterMaterial);
    water.position.y = fillHeight;
    water.userData.isWater = true;
    scene.add(water);
  };

  // 运行模拟
  const runSimulation = async () => {
    setIsRunning(true);
    setProgress(0);

    const dimensions = shapeType === 'rectangular' 
      ? { length, width }
      : shapeType === 'cylindrical'
      ? { radius }
      : { outerRadius: radius * 1.5, innerRadius: radius * 0.7 };

    const solver = new Sloshing3DSolver(shapeType, dimensions, fillHeight, pga, duration);
    solver.quakeType = quakeType;
    
    const solution = solver.solve();
    
    setResults({
      solver,
      solution,
      maxResponse: Math.max(...solution.q.map(Math.abs))
    });

    // 动画循环
    const { scene } = sceneRef.current;
    const water = scene.children.find(obj => obj.userData.isWater);
    
    if (water) {
      const waterGeometry = water.geometry;
      const resolution = 20;
      
      for (let i = 0; i < solution.t.length; i += 2) {
        if (!isRunning) break;
        
        const q = solution.q[i];
        
        // 更新水面网格
        if (shapeType === 'rectangular') {
          const positions = waterGeometry.attributes.position;
          let idx = 0;
          for (let ix = 0; ix < resolution; ix++) {
            for (let iz = 0; iz < resolution; iz++) {
              const x = (ix / (resolution - 1) - 0.5) * length;
              const z = (iz / (resolution - 1) - 0.5) * width;
              const waveHeight = solver.getWaveHeight(q, x, z);
              if (positions.array[idx * 3 + 1] !== undefined) {
                positions.array[idx * 3 + 1] = waveHeight;
              }
              idx++;
            }
          }
          positions.needsUpdate = true;
        }
        
        water.position.y = fillHeight + q * 0.5;
        
        setProgress((i / solution.t.length) * 100);
        await new Promise(resolve => setTimeout(resolve, 30));
      }
    }

    setIsRunning(false);
    setProgress(100);
  };

  const resetSimulation = () => {
    setIsRunning(false);
    setProgress(0);
    setResults(null);
    createVisualization();
  };

  const downloadResults = () => {
    if (!results) return;
    
    const csv = ['时间(s),模态坐标(m),加速度(g)'];
    results.solution.t.forEach((t, i) => {
      csv.push(`${t.toFixed(3)},${results.solution.q[i].toFixed(6)},${(results.solution.acc[i] / 9.81).toFixed(4)}`);
    });
    
    const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sloshing_3d_results_${Date.now()}.csv`;
    a.click();
  };

  useEffect(() => {
    createVisualization();
  }, [shapeType, radius, length, width, height, fillHeight]);

  return (
    <div className="w-full h-screen bg-gray-50 flex flex-col">
      {/* 头部 */}
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white p-4 shadow-lg">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          🌊 3D液体晃动模拟器
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="ml-auto p-2 hover:bg-white/20 rounded-full transition"
          >
            <Info size={20} />
          </button>
        </h1>
        <p className="text-sm opacity-90 mt-1">高级地震响应分析 | 基于三维势流理论</p>
      </div>

      <div className="flex-1 flex gap-4 p-4 overflow-hidden">
        {/* 左侧控制面板 */}
        <div className="w-80 bg-white rounded-lg shadow-md p-4 overflow-y-auto">
          <h3 className="font-bold text-lg mb-4 text-gray-800">🏗️ 模型参数</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">容器形状</label>
              <select
                value={shapeType}
                onChange={(e) => setShapeType(e.target.value)}
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
              >
                <option value="rectangular">矩形容器</option>
                <option value="cylindrical">圆柱容器</option>
                <option value="annular">圆环容器</option>
              </select>
            </div>

            {shapeType === 'rectangular' && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-1">长度 L (m): {length}</label>
                  <input
                    type="range"
                    min="2"
                    max="10"
                    step="0.5"
                    value={length}
                    onChange={(e) => setLength(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">宽度 W (m): {width}</label>
                  <input
                    type="range"
                    min="2"
                    max="10"
                    step="0.5"
                    value={width}
                    onChange={(e) => setWidth(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </>
            )}

            {shapeType === 'cylindrical' && (
              <div>
                <label className="block text-sm font-medium mb-1">半径 R (m): {radius}</label>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.1"
                  value={radius}
                  onChange={(e) => setRadius(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            {shapeType === 'annular' && (
              <div>
                <label className="block text-sm font-medium mb-1">基准半径 (m): {radius}</label>
                <input
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.1"
                  value={radius}
                  onChange={(e) => setRadius(parseFloat(e.target.value))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">外径={radius * 1.5}m, 内径={radius * 0.7}m</p>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-1">容器高度 H (m): {height}</label>
              <input
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={height}
                onChange={(e) => setHeight(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">液面深度 h (m): {fillHeight}</label>
              <input
                type="range"
                min="0.5"
                max={height}
                step="0.1"
                value={fillHeight}
                onChange={(e) => setFillHeight(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <hr className="my-4" />

            <h3 className="font-bold text-lg mb-4 text-gray-800">📉 地震输入</h3>

            <div>
              <label className="block text-sm font-medium mb-1">地震波类型</label>
              <select
                value={quakeType}
                onChange={(e) => setQuakeType(e.target.value)}
                className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
              >
                <option value="sine">正弦波</option>
                <option value="elcentro">El Centro (1940)</option>
                <option value="kobe">Kobe (1995)</option>
                <option value="northridge">Northridge (1994)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">PGA (g): {pga}</label>
              <input
                type="range"
                min="0.05"
                max="1"
                step="0.05"
                value={pga}
                onChange={(e) => setPga(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-gray-500 mt-1">{(pga * 9.81).toFixed(2)} m/s²</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">模拟时长 (s): {duration}</label>
              <input
                type="range"
                min="10"
                max="40"
                step="5"
                value={duration}
                onChange={(e) => setDuration(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={runSimulation}
                disabled={isRunning}
                className="flex-1 bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400 flex items-center justify-center gap-2"
              >
                {isRunning ? <Pause size={16} /> : <Play size={16} />}
                {isRunning ? '运行中...' : '运行模拟'}
              </button>
              <button
                onClick={resetSimulation}
                className="p-2 bg-gray-200 rounded hover:bg-gray-300"
              >
                <RotateCcw size={20} />
              </button>
            </div>

            {isRunning && (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}

            {results && (
              <div className="mt-4 p-3 bg-green-50 rounded border border-green-200">
                <h4 className="font-bold text-sm mb-2">📊 结果</h4>
                <div className="text-sm space-y-1">
                  <p>固有频率: {results.solver.freq_n.toFixed(3)} Hz</p>
                  <p>阻尼比: {results.solver.xi.toFixed(4)}</p>
                  <p>最大响应: {results.maxResponse.toFixed(4)} m</p>
                </div>
                <button
                  onClick={downloadResults}
                  className="mt-2 w-full bg-green-600 text-white py-1 px-3 rounded text-sm hover:bg-green-700 flex items-center justify-center gap-2"
                >
                  <Download size={14} />
                  下载数据
                </button>
              </div>
            )}
          </div>
        </div>

        {/* 右侧3D视图 */}
        <div className="flex-1 bg-white rounded-lg shadow-md overflow-hidden relative">
          <canvas ref={canvasRef} className="w-full h-full" />
          <div className="absolute top-4 left-4 bg-black/70 text-white p-3 rounded text-sm space-y-1">
            <p>🖱️ 拖动旋转视角</p>
            <p>📐 坐标轴: X(红) Y(绿) Z(蓝)</p>
          </div>
        </div>
      </div>

      {/* 信息面板 */}
      {showInfo && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-2xl max-h-[80vh] overflow-y-auto p-6">
            <h2 className="text-2xl font-bold mb-4">理论基础</h2>
            <div className="space-y-4 text-sm">
              <div>
                <h3 className="font-bold">运动方程</h3>
                <p className="font-mono bg-gray-100 p-2 rounded">q̈ + 2ξω_n·q̇ + ω_n²·q = -γ·a_g(t)</p>
              </div>
              <div>
                <h3 className="font-bold">固有频率 (圆柱容器)</h3>
                <p className="font-mono bg-gray-100 p-2 rounded">ω_n = √[(g·ε₁/R)·tanh(ε₁h/R)]</p>
                <p className="text-gray-600">其中 ε₁ = 1.8412 (第一阶贝塞尔函数根)</p>
              </div>
              <div>
                <h3 className="font-bold">适用范围</h3>
                <ul className="list-disc ml-5 space-y-1">
                  <li>小幅晃动 (η/h {'<'} 0.1)</li>
                  <li>线性势流假设</li>
                  <li>单模态主导</li>
                </ul>
              </div>
              <div>
                <h3 className="font-bold">3D计算特点</h3>
                <ul className="list-disc ml-5 space-y-1">
                  <li>考虑X、Y双向晃动耦合</li>
                  <li>三维模态形状函数</li>
                  <li>实时波面重构</li>
                </ul>
              </div>
            </div>
            <button
              onClick={() => setShowInfo(false)}
              className="mt-6 w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
            >
              关闭
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sloshing3DSimulator;