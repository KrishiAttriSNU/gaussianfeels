import React, { useMemo } from 'react'
import * as THREE from 'three'
import { useAppStore } from '../store/appStore'
import { extend, useFrame } from '@react-three/fiber'

const vertexShader = /* glsl */`
  attribute vec3 instancePosition;
  attribute vec3 instanceColor;
  attribute float instanceOpacity;
  attribute vec2 quadUv;
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vUv;
  void main() {
    vColor = instanceColor;
    vOpacity = instanceOpacity;
    vUv = quadUv;
    // Billboard quad in view space
    vec3 right = vec3(modelViewMatrix[0][0], modelViewMatrix[1][0], modelViewMatrix[2][0]);
    vec3 up = vec3(modelViewMatrix[0][1], modelViewMatrix[1][1], modelViewMatrix[2][1]);
    float size = 0.02; // base size; could be scaled per instance
    vec3 offset = (quadUv - 0.5) * 2.0;
    vec3 worldPos = instancePosition + right * offset.x * size + up * offset.y * size;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(worldPos, 1.0);
  }
`

const fragmentShader = /* glsl */`
  precision mediump float;
  varying vec3 vColor;
  varying float vOpacity;
  varying vec2 vUv;
  void main() {
    // Gaussian falloff in quad space (circular)
    vec2 p = (vUv - 0.5) * 2.0;
    float r2 = dot(p, p);
    float alpha = exp(-r2 * 4.0) * vOpacity;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(vColor, alpha);
  }
`

export function GaussianSplat() {
  const gaussians = useAppStore(s => s.gaussians)
  const maxPoints = useAppStore(s => s.maxPoints)

  const { geometry, material, count } = useMemo(() => {
    const geom = new THREE.InstancedBufferGeometry()
    // Quad geometry with 4 verts (two triangles) and UVs mapped to [0,1]
    const quad = new Float32Array([
      -0.5, -0.5, 0,   0, 0,
       0.5, -0.5, 0,   1, 0,
       0.5,  0.5, 0,   1, 1,
      -0.5,  0.5, 0,   0, 1,
    ])
    const indices = new Uint16Array([0,1,2, 2,3,0])
    const pos = new Float32Array(quad.length / 5 * 3)
    const uvs = new Float32Array(quad.length / 5 * 2)
    for (let i=0, vi=0, ui=0; i<quad.length; i+=5) {
      pos[vi++] = quad[i]; pos[vi++] = quad[i+1]; pos[vi++] = quad[i+2]
      uvs[ui++] = quad[i+3]; uvs[ui++] = quad[i+4]
    }
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geom.setAttribute('quadUv', new THREE.BufferAttribute(uvs, 2))
    geom.setIndex(new THREE.BufferAttribute(indices, 1))

    const mat = new THREE.ShaderMaterial({
      vertexShader, fragmentShader, transparent: true, depthWrite: false
    })
    return { geometry: geom, material: mat, count: 0 }
  }, [])

  // Update per-instance attributes from store
  useMemo(() => {
    if (!gaussians) return
    const n = Math.min(maxPoints, gaussians.points.length / 3)
    const instancePosition = new Float32Array(n * 3)
    const instanceColor = new Float32Array(n * 3)
    const instanceOpacity = new Float32Array(n)
    instanceOpacity.fill(1)
    instancePosition.set(gaussians.points.subarray(0, n*3))
    instanceColor.set(gaussians.colors.subarray(0, n*3))
    geometry.setAttribute('instancePosition', new THREE.InstancedBufferAttribute(instancePosition, 3))
    geometry.setAttribute('instanceColor', new THREE.InstancedBufferAttribute(instanceColor, 3))
    geometry.setAttribute('instanceOpacity', new THREE.InstancedBufferAttribute(instanceOpacity, 1))
    // @ts-ignore
    geometry.instanceCount = n
  }, [gaussians, geometry, maxPoints])

  // @ts-ignore
  return <instancedMesh args={[geometry as any, material as any, (geometry as any).instanceCount || 0]} />
}




