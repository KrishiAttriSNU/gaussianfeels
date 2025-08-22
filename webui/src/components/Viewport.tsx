import React, { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { Box } from '@mui/material'
import { useAppStore } from '../store/appStore'
import { GaussianSplat } from '../render/GaussianSplat'

function Points() {
  const gaussians = useAppStore(s => s.gaussians)
  const fetchGaussians = useAppStore(s => s.fetchGaussians)
  const maxPoints = useAppStore(s => s.maxPoints)
  const ref = useRef<THREE.Points>(null)

  useEffect(() => {
    fetchGaussians()
    const id = setInterval(fetchGaussians, 1000)
    return () => clearInterval(id)
  }, [fetchGaussians])

  const geometry = useMemo(() => new THREE.BufferGeometry(), [])
  const material = useMemo(() => new THREE.PointsMaterial({ size: 0.02, vertexColors: true }), [])

  useEffect(() => {
    if (!gaussians) return
    const count = Math.min(maxPoints, gaussians.points.length / 3)
    const positions = gaussians.points.subarray(0, count * 3)
    const colors = gaussians.colors.subarray(0, count * 3)
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.computeBoundingSphere()
  }, [gaussians, geometry, maxPoints])

  return <points ref={ref} args={[geometry, material]} />
}

function CameraRig() {
  const { camera } = useThree()
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime() * 0.1
    camera.position.x = Math.cos(t) * 2.5
    camera.position.z = Math.sin(t) * 2.5
    camera.lookAt(0, 0, 0)
  })
  return null
}

export function Viewport() {
  const show = useAppStore(s => s.showOverlays)
  const fetchFrame = useAppStore(s => s.fetchFrame)
  const frame = useAppStore(s => s.currentFrame)
  const frameData = useAppStore(s => s.frameData)

  useEffect(() => {
    fetchFrame()
    const id = setInterval(fetchFrame, 1000)
    return () => clearInterval(id)
  }, [fetchFrame, frame])

  return (
    <Box sx={{ position: 'relative' }}>
      <Canvas camera={{ position: [0, 0, 2.5], fov: 60 }} onPointerDown={(e) => {
        // Basic nearest-point picking
        const state = useAppStore.getState()
        const g = state.gaussians
        if (!g) return
        const mouse = new THREE.Vector2((e.nativeEvent.offsetX / (e.target as HTMLCanvasElement).clientWidth) * 2 - 1,
                                        -(e.nativeEvent.offsetY / (e.target as HTMLCanvasElement).clientHeight) * 2 + 1)
        const raycaster = new THREE.Raycaster()
        const { camera } = (e as any).camera ? (e as any) : { camera: undefined }
        if (!camera) return
        raycaster.setFromCamera(mouse, camera)
        const positions = g.points
        let best = { i: -1, d: Infinity }
        const tmp = new THREE.Vector3()
        for (let i = 0; i < Math.min(state.maxPoints, positions.length / 3); i++) {
          tmp.set(positions[i*3], positions[i*3+1], positions[i*3+2])
          const dist = raycaster.ray.distanceToPoint(tmp)
          if (dist < best.d) best = { i, d: dist }
        }
        state.setSelectedIndex(best.i >= 0 ? best.i : null)
      }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[1, 1, 1]} intensity={0.8} />
        <axesHelper args={[0.5]} />
        <GaussianSplat />
        <CameraRig />
      </Canvas>
      {/* Overlay panels */}
      {show.rgb && frameData?.rgb_images && (
        <Box sx={{ position: 'absolute', right: 8, bottom: 8, bgcolor: 'rgba(0,0,0,0.4)', p: 1, borderRadius: 1 }}>
          <img src={Object.values(frameData.rgb_images)[0]} alt="rgb" style={{ width: 160, height: 'auto' }} />
        </Box>
      )}
      {show.tactile && frameData?.tactile_images && (
        <Box sx={{ position: 'absolute', right: 8, bottom: 176, bgcolor: 'rgba(0,0,0,0.4)', p: 1, borderRadius: 1 }}>
          <img src={Object.values(frameData.tactile_images)[0]} alt="tactile" style={{ width: 160, height: 'auto' }} />
        </Box>
      )}
    </Box>
  )
}


