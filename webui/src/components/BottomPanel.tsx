import React from 'react'
import { Box, Slider, Stack, Typography } from '@mui/material'
import { useEffect, useRef, useState } from 'react'
import { useAppStore } from '../store/appStore'

export function BottomPanel() {
  const metrics = useAppStore(s => s.metrics)
  const [losses, setLosses] = useState<{ steps: number[]; map_losses: number[] }>({ steps: [], map_losses: [] })
  const [frame, setFrame] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const tick = async () => {
      try {
        const r = await fetch('/api/losses')
        const d = await r.json()
        setLosses({ steps: d.steps ?? [], map_losses: d.map_losses ?? [] })
      } catch {}
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    // Draw simple sparkline
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    const w = c.width, h = c.height
    ctx.clearRect(0, 0, w, h)
    const data = losses.map_losses
    if (!data || data.length < 2) return
    const min = Math.min(...data), max = Math.max(...data)
    const norm = (v: number) => max === min ? h/2 : h - ((v - min) / (max - min)) * h
    ctx.strokeStyle = '#3498db'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * w
      const y = norm(v)
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    })
    ctx.stroke()
  }, [losses])

  return (
    <Box sx={{ borderTop: '1px solid #e1e8ed', p: 1 }}>
      <Stack direction="row" spacing={2} alignItems="center">
        <Typography variant="caption">Frame</Typography>
        <Slider size="small" value={frame} onChange={(_, v) => setFrame(v as number)} min={0} max={Math.max(metrics.step ?? 0, 1)} sx={{ width: 300 }} />
        <Typography variant="caption">Loss: {losses.map_losses.at(-1)?.toFixed?.(4) ?? '0.0000'}</Typography>
        <canvas ref={canvasRef} width={160} height={30} style={{ display: 'block' }} />
      </Stack>
    </Box>
  )
}


