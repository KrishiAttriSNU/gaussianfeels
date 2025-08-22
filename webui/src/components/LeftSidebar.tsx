import React from 'react'
import { Box, Divider, Typography, Button, Stack, Switch, FormControlLabel, Slider } from '@mui/material'
import { useAppStore } from '../store/appStore'

export function LeftSidebar() {
  const overlays = useAppStore(s => s.showOverlays)
  const toggle = useAppStore(s => s.toggleOverlay)
  const maxPoints = useAppStore(s => s.maxPoints)
  const setMaxPoints = useAppStore(s => s.setMaxPoints)
  const autoTune = useAppStore(s => s.autoTunePerformance)
  const toggleAuto = useAppStore(s => s.toggleAutoTune)
  const start = async () => {
    await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: 'feelsight', object: 'contactdb_rubber_duck', max_steps: 1000 })
    })
  }
  const stop = async () => {
    await fetch('/api/training/stop', { method: 'POST' })
  }
  const pause = async () => {
    await fetch('/api/training/pause', { method: 'POST' })
  }
  const resume = async () => {
    await fetch('/api/training/resume', { method: 'POST' })
  }
  const checkpoint = async () => {
    await fetch('/api/training/checkpoint', { method: 'POST' })
  }

  return (
    <Box sx={{ borderRight: '1px solid #e1e8ed', p: 2, height: '100%', overflow: 'auto' }}>
      <Typography variant="subtitle1" gutterBottom>Dataset Browser</Typography>
      <Box className="card" sx={{ mb: 2 }}>
        <Typography variant="body2">feelsight / contactdb_rubber_duck</Typography>
      </Box>
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Training Controls</Typography>
      <Stack direction="row" spacing={1}>
        <Button variant="contained" size="small" onClick={start}>Start</Button>
        <Button variant="outlined" size="small" onClick={stop} color="error">Stop</Button>
        <Button variant="outlined" size="small" onClick={pause}>Pause</Button>
        <Button variant="outlined" size="small" onClick={resume}>Resume</Button>
        <Button variant="outlined" size="small" onClick={checkpoint}>Checkpoint</Button>
      </Stack>
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Overlays</Typography>
      <FormControlLabel control={<Switch checked={overlays.rgb} onChange={() => toggle('rgb')} />} label="RGB" />
      <FormControlLabel control={<Switch checked={overlays.tactile} onChange={() => toggle('tactile')} />} label="Tactile" />
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Render Budget</Typography>
      <Typography variant="caption">Max points: {maxPoints.toLocaleString()}</Typography>
      <Slider size="small" min={1000} max={100000} step={1000} value={maxPoints} onChange={(_, v) => setMaxPoints(v as number)} />
      <FormControlLabel control={<Switch checked={autoTune} onChange={toggleAuto} />} label="Auto-tune by FPS" />
    </Box>
  )
}


