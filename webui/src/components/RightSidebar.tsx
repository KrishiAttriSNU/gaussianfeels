import React from 'react'
import { Box, Typography, Divider, Stack, Button, TextField } from '@mui/material'
import { useAppStore } from '../store/appStore'

export function RightSidebar() {
  const metrics = useAppStore(s => s.metrics)
  const selected = useAppStore(s => s.selectedIndex)
  const gaussians = useAppStore(s => s.gaussians)
  const [diag, setDiag] = React.useState<any>(null)
  const [sensors, setSensors] = React.useState<any[]>([])
  const [feedback, setFeedback] = React.useState('')

  React.useEffect(() => {
    const tick = async () => {
      try {
        const r = await fetch('/api/sensors/status')
        const d = await r.json()
        setSensors(d.sensors ?? [])
      } catch {}
      try {
        const r2 = await fetch('/api/diagnostics')
        const d2 = await r2.json()
        setDiag(d2)
      } catch {}
    }
    tick()
    const id = setInterval(tick, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <Box sx={{ borderLeft: '1px solid #e1e8ed', p: 2, height: '100%', overflow: 'auto' }}>
      <Typography variant="subtitle1" gutterBottom>Metrics</Typography>
      <Typography variant="body2">Loss: {(metrics.recent_map_loss ?? 0).toFixed(6)}</Typography>
      <Typography variant="body2">Gaussians: {(metrics.num_gaussians ?? 0).toLocaleString()}</Typography>
      <Typography variant="body2">Memory: {(metrics.memory_usage_mb ?? 0).toFixed(1)} MB</Typography>
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Sensor Status</Typography>
      <Stack spacing={0.5}>
        {sensors.map((s, i) => (
          <Typography key={i} variant="caption">{s.name ?? 'sensor'}: {s.status ?? 'unknown'}</Typography>
        ))}
        {sensors.length === 0 && <Typography variant="caption">No sensors</Typography>}
        <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
          <Button size="small" variant="outlined" onClick={async ()=>{await fetch('/api/sensors/start', {method:'POST'})}}>Start</Button>
          <Button size="small" variant="outlined" onClick={async ()=>{await fetch('/api/sensors/stop', {method:'POST'})}}>Stop</Button>
        </Stack>
      </Stack>
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Diagnostics</Typography>
      <Typography variant="caption">Device: {diag?.device ?? 'n/a'} | GPUs: {diag?.gpu_count ?? 0}</Typography>
      <Typography variant="caption">Step: {diag?.trainer_step ?? 0} | Map avg: {(diag?.avg_map_time_ms ?? 0).toFixed(1)} ms | FPS: {(diag?.recent_fps ?? 0).toFixed(1)}</Typography>
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Selection</Typography>
      {selected != null && gaussians ? (
        <>
          <Typography variant="body2">Index: {selected}</Typography>
          <Typography variant="body2">Pos: [
            {gaussians.points[selected*3]?.toFixed(3)},
            {gaussians.points[selected*3+1]?.toFixed(3)},
            {gaussians.points[selected*3+2]?.toFixed(3)}
          ]</Typography>
        </>
      ) : (
        <Typography variant="caption">None</Typography>
      )}
      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle1" gutterBottom>Feedback</Typography>
      <Stack direction="row" spacing={1}>
        <TextField size="small" placeholder="Leave feedback" value={feedback} onChange={e=>setFeedback(e.target.value)} fullWidth inputProps={{ 'aria-label': 'feedback-input' }}/>
        <Button size="small" variant="contained" onClick={async ()=>{ if(!feedback) return; await fetch('/api/feedback',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: feedback})}); setFeedback('') }}>Send</Button>
      </Stack>
    </Box>
  )
}


