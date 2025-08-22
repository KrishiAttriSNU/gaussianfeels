import React from 'react'
import { Backdrop, Box, Button, Paper, Stack, Typography } from '@mui/material'

const steps = [
  {
    title: 'Welcome to GaussianFeels',
    body: 'This brief tour will help you start a training run and view overlays.'
  },
  {
    title: 'Left Sidebar',
    body: 'Use Start/Pause/Resume/Stop/Checkpoint controls. Toggle RGB/Tactile overlays and set render budget or enable auto-tune.'
  },
  {
    title: 'Viewport',
    body: 'The 3D view shows Gaussian splats. Click to select. Overlay panels show live RGB/Tactile frames.'
  },
  {
    title: 'Right Sidebar',
    body: 'View metrics, diagnostics, sensor status, selection info, and send feedback.'
  },
  {
    title: 'You are ready!',
    body: 'Start a training session to see live updates. You can replay this guide from Help ▶ Onboarding.'
  }
]

export function Onboarding({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [idx, setIdx] = React.useState(0)

  React.useEffect(() => {
    if (!open) setIdx(0)
  }, [open])

  const next = () => {
    if (idx < steps.length - 1) setIdx(idx + 1)
    else onClose()
  }

  if (!open) return null

  const step = steps[idx]

  return (
    <Backdrop open sx={{ zIndex: 2000, bgcolor: 'rgba(0,0,0,0.55)' }}>
      <Paper elevation={3} sx={{ maxWidth: 520, p: 3, m: 2 }}>
        <Stack spacing={1.5}>
          <Typography variant="h6">{step.title}</Typography>
          <Typography variant="body2">{step.body}</Typography>
          <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
            <Typography variant="caption">Step {idx + 1} / {steps.length}</Typography>
            <Stack direction="row" spacing={1}>
              <Button size="small" onClick={onClose}>Skip</Button>
              <Button size="small" variant="contained" onClick={next}>{idx === steps.length - 1 ? 'Finish' : 'Next'}</Button>
            </Stack>
          </Box>
        </Stack>
      </Paper>
    </Backdrop>
  )
}




