import React from 'react'
import { AppBar, Toolbar, Typography, Box, Chip, Stack, Button } from '@mui/material'
import { useAppStore } from '../store/appStore'

export function HeaderBar() {
  const connected = useAppStore(s => s.connected)
  const metrics = useAppStore(s => s.metrics)

  return (
    <AppBar position="static" color="default" elevation={1}>
      <Toolbar>
        <Typography variant="h6" sx={{ mr: 2 }}>GaussianFeels</Typography>
        <Chip label={connected ? 'Live' : 'Offline'} color={connected ? 'success' : 'default'} size="small" />
        <Box sx={{ flex: 1 }} />
        <Box display="flex" gap={2}>
          <Typography variant="body2">Step: {metrics.step ?? 0}</Typography>
          <Typography variant="body2">FPS: {(metrics.recent_fps ?? 0).toFixed(1)}</Typography>
          <Typography variant="body2">GPU: {(metrics.gpu_utilization ?? 0).toFixed(1)}%</Typography>
        </Box>
      </Toolbar>
    </AppBar>
  )
}


