import React, { useEffect } from 'react'
import { createTheme, ThemeProvider, CssBaseline, Box, IconButton, Tooltip } from '@mui/material'
import HelpOutlineIcon from '@mui/icons-material/HelpOutline'
import { Onboarding } from './components/Onboarding'
import { HeaderBar } from './components/HeaderBar'
import { LeftSidebar } from './components/LeftSidebar'
import { RightSidebar } from './components/RightSidebar'
import { BottomPanel } from './components/BottomPanel'
import { Viewport } from './components/Viewport'
import { useAppStore } from './store/appStore'
import { Snackbar, Alert } from '@mui/material'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#3498db' },
    success: { main: '#27ae60' },
    warning: { main: '#f39c12' },
    error: { main: '#e74c3c' }
  },
  typography: {
    fontFamily: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif'
  },
  components: {
    MuiButton: { defaultProps: { disableRipple: false } },
    MuiLink: { defaultProps: { underline: 'hover' } }
  }
})

export default function App() {
  const connect = useAppStore(s => s.connect)
  const connectionError = useAppStore(s => s.connectionError)
  const [showOnboarding, setShowOnboarding] = React.useState(false)

  useEffect(() => {
    connect()
  }, [connect])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box display="grid" gridTemplateRows="auto 1fr auto" height="100vh">
        <HeaderBar />
        <Box display="grid" gridTemplateColumns="300px 1fr 340px" overflow="hidden">
          <LeftSidebar />
          <Viewport />
          <RightSidebar />
        </Box>
        <BottomPanel />
        <Box sx={{ position: 'fixed', top: 8, right: 8 }}>
          <Tooltip title="Onboarding">
            <IconButton color="primary" onClick={() => setShowOnboarding(true)} aria-label="open-onboarding">
              <HelpOutlineIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      <Snackbar open={connectionError} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        <Alert severity="warning" variant="filled">Connection lost. Reconnecting…</Alert>
      </Snackbar>
      <Onboarding open={showOnboarding} onClose={() => setShowOnboarding(false)} />
    </ThemeProvider>
  )
}

