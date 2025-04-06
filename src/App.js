import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { Box, Container, Typography, AppBar, Toolbar, Button } from '@mui/material';
import './App.css';
import Sidebar from './components/Sidebar';
import RtuHistory from './components/RtuHistory';
import Maps from './components/Maps';
import Home from './components/Home';
import GoogleMapsAPITest from './components/GoogleMapsAPITest';

function App() {
  const [address, setAddress] = useState('');

  const handleLocationSelect = (location) => {
    setAddress(location.address);
  };

  return (
    <Router>
      <Box sx={{ display: 'flex' }}>
        <Sidebar />
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Container maxWidth="lg">
            <AppBar position="static">
              <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                  RTU Detection App
                </Typography>
                <Button color="inherit" component={Link} to="/">
                  Home
                </Button>
                <Button color="inherit" component={Link} to="/rtu">
                  RTU
                </Button>
                <Button color="inherit" component={Link} to="/map">
                  Maps
                </Button>
                <Button color="inherit" component={Link} to="/test-api">
                  Test API
                </Button>
              </Toolbar>
            </AppBar>
            <Routes>
              <Route path="/" element={<Navigate to="/home" replace />} />
              <Route path="/home" element={<Home />} />
              <Route path="/rtu" element={<RtuHistory />} />
              <Route path="/map" element={<Maps onLocationSelect={handleLocationSelect} />} />
              <Route path="/about" element={<About />} />
              <Route path="/help" element={<Help />} />
              <Route path="/test-api" element={<GoogleMapsAPITest />} />
            </Routes>
          </Container>
        </Box>
      </Box>
    </Router>
  );
}

const About = () => (
  <Box>
    <Typography variant="h4" gutterBottom>
      About RTU Detection System
    </Typography>
    <Typography variant="body1" paragraph>
      This application helps detect and manage RTU (Roof Top Unit) installations using image processing technology.
    </Typography>
    <Typography variant="body1" paragraph>
      Features:
      <ul>
        <li>Image upload and processing</li>
        <li>RTU detection using AI</li>
        <li>Location mapping integration</li>
        <li>Upload history management</li>
      </ul>
    </Typography>
  </Box>
);

const Help = () => (
  <Box>
    <Typography variant="h4" gutterBottom>
      Help and Documentation
    </Typography>
    <Typography variant="body1" paragraph>
      <strong>How to Use:</strong>
      <ol>
        <li>Upload an image containing RTU installations</li>
        <li>Wait for the AI to process and detect RTUs</li>
        <li>Enter location details for the installation</li>
        <li>View processed images in the history</li>
      </ol>
    </Typography>
  </Box>
);

export default App;
