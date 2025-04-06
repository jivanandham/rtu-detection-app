import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, Alert, CircularProgress } from '@mui/material';
import { testGoogleMapsAPI } from '../utils/testGoogleMapsAPI';

const GoogleMapsAPITest = () => {
  const [loading, setLoading] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [apiKey, setApiKey] = useState('');

  const handleTestAPI = async () => {
    if (!apiKey) {
      setTestResult({
        success: false,
        message: 'Please enter an API key',
      });
      return;
    }

    setLoading(true);
    try {
      const result = await testGoogleMapsAPI(apiKey);
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        message: 'Failed to test API key',
        error: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 2, maxWidth: 600, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        Google Maps API Key Tester
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Typography variant="body1" gutterBottom>
          Enter your Google Maps API key to test if it's working correctly:
        </Typography>
        <input
          type="text"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          style={{
            width: '100%',
            padding: '8px',
            marginBottom: '16px',
            borderRadius: '4px',
            border: '1px solid #ccc',
          }}
        />
        <Button
          variant="contained"
          onClick={handleTestAPI}
          disabled={loading}
          sx={{ width: '100%' }}
        >
          {loading ? (
            <CircularProgress size={24} color="inherit" />
          ) : (
            'Test API Key'
          )}
        </Button>
      </Box>

      {testResult && (
        <Alert
          severity={testResult.success ? 'success' : 'error'}
          sx={{ mt: 2 }}
        >
          {testResult.success ? (
            <>
              <strong>API Key is working!</strong><br />
              {testResult.message}
            </>
          ) : (
            <>
              <strong>API Key failed:</strong><br />
              {testResult.error || testResult.message}
            </>
          )}
        </Alert>
      )}
    </Box>
  );
};

export default GoogleMapsAPITest;
