import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography, CircularProgress, Alert, TextField, Button, Link, Dialog, DialogTitle, DialogContent, DialogActions, Grid } from '@mui/material';
import axios from 'axios';
import html2canvas from 'html2canvas';
import SaveIcon from '@mui/icons-material/Save';

// Ensure initMap is available globally for Google Maps
window.initMap = () => {
  // This will be called by Google Maps when it's ready
  console.log('Google Maps API initialized');
};

const Maps = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [locations, setLocations] = useState([]);
  const [map, setMap] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [geocoder, setGeocoder] = useState(null);
  const [searchAddress, setSearchAddress] = useState('');
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [detectionLoading, setDetectionLoading] = useState(false);
  const [detectionResults, setDetectionResults] = useState(null);
  const [detectionError, setDetectionError] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionDialogOpen, setDetectionDialogOpen] = useState(false);
  const [screenshotPreview, setScreenshotPreview] = useState(null);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [mapsLoaded, setMapsLoaded] = useState(false);
  const [mapsError, setMapsError] = useState(null);
  const [locationFormOpen, setLocationFormOpen] = useState(false);
  const [selectedLocationData, setSelectedLocationData] = useState({
    building_name: '',
    address: '',
    city: '',
    state: '',
    zip_code: '',
    lat: null,
    lng: null,
    rtu_count: 0,
    lead_score: 0,
    geojson: null
  });
  const searchBoxRef = useRef(null);
  const mapContainerRef = useRef(null);

  useEffect(() => {
    const loadGoogleMaps = async () => {
      return new Promise((resolve, reject) => {
        if (window.google && window.google.maps) {
          resolve(window.google.maps);
          return;
        }

        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.REACT_APP_GOOGLE_MAPS_API_KEY}&libraries=places&callback=initMap`;
        script.async = true;
        script.defer = true;

        script.onerror = () => {
          reject(new Error('Failed to load Google Maps API'));
        };

        script.onload = () => {
          if (window.google && window.google.maps) {
            resolve(window.google.maps);
          } else {
            reject(new Error('Google Maps API failed to initialize'));
          }
        };

        document.head.appendChild(script);
      });
    };

    const initializeMap = async (Maps) => {
      try {
        const mapElement = mapContainerRef.current;
        if (!mapElement) return;

        const defaultLocation = {
          lat: 40.4406,
          lng: -79.9959,
          zoom: 16
        };

        const mapOptions = {
          center: { lat: defaultLocation.lat, lng: defaultLocation.lng },
          zoom: defaultLocation.zoom,
          mapTypeId: 'roadmap',
          styles: [
            {
              featureType: 'poi',
              elementType: 'labels',
              stylers: [{ visibility: 'off' }],
            },
          ],
        };

        const newMap = new Maps.Map(mapElement, mapOptions);
        setMap(newMap);
        setGeocoder(new Maps.Geocoder());

        // Initialize Places Autocomplete
        const searchBox = new Maps.places.SearchBox(searchBoxRef.current);
        searchBox.setBounds(new Maps.LatLngBounds(
          new Maps.LatLng(30, -130),
          new Maps.LatLng(50, -60)
        ));

        // Listen for search box changes
        searchBox.addListener('places_changed', () => {
          const places = searchBox.getPlaces();
          if (places.length === 0) return;

          const place = places[0];
          if (!place.geometry || !place.geometry.location) return;

          const latLng = place.geometry.location;
          const newLocation = {
            lat: latLng.lat(),
            lng: latLng.lng(),
            address: place.formatted_address,
            building_name: place.name || 'Unknown Building',
            rtu_count: 0,
            lead_score: 0,
          };

          setSelectedLocation(newLocation);
          newMap.setCenter(latLng);
          newMap.setZoom(15);
        });

        // Add click handler to add new marker
        const handleMapClick = async (event) => {
          const lat = event.latLng.lat();
          const lng = event.latLng.lng();
          
          // First, get the address information
          const locationData = await reverseGeocode(lat, lng);
          if (locationData) {
            // Then create the marker
            const marker = new window.google.maps.Marker({
              position: { lat, lng },
              map: newMap,
              draggable: true,
            });

            // Add info window
            const infoWindow = new window.google.maps.InfoWindow({
              content: `
                <div style="padding: 10px;">
                  <h4>Location Details</h4>
                  <p><strong>Address:</strong> ${locationData.address}</p>
                  <p><strong>Coordinates:</strong> ${lat.toFixed(6)}, ${lng.toFixed(6)}</p>
                </div>
              `
            });
            infoWindow.open(newMap, marker);

            // Add marker to state
            setMarkers(prev => [...prev, marker]);

            // Add drag listener
            const handleDragEnd = () => {
              const newLat = marker.getPosition().lat();
              const newLng = marker.getPosition().lng();
              reverseGeocode(newLat, newLng).then(data => {
                if (data) {
                  infoWindow.setContent(`
                    <div style="padding: 10px;">
                      <h4>Location Details</h4>
                      <p><strong>Address:</strong> ${data.address}</p>
                      <p><strong>Coordinates:</strong> ${newLat.toFixed(6)}, ${newLng.toFixed(6)}</p>
                    </div>
                  `);
                }
              });
            };
            marker.addListener('dragend', handleDragEnd);

            // Update location data immediately
            setSelectedLocationData(prev => ({
              ...prev,
              address: locationData.address,
              city: locationData.city,
              state: locationData.state,
              zip_code: locationData.zip,
              lat,
              lng,
              place_id: locationData.place_id,
              types: locationData.types
            }));
          }
        };

        // Add click handler to map
        newMap.addListener('click', handleMapClick);

        // Helper function to update location info
        const updateLocationInfo = (latLng, marker, infoWindow, Maps) => {
          if (geocoder) {
            geocoder.geocode({ location: latLng }, (results, status) => {
              if (status === 'OK' && results[0]) {
                const locationInfo = {
                  lat: latLng.lat(),
                  lng: latLng.lng(),
                  address: results[0].formatted_address,
                  building_name: results[0].name || 'Unknown Building',
                  rtu_count: 0,
                  lead_score: 0,
                };
                setSelectedLocation(locationInfo);

                // Update info window content
                infoWindow.setContent(`
                  <div style="padding: 10px;">
                    <h4>${locationInfo.building_name || locationInfo.address}</h4>
                    <p>${locationInfo.address}</p>
                    <p>RTU Count: ${locationInfo.rtu_count}</p>
                    <p>Lead Score: ${getLeadScoreLabel(locationInfo.lead_score)}</p>
                    <p>Coordinates: ${locationInfo.lat.toFixed(6)}, ${locationInfo.lng.toFixed(6)}</p>
                  </div>
                `);
              } else {
                infoWindow.setContent(`
                  <div style="padding: 10px;">
                    <h4>Location Details</h4>
                    <p>Could not find location information</p>
                  </div>
                `);
              }
            });
          }
        };
      } catch (error) {
        console.error('Error initializing map:', error);
        throw error;
      }
    };

    const initMap = async () => {
      try {
        const Maps = await loadGoogleMaps();
        setMapsLoaded(true);
        setMapsError(null);
        await initializeMap(Maps);
      } catch (error) {
        console.error('Error initializing Google Maps:', error);
        setMapsLoaded(false);
        setMapsError(error.message);
      }
    };

    initMap();

    // Cleanup
    return () => {
      if (map) {
        markers.forEach(marker => marker.setMap(null));
        setMarkers([]);
        setMap(null);
      }
    };
  }, []);

  const fetchLocations = async () => {
    try {
      const response = await axios.get('http://localhost:8000/history');
      const locations = response.data.map(record => ({
        ...record,
        lat: parseFloat(record.lat || 40.7128),
        lng: parseFloat(record.lng || -74.0060),
        rtu_count: parseInt(record.rtu_count || 0),
        lead_score: parseInt(record.lead_score || 0),
      }));
      
      // Geocode any locations that don't have coordinates
      const locationsWithCoords = await Promise.all(locations.map(async (location) => {
        if (location.lat && location.lng) return location;
        
        try {
          const results = await geocoder.geocode({ address: location.address });
          if (results.status === 'OK' && results.results[0]) {
            const latLng = results.results[0].geometry.location;
            return {
              ...location,
              lat: latLng.lat(),
              lng: latLng.lng(),
            };
          }
          return location;
        } catch (error) {
          console.error('Error geocoding location:', error);
          return location;
        }
      }));

      setLocations(locationsWithCoords);
      addMarkers(locationsWithCoords);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching locations:', error);
      setError('Failed to fetch locations');
      setLoading(false);
    }
  };

  const addMarkers = (locations) => {
    if (!map) return;

    // Clear existing markers
    markers.forEach(marker => marker.setMap(null));

    // Add new markers
    const newMarkers = locations.map(location => {
      const marker = new window.google.maps.Marker({
        position: { lat: location.lat, lng: location.lng },
        map: map,
        icon: getMarkerIcon(location.lead_score),
      });

      // Add info window
      const infoWindow = new window.google.maps.InfoWindow({
        content: `
          <div style="padding: 10px;">
            <h4>${location.building_name || 'Unknown Building'}</h4>
            <p>${location.address || 'Unknown Address'}</p>
            <p>RTU Count: ${location.rtu_count}</p>
            <p>Lead Score: ${getLeadScoreLabel(location.lead_score)}</p>
          </div>
        `,
      });

      // Add click listener
      const handleClick = () => {
        infoWindow.open(map, marker);
      };
      marker.addListener('click', handleClick);

      return marker;
    });

    setMarkers(newMarkers);
  };

  const handleDetectRTUs = async () => {
    if (!map) {
      setDetectionError('Map is not initialized');
      return;
    }

    try {
      setIsDetecting(true);
      setDetectionError(null);
      setDetectionResults(null);

      // Get current map center
      const center = map.getCenter();
      
      // First get location details using reverse geocoding
      const locationData = await reverseGeocode(center.lat(), center.lng());
      if (locationData) {
        // Update location data with geocoding results
        setSelectedLocationData({
          building_name: '', // Leave empty for user to fill
          address: locationData.address,
          city: locationData.city,
          state: locationData.state,
          zip_code: locationData.zip,
          lat: center.lat(),
          lng: center.lng(),
          place_id: locationData.place_id,
          types: locationData.types
        });
      }

      // Capture the map screenshot
      const screenshot = await captureMapScreenshot();
      setScreenshotPreview(screenshot);

      // Send the screenshot to the backend for RTU detection
      const formData = new FormData();
      const blob = await convertImageToBlob(screenshot);
      formData.append('file', blob, 'map_screenshot.png');
      formData.append('latitude', center.lat());
      formData.append('longitude', center.lng());
      formData.append('address', locationData?.address || '');

      const response = await axios.post(`${process.env.REACT_APP_API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Handle the detection results
      const results = response.data;
      setDetectionResults({
        total_rtus: results.rtu_count,
        detections: results.detections || [],
        processed_image: results.processed_image,
        lead_score: results.lead_score || 0.5 // Default moderate score if not provided
      });
      setDetectionDialogOpen(true);

    } catch (error) {
      console.error('Error during RTU detection:', error);
      setDetectionError(error.response?.data?.error || 'Failed to detect RTUs');
    } finally {
      setIsDetecting(false);
    }
  };

  const captureMapScreenshot = async () => {
    try {
      const canvas = await html2canvas(mapContainerRef.current, {
        scale: 2, // Higher quality screenshot
        logging: false,
        useCORS: true,
      });

      return canvas.toDataURL('image/png');
    } catch (error) {
      console.error('Error capturing screenshot:', error);
      throw new Error('Failed to capture map screenshot');
    }
  };

  const convertImageToBlob = async (dataUrl) => {
    try {
      const response = await fetch(dataUrl);
      const blob = await response.blob();
      return blob;
    } catch (error) {
      console.error('Error converting image to blob:', error);
      throw new Error('Failed to convert image to blob');
    }
  };

  const handleSaveLocation = async () => {
    try {
      if (!detectionResults) {
        setError('Please run RTU detection first');
        return;
      }

      // Create FormData with all required fields
      const formData = new FormData();
      formData.append('building_name', selectedLocationData.building_name || 'Unknown Building');
      formData.append('address', selectedLocationData.address);
      formData.append('city', selectedLocationData.city || '');
      formData.append('state', selectedLocationData.state || '');
      formData.append('zip_code', selectedLocationData.zip_code || '');
      formData.append('processed_image', detectionResults.processed_image);
      formData.append('rtu_count', detectionResults.total_rtus || 0);

      // Send to backend
      const saveResponse = await axios.post(`${process.env.REACT_APP_API_URL}/save_upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Backend response:', saveResponse.data);

      if (saveResponse.data) {
        fetchLocations();
        setLocationFormOpen(false);
        setSelectedLocationData({
          building_name: '',
          address: '',
          city: '',
          state: '',
          zip_code: '',
          lat: null,
          lng: null,
          rtu_count: 0,
          lead_score: 0,
        });
        setDetectionDialogOpen(false);
      }

    } catch (error) {
      console.error('Error details:', {
        error: error,
        response: error.response,
        data: error.response?.data,
        status: error.response?.status,
        statusText: error.response?.statusText
      });

      let errorMessage = 'Failed to add location';
      if (error.response?.data?.detail) {
        // Format validation errors nicely
        const validationErrors = error.response.data.detail.map(err => `
          ${err.loc.join('.')}: ${err.msg}
        `).join('\n');
        errorMessage = `Validation errors:\n${validationErrors}`;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.response?.data) {
        errorMessage = JSON.stringify(error.response.data);
      }

      setError(errorMessage);
    }
  };

  const handleLocationFormClose = () => {
    setLocationFormOpen(false);
    setSelectedLocationData({
      building_name: '',
      address: '',
      city: '',
      state: '',
      zip_code: '',
      lat: null,
      lng: null,
      rtu_count: 0,
      lead_score: 0,
    });
  };

  const closeDetectionDialog = () => {
    setDetectionDialogOpen(false);
    setDetectionResults(null);
    setDetectionError(null);
  };

  const getMarkerIcon = (score) => {
    if (!score) return null;
    const parsedScore = parseInt(score, 10);
    if (parsedScore >= 0 && parsedScore <= 5) {
      return 'https://maps.google.com/mapfiles/ms/icons/red-dot.png';
    }
    if (parsedScore >= 6 && parsedScore <= 10) {
      return 'https://maps.google.com/mapfiles/ms/icons/orange-dot.png';
    }
    return 'https://maps.google.com/mapfiles/ms/icons/green-dot.png';
  };

  const getLeadScoreLabel = (score) => {
    if (!score) return 'N/A';
    const parsedScore = parseInt(score, 10);
    if (parsedScore >= 0 && parsedScore <= 5) return 'Bad';
    if (parsedScore >= 6 && parsedScore <= 10) return 'Fair';
    return 'Good';
  };

  const reverseGeocode = async (lat, lng) => {
    try {
      const url = `https://maps.googleapis.com/maps/api/geocode/json?latlng=${lat},${lng}&key=${process.env.REACT_APP_GOOGLE_MAPS_API_KEY}`;
      const response = await axios.get(url);
      if (response.data.results.length > 0) {
        const result = response.data.results[0];
        const addressComponents = result.address_components;
        return {
          address: result.formatted_address,
          city: addressComponents.find(c => c.types.includes('locality'))?.long_name || '',
          state: addressComponents.find(c => c.types.includes('administrative_area_level_1'))?.short_name || '',
          zip: addressComponents.find(c => c.types.includes('postal_code'))?.long_name || '',
          place_id: result.place_id,
          types: result.types
        };
      }
      return null;
    } catch (error) {
      console.error('Error in reverse geocoding:', error);
      return null;
    }
  };

  return (
    <Box sx={{ height: '100vh', width: '100%', position: 'relative' }}>
      {/* Location Form Dialog */}
      <Dialog open={locationFormOpen} onClose={handleLocationFormClose} maxWidth="sm" fullWidth>
        <DialogTitle>Add Location</DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Building Name"
              value={selectedLocationData.building_name}
              onChange={(e) => setSelectedLocationData(prev => ({
                ...prev,
                building_name: e.target.value
              }))}
              margin="normal"
            />
            <TextField
              fullWidth
              label="Address"
              value={selectedLocationData.address}
              disabled
              margin="normal"
            />
            <TextField
              fullWidth
              label="City"
              value={selectedLocationData.city}
              disabled
              margin="normal"
            />
            <TextField
              fullWidth
              label="State"
              value={selectedLocationData.state}
              disabled
              margin="normal"
            />
            <TextField
              fullWidth
              label="ZIP Code"
              value={selectedLocationData.zip_code}
              disabled
              margin="normal"
            />
            <Typography variant="subtitle2" sx={{ mt: 2 }}>
              RTU Detection Results:
            </Typography>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={6}>
                <Typography variant="body1">
                  RTUs Detected: {selectedLocationData.rtu_count}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="body1">
                  Lead Score: {getLeadScoreLabel(selectedLocationData.lead_score)}
                </Typography>
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleLocationFormClose} color="inherit">
            Cancel
          </Button>
          <Button onClick={handleSaveLocation} variant="contained" color="primary">
            Add Location
          </Button>
        </DialogActions>
      </Dialog>

      {!mapsLoaded && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
          }}
        >
          {mapsError ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {mapsError}
              <br />
              Please ensure you have a valid Google Maps API key and no ad blockers are blocking the API requests.
            </Alert>
          ) : (
            <CircularProgress />
          )}
        </Box>
      )}

      {mapsLoaded && (
        <>
          <Box 
            ref={mapContainerRef} 
            sx={{ 
              height: '100%', 
              width: '100%',
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0
            }}
          />
          
          {/* Search Box */}
          <Box
            ref={searchBoxRef}
            sx={{
              position: 'absolute',
              top: '1rem',
              left: '1rem',
              zIndex: 1,
              width: '300px',
              backgroundColor: 'white',
              padding: '0.5rem',
              borderRadius: '4px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
          >
            <TextField
              fullWidth
              placeholder="Search location..."
              variant="outlined"
              size="small"
            />
          </Box>

          {/* Add Location Button */}
          <Button
            variant="contained"
            color="primary"
            onClick={handleSaveLocation}
            sx={{
              position: 'absolute',
              bottom: '1rem',
              left: '1rem',
              zIndex: 1
            }}
          >
            Add Location
          </Button>

          {/* Detect RTUs Button */}
          <Button
            variant="contained"
            color="secondary"
            onClick={handleDetectRTUs}
            disabled={isDetecting || !map}
            sx={{
              position: 'absolute',
              bottom: '1rem',
              right: '1rem',
              zIndex: 1
            }}
          >
            {isDetecting ? (
              <CircularProgress size={24} />
            ) : (
              'Detect RTUs'
            )}
          </Button>

          {/* Detection Results Dialog */}
          <Dialog open={detectionDialogOpen} onClose={closeDetectionDialog} maxWidth="md" fullWidth>
            <DialogTitle>RTU Detection Results</DialogTitle>
            <DialogContent>
              {detectionResults ? (
                <Box sx={{ mt: 2 }}>
                  {/* Display original and processed images */}
                  <Box sx={{ display: 'flex', gap: '20px', mb: 3 }}>
                    <Box sx={{ flex: 1, textAlign: 'center' }}>
                      <Typography variant="h6" gutterBottom>
                        Original Map View
                      </Typography>
                      <img 
                        src={screenshotPreview} 
                        alt="Original map view" 
                        style={{ maxWidth: '100%', maxHeight: '300px' }}
                      />
                    </Box>
                    <Box sx={{ flex: 1, textAlign: 'center' }}>
                      <Typography variant="h6" gutterBottom>
                        Processed Image
                      </Typography>
                      {detectionResults.processed_image ? (
                        <img 
                          src={`${process.env.REACT_APP_API_URL}${detectionResults.processed_image}`} 
                          alt="Processed image with RTU detection" 
                          style={{ maxWidth: '100%', maxHeight: '300px' }}
                        />
                      ) : (
                        <Typography variant="body1">No processed image available</Typography>
                      )}
                    </Box>
                  </Box>

                  {/* Display detection results */}
                  <Typography variant="h6" gutterBottom>
                    Detection Summary
                  </Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      RTUs Detected: {detectionResults.total_rtus}
                    </Typography>
                    <Typography variant="subtitle1" gutterBottom>
                      Lead Score: {getLeadScoreLabel(detectionResults.lead_score)}
                    </Typography>
                  </Box>

                  {/* Display individual detections */}
                  {detectionResults.detections.length > 0 && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h6" gutterBottom>
                        Individual Detections
                      </Typography>
                      {detectionResults.detections.map((detection, index) => (
                        <Box
                          key={index}
                          sx={{
                            mb: 2,
                            p: 2,
                            border: '1px solid #ddd',
                            borderRadius: 1,
                            '&:hover': { border: '1px solid #007bff' }
                          }}
                        >
                          <Typography variant="subtitle2" gutterBottom>
                            RTU {index + 1}
                          </Typography>
                          <Typography variant="body2">
                            Confidence: {(detection.confidence * 100).toFixed(2)}%
                          </Typography>
                          <Typography variant="body2">
                            Location: ({detection.x}, {detection.y})
                          </Typography>
                        </Box>
                      ))}
                    </Box>
                  )}

                  {/* Location Form */}
                  <Box sx={{ mb: 3, p: 2, border: '1px solid #ddd', borderRadius: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      Location Details
                    </Typography>
                    <TextField
                      fullWidth
                      label="Building Name"
                      value={selectedLocationData.building_name}
                      onChange={(e) => setSelectedLocationData(prev => ({
                        ...prev,
                        building_name: e.target.value
                      }))}
                      margin="normal"
                    />
                    <TextField
                      fullWidth
                      label="Address"
                      value={selectedLocationData.address}
                      disabled
                      margin="normal"
                    />
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      <Grid item xs={6}>
                        <TextField
                          fullWidth
                          label="City"
                          value={selectedLocationData.city}
                          disabled
                          margin="normal"
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <TextField
                          fullWidth
                          label="State"
                          value={selectedLocationData.state}
                          disabled
                          margin="normal"
                        />
                      </Grid>
                      <Grid item xs={3}>
                        <TextField
                          fullWidth
                          label="ZIP Code"
                          value={selectedLocationData.zip_code}
                          disabled
                          margin="normal"
                        />
                      </Grid>
                    </Grid>
                  </Box>

                  {/* Save Location Button */}
                  <Box sx={{ textAlign: 'right' }}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleSaveLocation}
                      startIcon={<SaveIcon />}
                      disabled={detectionLoading}
                    >
                      Save Location
                    </Button>
                  </Box>
                </Box>
              ) : (
                <Typography variant="body1">No detection results available</Typography>
              )}
            </DialogContent>
          </Dialog>
        </>
      )}
    </Box>
  );
};

export default Maps;
