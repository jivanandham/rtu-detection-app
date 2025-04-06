import React, { useState, useEffect } from 'react';
import { 
  Button, 
  TextField, 
  Box, 
  Typography, 
  Paper, 
  IconButton, 
  CircularProgress, 
  Alert, 
  Grid,
  Card,
  CardContent,
  CardMedia
} from '@mui/material';
import { styled } from '@mui/material/styles';
import Lightbox from './Lightbox';
import { Visibility, Delete, CloudUpload, CheckCircle } from '@mui/icons-material';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  textAlign: 'center',
  color: theme.palette.text.secondary,
  maxWidth: 1200,
  margin: '0 auto',
  borderRadius: 16,
  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
}));

const UploadArea = styled(Box)(({ theme }) => ({
  border: '2px dashed #ccc',
  borderRadius: 8,
  padding: theme.spacing(4),
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: theme.palette.primary.main,
    backgroundColor: 'rgba(0, 0, 0, 0.02)',
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  marginTop: theme.spacing(2),
  marginBottom: theme.spacing(2),
  minWidth: 150,
  '&.MuiButton-contained': {
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
}));

const ImageContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  gap: theme.spacing(4),
  marginTop: theme.spacing(4),
  flexWrap: 'wrap',
}));

const ImageCard = styled(Card)(({ theme }) => ({
  maxWidth: 300,
  transition: 'transform 0.2s',
  '&:hover': {
    transform: 'scale(1.02)',
  },
}));

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [rtuCount, setRtuCount] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [showLightbox, setShowLightbox] = useState({
    open: false,
    type: 'original',
    src: ''
  });
  const [buildingName, setBuildingName] = useState('');
  const [address, setAddress] = useState('');
  const [city, setCity] = useState('');
  const [state, setState] = useState('');
  const [zipCode, setZipCode] = useState('');
  const [showAddressForm, setShowAddressForm] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 5 * 1024 * 1024) { // 5MB limit
        setError('File size must be less than 5MB');
        return;
      }
      
      setFile(selectedFile);
      setError(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDetectRTU = async () => {
    if (!file) {
      setError('Please select an image first');
      return;
    }

    setError(null);
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to detect RTUs');
      }

      setProcessedImage(data.processed_image);
      setRtuCount(data.rtu_count);
      setShowAddressForm(true);
      setFile(null);
      setError(null);
    } catch (error) {
      setError(error.message);
      console.error('Error detecting RTUs:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleUpload = async () => {
    if (!buildingName || !address || !city || !state || !zipCode) {
      setError('Please fill in all address fields');
      return;
    }

    setError(null);
    setUploading(true);
    const formData = new FormData();
    formData.append('building_name', buildingName);
    formData.append('address', address);
    formData.append('city', city);
    formData.append('state', state);
    formData.append('zip_code', zipCode);
    formData.append('processed_image', processedImage);
    formData.append('rtu_count', rtuCount);

    try {
      const response = await fetch('http://localhost:8000/save_upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      setFile(null);
      setImagePreview(null);
      setProcessedImage(null);
      setRtuCount(null);
      setBuildingName('');
      setAddress('');
      setCity('');
      setState('');
      setZipCode('');
      setShowAddressForm(false);
      setError(null);
      
      alert('Image and address saved successfully!');
    } catch (error) {
      setError('Failed to save data. Please try again.');
      console.error('Error saving data:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleImageClick = (type, src) => {
    const imageUrl = type === 'processed' 
      ? `http://localhost:8000${src}`
      : src;
    
    setShowLightbox({
      open: true,
      type: type,
      src: imageUrl
    });
  };

  const handleCloseLightbox = () => {
    setShowLightbox({
      open: false,
      type: 'original',
      src: ''
    });
  };

  return (
    <StyledPaper>
      <Typography variant="h4" gutterBottom>
        RTU Detection Image Upload
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <UploadArea>
        <input
          accept="image/*"
          style={{ display: 'none' }}
          id="raised-button-file"
          type="file"
          onChange={handleFileChange}
        />
        <label htmlFor="raised-button-file">
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <CloudUpload sx={{ fontSize: 40, color: 'primary.main' }} />
            <Typography variant="h6" color="primary">
              {imagePreview ? 'Change Image' : 'Upload Image'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Drag and drop or click to select an image
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Max file size: 5MB
            </Typography>
          </Box>
        </label>
      </UploadArea>

      {imagePreview && (
        <ImageContainer>
          <ImageCard>
            <CardMedia
              component="img"
              height="300"
              image={imagePreview}
              alt="Preview"
              onClick={() => handleImageClick('original', imagePreview)}
              sx={{
                cursor: 'pointer',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'scale(1.05)',
                },
              }}
            />
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Original Image
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Click to view full size
              </Typography>
            </CardContent>
          </ImageCard>

          {processedImage && (
            <ImageCard>
              <CardMedia
                component="img"
                height="300"
                image={`http://localhost:8000${processedImage}`}
                alt="Processed"
                onClick={() => handleImageClick('processed', processedImage)}
                sx={{
                  cursor: 'pointer',
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'scale(1.05)',
                  },
                }}
              />
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Processed Image
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  RTU Count: {rtuCount}
                </Typography>
              </CardContent>
            </ImageCard>
          )}
        </ImageContainer>
      )}

      {showAddressForm && (
        <Box sx={{ mt: 4, p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
            Enter Building Details
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Building Name"
                value={buildingName}
                onChange={(e) => setBuildingName(e.target.value)}
                disabled={uploading}
                required
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Address"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                disabled={uploading}
                required
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="City"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                disabled={uploading}
                required
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="State"
                value={state}
                onChange={(e) => setState(e.target.value)}
                disabled={uploading}
                required
                variant="outlined"
              />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField
                fullWidth
                label="Zip Code"
                value={zipCode}
                onChange={(e) => setZipCode(e.target.value)}
                disabled={uploading}
                required
                variant="outlined"
              />
            </Grid>
          </Grid>

          <StyledButton
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={
              !buildingName ||
              !address ||
              !city ||
              !state ||
              !zipCode ||
              uploading
            }
            startIcon={uploading ? <CircularProgress size={24} /> : <CheckCircle />}
            fullWidth
            sx={{ mt: 2 }}
          >
            Save Building Details
          </StyledButton>
        </Box>
      )}

      <StyledButton
        variant="contained"
        onClick={handleDetectRTU}
        disabled={!file || uploading || showAddressForm}
        startIcon={uploading ? <CircularProgress size={24} /> : <CloudUpload />}
        fullWidth
        sx={{ mt: 4 }}
      >
        {showAddressForm ? 'Address Form Ready' : 'Detect RTUs'}
      </StyledButton>

      {showLightbox.open && (
        <Lightbox
          src={showLightbox.src}
          title={showLightbox.type === 'original' ? 'Original Image' : `Processed Image (RTU Count: ${rtuCount})`}
          onClose={handleCloseLightbox}
        />
      )}
    </StyledPaper>
  );
};

export default ImageUpload;
