import React, { useState, useEffect } from 'react';
import { styled } from '@mui/material/styles';

const LightboxOverlay = styled('div')(({ theme }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  width: '100vw',
  height: '100vh',
  backgroundColor: 'rgba(0, 0, 0, 0.9)',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  zIndex: 9999,
  padding: theme.spacing(2),
}));

const LightboxContent = styled('div')(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: theme.spacing(2),
  maxWidth: '90vw',
  maxHeight: '90vh',
  overflow: 'auto',
}));

const LightboxImage = styled('img')(({ theme }) => ({
  maxWidth: '100%',
  maxHeight: '100%',
  objectFit: 'contain',
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[8],
  transition: 'transform 0.3s ease',
  cursor: 'zoom-out',
  '&:hover': {
    transform: 'scale(1.02)',
  },
}));

const CloseButton = styled('button')(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  backgroundColor: 'rgba(255, 255, 255, 0.2)',
  color: 'white',
  border: 'none',
  borderRadius: '50%',
  width: '32px',
  height: '32px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
}));

const ImageTitle = styled('div')(({ theme }) => ({
  color: 'white',
  fontSize: '1.2rem',
  fontWeight: 'bold',
  textAlign: 'center',
  padding: theme.spacing(1),
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  borderRadius: theme.shape.borderRadius,
  width: '100%',
  maxWidth: '600px',
}));

const Lightbox = ({ src, title, onClose }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleImageClick = () => {
    setIsFullscreen(!isFullscreen);
  };

  return (
    <LightboxOverlay>
      <CloseButton onClick={onClose}>
        ✕
      </CloseButton>
      <LightboxContent>
        <ImageTitle>{title}</ImageTitle>
        <LightboxImage
          src={src}
          alt={title}
          onClick={handleImageClick}
          style={{
            width: isFullscreen ? '100vw' : 'auto',
            height: isFullscreen ? '100vh' : 'auto',
            objectFit: isFullscreen ? 'cover' : 'contain',
            cursor: isFullscreen ? 'default' : 'zoom-out',
          }}
        />
      </LightboxContent>
    </LightboxOverlay>
  );
};

export default Lightbox;
