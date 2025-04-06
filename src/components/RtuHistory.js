import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Divider, 
  IconButton,
  Drawer,
  Stack
} from '@mui/material';
import { 
  Image as ImageIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import ImageUpload from './ImageUpload';
import History from './History';

const RtuHistory = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [drawerOpen, setDrawerOpen] = useState(false);

    const handleImageClick = (imageData) => {
        setSelectedImage(imageData);
        setDrawerOpen(true);
    };

    const handleCloseDrawer = () => {
        setDrawerOpen(false);
        setSelectedImage(null);
    };

    return (
        <Box sx={{ mt: 2 }}>
            <Typography variant="h4" gutterBottom>
                RTU Detection & History
            </Typography>
            <Divider sx={{ my: 2 }} />
            
            {/* Image Upload Section */}
            <Paper sx={{ p: 2, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Upload New Image
                </Typography>
                <ImageUpload onImageClick={handleImageClick} />
            </Paper>

            {/* History Section */}
            <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                    Upload History
                </Typography>
                <History onImageClick={handleImageClick} />
            </Paper>

            {/* Image Viewer Drawer */}
            <Drawer
                anchor="right"
                open={drawerOpen}
                onClose={handleCloseDrawer}
                PaperProps={{
                    sx: {
                        width: '50%',
                        maxWidth: '600px',
                        minWidth: '300px',
                        borderRadius: '16px 0 0 16px',
                        boxShadow: '0 2px 4px -1px rgba(0, 0, 0, 0.1), 0 4px 5px 0 rgba(0, 0, 0, 0.05)',
                    },
                }}
            >
                <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">
                        Image Viewer
                    </Typography>
                    <IconButton onClick={handleCloseDrawer}>
                        <CloseIcon />
                    </IconButton>
                </Box>
                
                {selectedImage && (
                    <Box sx={{ p: 2 }}>
                        <Stack spacing={2}>
                            <Typography variant="subtitle1" gutterBottom>
                                {selectedImage.building_name}
                            </Typography>
                            <img
                                src={selectedImage.image_url}
                                alt={selectedImage.building_name}
                                style={{
                                    width: '100%',
                                    height: 'auto',
                                    borderRadius: '8px',
                                    boxShadow: '0 2px 4px -1px rgba(0, 0, 0, 0.1), 0 4px 5px 0 rgba(0, 0, 0, 0.05)',
                                }}
                            />
                            <Typography variant="body2" color="text.secondary">
                                RTU Count: {selectedImage.rtu_count}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Lead Score: {selectedImage.lead_score}/10
                            </Typography>
                        </Stack>
                    </Box>
                )}
            </Drawer>
        </Box>
    );
};

export default RtuHistory;
