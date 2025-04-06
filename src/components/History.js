import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography,
  IconButton,
  Stack,
  Drawer,
  Paper,
  Button,
  TextField,
  Pagination,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Checkbox,
  TablePagination,
} from '@mui/material';
import { 
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Search as SearchIcon,
  Close as CloseIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  RotateLeft as RotateLeftIcon,
  RotateRight as RotateRightIcon,
} from '@mui/icons-material';
import axios from 'axios';

export default function History() {
  const [history, setHistory] = useState([]);
  const [search, setSearch] = useState('');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [imageIndex, setImageIndex] = useState(0);
  const [selectedRows, setSelectedRows] = useState([]);
  const [page, setPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState(0);

  const fetchHistory = async () => {
    try {
      const response = await axios.get('http://localhost:8000/history');
      console.log('History response:', response.data); // Debug log
      setHistory(response.data || []);
    } catch (error) {
      console.error('Error fetching history:', error);
      setHistory([]); // Set empty array if fetch fails
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSearch = (event) => {
    setSearch(event.target.value);
  };

  const handlePageChange = (event, newPage) => {
    setPage(newPage);
  };

  const handleRowsPerPageChange = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0); // Reset page when rowsPerPage changes
  };

  const handleSelectRow = (event, id) => {
    const selectedIndex = selectedRows.indexOf(id);
    let newSelected = [];

    if (selectedIndex === -1) {
      newSelected = newSelected.concat(selectedRows, id);
    } else if (selectedIndex === 0) {
      newSelected = newSelected.concat(selectedRows.slice(1));
    } else if (selectedIndex === selectedRows.length - 1) {
      newSelected = newSelected.concat(selectedRows.slice(0, -1));
    } else if (selectedIndex > 0) {
      newSelected = newSelected.concat(
        selectedRows.slice(0, selectedIndex),
        selectedRows.slice(selectedIndex + 1)
      );
    }

    setSelectedRows(newSelected);
  };

  const handleSelectAllRows = (event) => {
    if (event.target.checked) {
      const newSelected = filteredHistory.map((n) => n.id);
      setSelectedRows(newSelected);
      return;
    }
    setSelectedRows([]);
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this record? This action cannot be undone.')) return;

    try {
      await axios.delete(`http://localhost:8000/history/${id}`);
      fetchHistory();
      // Clear selected rows if the deleted item was selected
      setSelectedRows(prev => prev.filter(rowId => rowId !== id));
    } catch (error) {
      console.error('Error deleting record:', error);
      alert('Failed to delete record. Please try again.');
    }
  };

  const handleDownload = async () => {
    if (selectedRows.length === 0) {
      alert('Please select at least one record to download');
      return;
    }

    try {
      for (const id of selectedRows) {
        const response = await axios.get(`http://localhost:8000/history/${id}/report`, {
          responseType: 'blob',
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `report_${id}.pdf`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      console.error('Error downloading reports:', error);
      alert('Failed to download reports. Please try again.');
    }
  };

  const handleOpenDrawer = (record) => {
    setSelectedRecord(record);
    setDrawerOpen(true);
    setImageIndex(0);
    setZoom(1);
    setRotation(0);
  };

  const handleCloseDrawer = () => {
    setDrawerOpen(false);
    setSelectedRecord(null);
    setImageIndex(0);
    setZoom(1);
    setRotation(0);
  };

  const handlePrevImage = () => {
    if (imageIndex > 0) {
      setImageIndex(prev => prev - 1);
    }
  };

  const handleNextImage = () => {
    if (selectedRecord && imageIndex < (selectedRecord.processed_image ? 1 : 0)) {
      setImageIndex(prev => prev + 1);
    }
  };

  const handleZoomIn = () => {
    if (zoom < 3) {
      setZoom(prev => prev + 0.5);
    }
  };

  const handleZoomOut = () => {
    if (zoom > 0.5) {
      setZoom(prev => prev - 0.5);
    }
  };

  const handleRotateLeft = () => {
    setRotation(prev => (prev - 90 + 360) % 360);
  };

  const handleRotateRight = () => {
    setRotation(prev => (prev + 90) % 360);
  };

  const getLeadScore = (rtuCount) => {
    if (!rtuCount) return null;
    const parsedCount = parseInt(rtuCount, 10);
    if (parsedCount <= 0) return 0;
    if (parsedCount >= 15) return 15;
    return parsedCount;
  };

  const getLeadScoreColor = (score) => {
    if (!score) return 'text.secondary';
    const parsedScore = parseInt(score, 10);
    if (parsedScore >= 0 && parsedScore <= 5) return '#d32f2f'; // Red for bad
    if (parsedScore >= 6 && parsedScore <= 10) return '#f57c00'; // Orange for fair
    if (parsedScore >= 11 && parsedScore <= 15) return '#43a047'; // Green for good
    return '#1e88e5'; // Blue for excellent
  };

  const getLeadScoreLabel = (score) => {
    if (!score) return 'N/A';
    const parsedScore = parseInt(score, 10);
    if (parsedScore >= 0 && parsedScore <= 5) return 'Bad';
    if (parsedScore >= 6 && parsedScore <= 10) return 'Fair';
    if (parsedScore >= 11 && parsedScore <= 15) return 'Good';
    return 'Excellent';
  };

  const filteredHistory = history.filter(
    (record) =>
      record.building_name.toLowerCase().includes(search.toLowerCase()) ||
      record.address.toLowerCase().includes(search.toLowerCase())
  ).map(record => ({
    ...record,
    lead_score: getLeadScore(record.rtu_count)
  }));

  const emptyRows = page > 0 ? Math.max(0, (1 + page) * rowsPerPage - filteredHistory.length) : 0;

  return (
    <Box sx={{ p: 2 }}>
      {/* Header Section */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          History
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          View past RTU detection results and reports
        </Typography>
      </Box>

      {/* Search Bar */}
      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          placeholder="Search by building name or address..."
          value={search}
          onChange={handleSearch}
          InputProps={{
            startAdornment: (
              <SearchIcon sx={{ color: 'text.secondary', mr: 1 }} />
            ),
          }}
        />
      </Box>

      {/* Table Container */}
      <Paper sx={{ mb: 2 }}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={selectedRows.length > 0 && selectedRows.length < filteredHistory.length}
                    checked={filteredHistory.length > 0 && selectedRows.length === filteredHistory.length}
                    onChange={handleSelectAllRows}
                  />
                </TableCell>
                <TableCell>Building Name</TableCell>
                <TableCell>Address</TableCell>
                <TableCell>RTU Count</TableCell>
                <TableCell>Lead Score</TableCell>
                <TableCell>Date</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredHistory.map((record) => (
                <TableRow
                  key={record.id}
                  hover
                  onClick={() => handleOpenDrawer(record)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={selectedRows.indexOf(record.id) !== -1}
                      onChange={(event) => handleSelectRow(event, record.id)}
                    />
                  </TableCell>
                  <TableCell>{record.building_name || 'N/A'}</TableCell>
                  <TableCell>{record.address || 'N/A'}</TableCell>
                  <TableCell>{record.rtu_count || 'N/A'}</TableCell>
                  <TableCell>
                    <Typography 
                      variant="body2" 
                      sx={{
                        color: getLeadScoreColor(record.lead_score),
                        fontWeight: 'bold',
                      }}
                    >
                      {getLeadScoreLabel(record.lead_score)}
                    </Typography>
                  </TableCell>
                  <TableCell>{record.created_at ? new Date(record.created_at).toLocaleString() : 'N/A'}</TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={1}>
                      <IconButton
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleOpenDrawer(record);
                        }}
                      >
                        <ZoomInIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(record.id);
                        }}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
              {emptyRows > 0 && (
                <TableRow style={{ height: 53 * emptyRows }}>
                  <TableCell colSpan={7} />
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={filteredHistory.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handlePageChange}
          onRowsPerPageChange={handleRowsPerPageChange}
          rowsPerPageOptions={[5, 10, 25]}
          labelRowsPerPage="Rows per page:"
        />
      </Paper>

      {/* No records message */}
      {!filteredHistory.length && !search && (
        <Box sx={{ mt: 3, textAlign: 'center', color: 'text.secondary' }}>
          No history records available
        </Box>
      )}

      {/* Bulk Actions */}
      {selectedRows.length > 0 && (
        <Box sx={{ mb: 3, display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<DownloadIcon />}
            onClick={handleDownload}
          >
            Download Selected ({selectedRows.length})
          </Button>
          <Button
            variant="contained"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={handleDelete}
          >
            Delete Selected
          </Button>
        </Box>
      )}

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
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
          },
        }}
      >
        {/* Header */}
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {selectedRecord?.building_name || 'Image Viewer'}
          </Typography>
          <IconButton onClick={handleCloseDrawer} size="small">
            <CloseIcon sx={{ color: 'text.secondary' }} />
          </IconButton>
        </Box>

        {/* Image Controls */}
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', gap: 2, mb: 2 }}>
          <IconButton 
            onClick={handlePrevImage} 
            disabled={imageIndex === 0}
            size="small"
          >
            <RotateLeftIcon sx={{ color: imageIndex === 0 ? 'text.disabled' : 'primary.main' }} />
          </IconButton>
          <IconButton 
            onClick={handleZoomOut} 
            disabled={zoom <= 0.5}
            size="small"
          >
            <ZoomOutIcon sx={{ color: zoom <= 0.5 ? 'text.disabled' : 'primary.main' }} />
          </IconButton>
          <IconButton 
            onClick={handleZoomIn} 
            disabled={zoom >= 3}
            size="small"
          >
            <ZoomInIcon sx={{ color: zoom >= 3 ? 'text.disabled' : 'primary.main' }} />
          </IconButton>
          <IconButton 
            onClick={handleRotateLeft} 
            size="small"
          >
            <RotateLeftIcon sx={{ color: 'primary.main' }} />
          </IconButton>
          <IconButton 
            onClick={handleRotateRight} 
            size="small"
          >
            <RotateRightIcon sx={{ color: 'primary.main' }} />
          </IconButton>
          <IconButton 
            onClick={handleNextImage} 
            disabled={imageIndex === (selectedRecord?.processed_image ? 1 : 0)}
            size="small"
          >
            <RotateRightIcon sx={{ color: imageIndex === (selectedRecord?.processed_image ? 1 : 0) ? 'text.disabled' : 'primary.main' }} />
          </IconButton>
        </Box>

        {/* Image Grid */}
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            p: 2,
          }}
        >
          {/* Original Image */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              mb: 2,
              borderRadius: 1,
              border: 1,
              borderColor: 'divider',
              p: 2,
              height: '50%',
              overflow: 'hidden',
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Original Image
            </Typography>
            {selectedRecord?.filename ? (
              <img
                src={`http://localhost:8000/uploads/${selectedRecord.filename}`}
                alt="Original"
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  transform: `scale(${zoom}) rotate(${rotation}deg)`,
                  transition: 'transform 0.3s ease',
                  objectFit: 'contain',
                }}
              />
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                No original image available
              </Typography>
            )}
          </Box>

          {/* Processed Image */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              borderRadius: 1,
              border: 1,
              borderColor: 'divider',
              p: 2,
              height: '50%',
              overflow: 'hidden',
            }}
          >
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Processed Image (RTU Count: {selectedRecord?.rtu_count || 'N/A'})
            </Typography>
            {selectedRecord?.processed_image ? (
              <img
                src={`http://localhost:8000${selectedRecord.processed_image}`}
                alt="Processed"
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  transform: `scale(${zoom}) rotate(${rotation}deg)`,
                  transition: 'transform 0.3s ease',
                  objectFit: 'contain',
                }}
              />
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
                No processed image available
              </Typography>
            )}
          </Box>
        </Box>

        {/* Image Info */}
        <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
          <Stack spacing={2}>
            <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
              Details
            </Typography>
            <Stack spacing={1}>
              <Typography variant="body2" color="text.secondary">
                <strong>RTU Count:</strong> {selectedRecord?.rtu_count || 'N/A'}
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="body2" color="text.secondary">
                  <strong>Lead Score:</strong>
                </Typography>
                <Typography 
                  variant="body2" 
                  sx={{
                    color: getLeadScoreColor(selectedRecord?.lead_score),
                    fontWeight: 'bold',
                  }}
                >
                  {getLeadScoreLabel(selectedRecord?.lead_score)}
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                <strong>Address:</strong> {selectedRecord?.address || 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                <strong>Date:</strong> {selectedRecord?.created_at ? new Date(selectedRecord.created_at).toLocaleString() : 'N/A'}
              </Typography>
            </Stack>
          </Stack>
        </Box>
      </Drawer>
    </Box>
  );
}
