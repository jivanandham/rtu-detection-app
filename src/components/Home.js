import React from 'react';
import { 
  Box, 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  useTheme, 
  useMediaQuery,
  Stack,
  Paper,
  Avatar,
  Divider,
  Chip,
  IconButton,
  Button,
  Tooltip
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  Analytics, 
  Assessment, 
  BarChart, 
  Description, 
  History, 
  Settings,
  Notifications,
  Help,
  Person,
  Download,
  Upload,
  FileDownloadDone,
  FileUploadDone
} from '@mui/icons-material';

const Home = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const metrics = [
    {
      title: 'Total Images Processed',
      value: '1,234',
      icon: <Analytics sx={{ fontSize: 32 }} />,
      color: theme.palette.primary.main,
      trend: '23%'
    },
    {
      title: 'RTUs Detected',
      value: '897',
      icon: <Assessment sx={{ fontSize: 32 }} />,
      color: theme.palette.success.main,
      trend: '15%'
    },
    {
      title: 'Success Rate',
      value: '92%',
      icon: <TrendingUp sx={{ fontSize: 32 }} />,
      color: theme.palette.success.main,
      trend: '↑ 3%'
    },
    {
      title: 'Failed Detections',
      value: '12',
      icon: <TrendingDown sx={{ fontSize: 32 }} />,
      color: theme.palette.error.main,
      trend: '↓ 2%'
    },
    {
      title: 'Average Lead Score',
      value: '8.5/10',
      icon: <BarChart sx={{ fontSize: 32 }} />,
      color: theme.palette.secondary.main,
      trend: '4.5%'
    },
    {
      title: 'Buildings Analyzed',
      value: '345',
      icon: <Description sx={{ fontSize: 32 }} />,
      color: theme.palette.info.main,
      trend: '8%'
    }
  ];

  return (
    <Box sx={{ p: 3 }}>
      <Grid container spacing={3}>
        {/* Header Section */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h4" component="h1">
              Dashboard
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center">
              <Tooltip title="Notifications">
                <IconButton color="primary">
                  <Notifications />
                </IconButton>
              </Tooltip>
              <Tooltip title="Help">
                <IconButton color="primary">
                  <Help />
                </IconButton>
              </Tooltip>
              <Tooltip title="Profile">
                <Avatar sx={{ bgcolor: theme.palette.primary.main }}>
                  U
                </Avatar>
              </Tooltip>
            </Stack>
          </Box>
        </Grid>

        {/* Metrics Cards */}
        {metrics.map((metric, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                p: 3,
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                transition: 'all 0.2s ease',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
                },
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: theme.palette.mode === 'dark' ? theme.palette.background.paper : theme.palette.background.default,
                    color: metric.color,
                    mr: 2,
                  }}
                >
                  {metric.icon}
                </Box>
                <Typography variant="h6" color="text.primary">
                  {metric.title}
                </Typography>
              </Box>
              <Typography variant="h4" color="primary" sx={{ mb: 1 }}>
                {metric.value}
              </Typography>
              <Typography variant="body2" color={metric.trend.includes('↑') ? 'success.main' : 'error.main'}>
                {metric.trend}
              </Typography>
            </Card>
          </Grid>
        ))}

        {/* Recent Activity Section */}
        <Grid item xs={12} md={8}>
          <Paper
            sx={{
              p: 3,
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              borderRadius: 2,
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6" color="primary">
                Recent Activity
              </Typography>
              <Button variant="outlined" startIcon={<History />}>
                View All
              </Button>
            </Box>
            {/* Add your recent activity content here */}
          </Paper>
        </Grid>

        {/* Quick Actions Section */}
        <Grid item xs={12} md={4}>
          <Paper
            sx={{
              p: 3,
              height: '100%',
              borderRadius: 2,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <Typography variant="h6" color="primary" sx={{ mb: 3 }}>
              Quick Actions
            </Typography>
            <Stack spacing={2}>
              <Button
                variant="contained"
                startIcon={<Upload />}
                fullWidth
                sx={{
                  textTransform: 'none',
                  fontWeight: 600,
                  fontSize: '0.875rem',
                }}
              >
                Upload New Image
              </Button>
              <Button
                variant="outlined"
                startIcon={<Download />}
                fullWidth
                sx={{
                  textTransform: 'none',
                  fontWeight: 600,
                  fontSize: '0.875rem',
                }}
              >
                Download Report
              </Button>
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Home;
