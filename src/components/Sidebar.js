import React from 'react';
import { 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Typography, 
  Box, 
  Divider 
} from '@mui/material';
import { 
  Home, 
  Description, 
  History, 
  Settings, 
  ExitToApp, 
  Info,
  Map,
  Construction
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { styled } from '@mui/material/styles';

const StyledDrawer = styled(Drawer)(({ theme }) => ({
  width: 240,
  flexShrink: 0,
  '& .MuiDrawer-paper': {
    width: 240,
    boxSizing: 'border-box',
    borderRight: '1px solid rgba(0, 0, 0, 0.12)',
  },
}));

const StyledListItem = styled(ListItem)(({ theme }) => ({
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
    borderRadius: theme.shape.borderRadius,
  },
  '&.active': {
    backgroundColor: theme.palette.primary.light,
    '& .MuiListItemIcon-root': {
      color: theme.palette.primary.main,
    },
    '& .MuiListItemText-primary': {
      color: theme.palette.primary.main,
    },
  },
}));

const Sidebar = () => {
  const navigate = useNavigate();
  const [activeRoute, setActiveRoute] = React.useState('/');

  const menuItems = [
    { text: 'Home', icon: <Home />, path: '/' },
    { text: 'RTU Detection', icon: <Construction />, path: '/rtu' },
    { text: 'Upload History', icon: <History />, path: '/history' },
    { text: 'Map View', icon: <Map />, path: '/map' },
    { text: 'About', icon: <Info />, path: '/about' },
    { text: 'Help', icon: <Description />, path: '/help' }
  ];

  const handleNavigation = (path) => {
    navigate(path);
    setActiveRoute(path);
  };

  return (
    <StyledDrawer variant="permanent" anchor="left" open={true}>
      <Box sx={{ p: 3 }}>
        <Typography variant="h6" color="primary" sx={{ mb: 3 }}>
          RTU Detection App
        </Typography>
      </Box>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <StyledListItem
            key={item.text}
            button="true"
            onClick={() => handleNavigation(item.path)}
            className={activeRoute === item.path ? 'active' : ''}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.text} />
          </StyledListItem>
        ))}
      </List>
    </StyledDrawer>
  );
};

export default Sidebar;
