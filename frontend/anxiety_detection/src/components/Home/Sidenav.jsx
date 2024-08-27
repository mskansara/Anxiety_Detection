import React from 'react'

import AccountCircle from '@mui/icons-material/AccountCircle';
import LogoutOutlinedIcon from '@mui/icons-material/LogoutOutlined';
import WorkOutlineOutlinedIcon from '@mui/icons-material/WorkOutlineOutlined';
import { Box, IconButton, List, ListItem, ListItemButton, ListItemIcon, Typography } from '@mui/material';
import MuiDrawer from '@mui/material/Drawer';
import { styled } from '@mui/material/styles';
import { useLocation, useNavigate } from 'react-router-dom';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import logo from '../../images/logo.png';
import axios from 'axios';


const drawerWidth = 80;
const Drawer = styled(MuiDrawer, { shouldForwardProp: (prop) => prop !== 'open' })(
    ({ theme, open }) => ({
        width: drawerWidth,
        flexShrink: 0,
        whiteSpace: 'nowrap',
        boxSizing: 'border-box',
        '& .MuiDrawer-paper': {
            backgroundColor: '#021526',
            width: drawerWidth,
        },
    }),
);

const menuItems = [
    { route: '/', title: 'Home', icon: <DashboardOutlinedIcon /> },
    { route: '/session', title: 'Session', icon: <WorkOutlineOutlinedIcon /> },
]

const settingsMenu = [
    { route: '/profile', title: 'Profile', icon: <AccountCircle /> },
    { title: 'Logout', icon: <LogoutOutlinedIcon /> },
]

function Sidenav() {
    const location = useLocation();
    const navigate = useNavigate();

    const logout = async () => {
        try {
            const token = localStorage.getItem("token");

            // Optional: Call the backend logout endpoint
            await axios.post("http://localhost:8000/logout", {}, {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            });

            // Remove token from local storage
            localStorage.removeItem("token");
            localStorage.removeItem("role");

            // Redirect to login page
            navigate("/login");
        } catch (error) {
            console.error("Logout failed", error);
            // Handle logout failure (e.g., show error message)
        }
    };

    // Function to determine if the route matches the current location
    const isRouteActive = (route) => {
        return location.pathname === route;
    };



    return (
        <Box sx={{ display: 'flex', }}>
            <Box height={60} />
            <Drawer variant="permanent" sx={{ backgroundColor: '#6EACDA' }}>
                <IconButton>
                    <img
                        src={logo}
                        alt='TalentCen'
                        loading="lazy"
                        height='60px'
                    />
                </IconButton>
                <br /><br />
                <Box sx={{ height: '100%', backgroundColor: '#021526', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                    <List sx={{ height: '100%', backgroundColor: '#021526', display: 'flex', flexDirection: 'column', justifyContent: 'flex-start' }}>
                        {menuItems.map((item, index) => (
                            <ListItem key={index} disablePadding sx={{ display: 'block', mb: 1 }} onClick={() => navigate(item.route)}>
                                <ListItemButton
                                    sx={{
                                        minHeight: 48,
                                        justifyContent: 'center',
                                        px: 2.5,
                                        flexDirection: 'column',
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 0, justifyContent: 'center', size: 'large', color: isRouteActive(item.route) ? '#E2E2B6' : '#E2E2B6' }}>
                                        {item.icon}
                                    </ListItemIcon>
                                    <Typography sx={{ color: isRouteActive(item.route) ? '#E2E2B6' : '#E2E2B6', fontSize: '0.8rem' }}>{item.title}</Typography>
                                </ListItemButton>
                            </ListItem>
                        ))}
                    </List>

                    <List sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                        {/* Company */}
                        {settingsMenu.map((item, index) => (
                            <ListItem key={index} disablePadding sx={{ display: 'block', mb: 1 }} onClick={item.title === 'Logout' ? logout : () => navigate(item.route)}>
                                <ListItemButton
                                    sx={{
                                        minHeight: 48,
                                        justifyContent: 'center',
                                        px: 2.5,
                                        flexDirection: 'column',

                                    }}

                                >
                                    <ListItemIcon sx={{ minWidth: 0, justifyContent: 'center', size: 'large', color: isRouteActive(item.route) ? '#E2E2B6' : '#E2E2B6' }}>
                                        {item.icon}
                                    </ListItemIcon>
                                    <Typography sx={{ color: isRouteActive(item.route) ? '#E2E2B6' : '#E2E2B6', fontSize: '0.8rem' }}>{item.title}</Typography>
                                </ListItemButton>
                            </ListItem>
                        ))}
                    </List>
                </Box>
            </Drawer>
        </Box>
    )
}

export default Sidenav