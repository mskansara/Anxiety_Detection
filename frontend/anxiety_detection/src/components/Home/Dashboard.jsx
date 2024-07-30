import { Box, Grid, Typography } from '@mui/material'
import React from 'react'
import Sidenav from './Sidenav'

function Dashboard() {
    return (
        <Grid sx={{ display: "flex", backgroundColor: '#EEEEEE', height: '100%', minHeight: '100vh' }}>
            <Sidenav />
            <Grid>
                <Box sx={{ p: 4 }}>
                    <Typography variant='h4'>Home</Typography>
                </Box>

            </Grid>


        </Grid>
    )
}

export default Dashboard