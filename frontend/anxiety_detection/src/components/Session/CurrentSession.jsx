import { Box, Button, Card, CardContent, Grid, Typography } from '@mui/material'
import React, { useState } from 'react'
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import Backdrop from '@mui/material/Backdrop';

function CurrentSession({ setIsSession, setIsSessionCompleted, patient }) {
    const [loading, setLoading] = useState(false);
    const stopStreaming = async () => {
        try {
            setLoading(true);
            const response = await axios.post('http://localhost:8000/stop_streaming');
            if (response.data.status === "Streaming stopped") {
                setLoading(false)
                // setSummary(response.data.summary);
                // alert('Streaming stopped');
                return {
                    "status": "success"
                }
            }
        } catch (error) {
            console.error('Error stopping streaming:', error);
            alert('Failed to stop streaming');
        }
    };
    const stopSession = async () => {
        let sessionStatus = await stopStreaming()
        if (sessionStatus['status'] == "success") {
            setIsSession(false)
            setIsSessionCompleted(true)
            console.log("Session stopped")
        } else {
            setIsSession(true)
            setIsSessionCompleted(false)
            console.log("Failed to stop the session")
        }

    }
    return (
        <>
            {
                patient && (
                    <Box sx={{ p: 4, width: '100%' }}>
                        <Typography variant='h6'>Current Session: {patient.id} - {patient.name}</Typography>
                        <br />
                        <Typography variant='body2'>Session is in progress...</Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12} md={9}>
                                <Card sx={{ minWidth: '100%' }}>
                                    <CardContent>
                                        <Typography>Transcription</Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                            <Grid item xs={12} md={3}>
                                <Card sx={{ minWidth: '100%' }}>
                                    <CardContent>
                                        <Typography>Detected Mood</Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                        </Grid>
                        <br />
                        {loading ? (
                            <Backdrop
                                sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
                                open={loading}
                            // onClick={handleClose}
                            >
                                <CircularProgress value={100} />
                            </Backdrop>
                        ) : (
                            <Button
                                variant='contained'
                                sx={{ backgroundColor: "#021526" }}
                                onClick={() => stopSession()}
                            >
                                Stop Session
                            </Button>
                        )}
                    </Box>
                )
            }

        </>

    )
}

export default CurrentSession