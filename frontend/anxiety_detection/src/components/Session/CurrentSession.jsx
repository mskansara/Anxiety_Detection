import { Box, Button, Card, CardContent, Grid, Typography } from '@mui/material';
import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import Backdrop from '@mui/material/Backdrop';

function CurrentSession({ setIsSession, setIsSessionCompleted, patient }) {
    const [loading, setLoading] = useState(false);
    const [transcriptions, setTranscriptions] = useState([]);
    const websocketRef = useRef(null);
    const colorCodes = {
        neutral: 'black',
        happy: 'blue',
        sad: 'orange',
        angry: 'red',
        null: 'black'
    };

    useEffect(() => {
        if (patient) {
            websocketRef.current = new WebSocket('ws://localhost:8000/ws');

            websocketRef.current.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.transcription) {
                    setTranscriptions(prevTranscriptions => [
                        ...prevTranscriptions,
                        {
                            text: data.transcription,
                            start_timestamp: data.start_timestamp,
                            end_timestamp: data.end_timestamp,
                            mood: data.mood,
                        }
                    ]);
                } else if (data.mood) {
                    setTranscriptions(prevTranscriptions => [
                        ...prevTranscriptions,
                        {
                            text: '',
                            start_timestamp: data.start_timestamp,
                            end_timestamp: data.end_timestamp,
                            mood: data.mood,
                        }
                    ]);
                }
            };

            websocketRef.current.onopen = () => {
                console.log('WebSocket connection established');
            };

            websocketRef.current.onclose = () => {
                console.log('WebSocket connection closed');
            };

            websocketRef.current.onerror = (error) => {
                console.log('WebSocket error:', error);
            };

            return () => {
                if (websocketRef.current) {
                    websocketRef.current.close();
                }
            };
        }
    }, [patient]);

    const renderTranscriptions = () => {
        return transcriptions.map((transcription, index) => {
            const { text, mood, start_timestamp, end_timestamp } = transcription;
            const emotion = mood?.emotion || 'null';
            const color = colorCodes[emotion];
            return (
                <span key={index} style={{ color: color }}>
                    {text}
                </span>
            );
        });
    };

    const renderMood = () => {
        if (transcriptions.length > 0) {
            const lastTranscription = transcriptions[transcriptions.length - 1];
            const mood = lastTranscription.mood?.emotion || 'null';
            const color = colorCodes[mood];
            return (
                <Typography style={{ color: color }}>
                    {mood}
                </Typography>
            );
        }
        return <Typography>Neutral</Typography>;
    };

    const stopStreaming = async () => {
        try {
            setLoading(true);
            const token = localStorage.getItem('token'); // Retrieve JWT token from localStorage
            const response = await axios.post('http://localhost:8000/stop_streaming', {}, {
                headers: {
                    'Authorization': `Bearer ${token}` // Pass the token in the Authorization header
                }
            });
            setLoading(false);
            if (response.data.status === "Streaming stopped") {
                // Handle the successful stop of streaming
                return {
                    "status": "success"
                };
            }
        } catch (error) {
            setLoading(false);
            console.error('Error stopping streaming:', error);
            alert('Failed to stop streaming');
        }
    };

    const stopSession = async () => {
        let sessionStatus = await stopStreaming();
        if (sessionStatus['status'] === "success") {
            setIsSession(false);
            setIsSessionCompleted(true);
            console.log("Session stopped");
        } else {
            setIsSession(true);
            setIsSessionCompleted(false);
            console.log("Failed to stop the session");
        }
        // setIsSession(false)
        // setIsSessionCompleted(true)
    };

    return (
        <>
            {
                patient && (
                    <Box sx={{ p: 4, width: '100%' }}>
                        <Typography variant='h6'>Current Session: {patient.id} - {patient.name}</Typography>
                        <br />
                        <Typography variant='body2'>Session is in progress...</Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12} md={8}>
                                <Card>
                                    <CardContent>
                                        <Typography variant='body2'>Transcriptions:</Typography>
                                        {renderTranscriptions()}
                                    </CardContent>
                                </Card>
                            </Grid>
                            <Grid item xs={12} md={4}>
                                <Card>
                                    <CardContent>
                                        <Typography variant='body2'>Mood:</Typography>
                                        {renderMood()}
                                    </CardContent>
                                </Card>
                            </Grid>

                        </Grid>
                        <br />
                        <Button onClick={stopSession} variant='contained' color='error'>Stop Session</Button>
                    </Box>
                )
            }
            <Backdrop sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }} open={loading}>
                <CircularProgress color="inherit" />
            </Backdrop>
        </>
    );
}

export default CurrentSession;
