import React, { useState } from 'react'
import SessionHeader from './SessionHeader'
import { Box, Button, Typography } from '@mui/material'
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import Backdrop from '@mui/material/Backdrop';



function NewSession({ setIsSession, setIsSessionCompleted, patientList, setPatient, patient }) {
    const [loading, setLoading] = useState(false);
    const handleChange = (event) => {
        setPatient(event.target.value);
    };

    const startStreaming = async () => {
        try {
            setLoading(true);
            const token = localStorage.getItem('token'); // Retrieve JWT token from localStorage
            const response = await axios.post('http://localhost:8000/start_streaming', {
                interface: 'usb',
                profile_name: 'profile18'
            }, {
                headers: {
                    'Authorization': `Bearer ${token}` // Pass the token in the Authorization header
                }
            });
            setLoading(false);
            if (response.data.status === "Streaming started") {
                // Handle the successful start of streaming
                return {
                    "status": "success"
                };
            }
        } catch (error) {
            setLoading(false);
            console.error('Error starting streaming:', error);
            alert('Failed to start streaming');
        }
    };
    const startSession = async () => {
        let sessionStatus = await startStreaming()
        try {
            if (sessionStatus['status'] == "success") {
                setIsSession(true)
                setIsSessionCompleted(false)
                console.log("Session started")
            } else {
                setIsSession(false)
                setIsSessionCompleted(false)
                console.log("Failed to start the session")
            }
        } catch (error) {
            return
        }

        // setIsSession(true)
        // setIsSessionCompleted(false)


    }
    return (
        <>
            <Box sx={{ p: 4 }}>
                <Typography variant='h6'>New Session</Typography>
                <Box sx={{ p: 4 }}>
                    <FormControl fullWidth>
                        <InputLabel id="patient-label">Patients</InputLabel>
                        <Select
                            labelId="patient-label"
                            id="patient-select"
                            value={patient}
                            label="Patient"
                            onChange={handleChange}
                        >
                            {
                                patientList.map((data, index) => (
                                    <MenuItem value={data}>{data.id} - {data.name}</MenuItem>
                                ))
                            }
                        </Select>
                    </FormControl>
                </Box>

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
                        sx={{ backgroundColor: "#021526", color: 'white' }}
                        onClick={() => startSession()}
                    >
                        Start Session
                    </Button>
                )}
            </Box>

        </>

    )
}

export default NewSession