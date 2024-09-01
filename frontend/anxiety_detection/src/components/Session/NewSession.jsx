import React, { useState, useEffect } from 'react';
import SessionHeader from './SessionHeader';
import { Box, Button, Typography } from '@mui/material';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import axios from 'axios';
import CircularProgress from '@mui/material/CircularProgress';
import Backdrop from '@mui/material/Backdrop';

function NewSession({ setIsSession, setIsSessionCompleted, setPatient, patient, setSessionId }) {
    const [loading, setLoading] = useState(false);
    const [patientList, setPatientList] = useState([]);


    useEffect(() => {
        const fetchPatients = async () => {
            try {
                const token = localStorage.getItem('token'); // Retrieve JWT token from localStorage
                const doctorId = localStorage.getItem('id')// Replace with the actual doctor ID or fetch it from the logged-in user details
                console.log(doctorId)
                const response = await axios.post(`http://localhost:8000/doctors/patients`,
                    {
                        doctor_id: doctorId
                    },
                    {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });
                console.log(response)

                setPatientList(response.data);
            } catch (error) {
                console.error('Error fetching patients:', error);
                alert('Failed to fetch patients');
            }
        };

        fetchPatients();
    }, []);

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
        let sessionStatus = await startStreaming();

        try {
            if (sessionStatus && sessionStatus['status'] === "success") {
                const doctorId = localStorage.getItem("id");
                const patientId = patient._id;
                console.log(doctorId)
                console.log(patientId)

                const response = await axios.post("http://localhost:8000/sessions", {
                    'doctor_id': doctorId,
                    'patient_id': patientId
                });

                if (response.data.session_id) {
                    setIsSession(true);
                    setIsSessionCompleted(false);
                    console.log("Session started, Session ID:", response.data.session_id);
                    setSessionId(response.data.session_id)
                    localStorage.setItem('session_id', response.data.session_id)
                    // You may want to store the session_id in state or localStorage if needed
                } else {
                    setIsSession(false);
                    setIsSessionCompleted(false);
                    console.log("Failed to create a session");
                }
            } else {
                setIsSession(false);
                setIsSessionCompleted(false);
                console.log("Failed to start the session");
            }
        } catch (error) {
            return;
        }

    };

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
                                    <MenuItem key={index} value={data}>{data._id} - {data.name}</MenuItem>
                                ))
                            }
                        </Select>
                    </FormControl>
                </Box>

                {loading ? (
                    <Backdrop
                        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
                        open={loading}
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
    );
}

export default NewSession;