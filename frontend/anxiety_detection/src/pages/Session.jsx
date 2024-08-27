import React, { useState } from 'react';
import Sidenav from '../components/Home/Sidenav';
import { Box, Grid, Typography } from '@mui/material';
import NewSession from '../components/Session/NewSession';
import CurrentSession from '../components/Session/CurrentSession';
import SessionSummary from '../components/Session/SessionSummary';

function Session() {
    const [isSession, setIsSession] = useState(false);
    const [isSessionCompleted, setIsSessionCompleted] = useState(false);
    const [patientList, setPatientList] = useState([
        { id: "01", name: "Manthan Kansara" },
        { id: "02", name: "Manthan Kansara" },
        { id: "03", name: "Manthan Kansara" },
        { id: "03", name: "Manthan Kansara" }
    ]);
    const [patient, setPatient] = useState();

    return (
        <Grid container sx={{ backgroundColor: '#EEEEEE', minHeight: '100vh' }}>
            <Grid item xs={1} sm={1} md={1}>
                <Sidenav />
            </Grid>

            <Grid item xs={12} sm={9} md={10} sx={{ pt: 4 }}>
                <Box>
                    <Typography variant='h4'>Session</Typography>
                </Box>
                <Box sx={{ mt: 2 }}>
                    {(!isSession && !isSessionCompleted) && (
                        <NewSession
                            setIsSession={setIsSession}
                            setIsSessionCompleted={setIsSessionCompleted}
                            patientList={patientList}
                            setPatient={setPatient}
                            patient={patient}
                        />
                    )}
                    {(isSession && !isSessionCompleted) && (
                        <CurrentSession
                            setIsSession={setIsSession}
                            setIsSessionCompleted={setIsSessionCompleted}
                            patient={patient}
                        />
                    )}
                    {(!isSession && isSessionCompleted) && (
                        <SessionSummary
                            setIsSession={setIsSession}
                            setIsSessionCompleted={setIsSessionCompleted}
                            patient={patient}
                        />
                    )}
                </Box>
            </Grid>
        </Grid>
    );
}

export default Session;
