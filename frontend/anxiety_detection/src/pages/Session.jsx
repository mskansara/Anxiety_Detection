import React, { useState } from 'react'
import Sidenav from '../components/Home/Sidenav'
import SessionHeader from '../components/Session/SessionHeader'
import { Box, Grid, Typography } from '@mui/material'
import NewSession from '../components/Session/NewSession'
import CurrentSession from '../components/Session/CurrentSession'
import SessionSummary from '../components/Session/SessionSummary'

function Session() {
    const [isSession, setIsSession] = useState(false)
    const [isSessionCompleted, setIsSessionCompleted] = useState(false)
    const [patientList, setPatientList] = useState([
        {
            "id": "01",
            "name": "Manthan Kansara"
        },
        {
            "id": "02",
            "name": "Manthan Kansara"
        },
        {
            "id": "03",
            "name": "Manthan Kansara"
        },
        {
            "id": "03",
            "name": "Manthan Kansara"
        }
    ])
    const [patient, setPatient] = useState()
    console.log(patient)

    return (
        <>
            <Grid container sx={{ display: "flex", backgroundColor: '#EEEEEE', height: '100%', minHeight: '100vh', width: '100%' }}>
                <Grid item sx={{ width: 'auto' }}>
                    <Sidenav />
                </Grid>

                <Grid item sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ p: 4 }}>
                        <Typography variant='h4'>Session</Typography>
                    </Box>
                    <Box sx={{ flexGrow: 1 }}>
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
        </>
    )
}

export default Session
