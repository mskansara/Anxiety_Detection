import { Box, Button, Card, CardContent, Grid, Typography } from '@mui/material'
import React from 'react'

function SessionSummary({ setIsSession, setIsSessionCompleted, patient }) {
    const saveSession = () => {
        setIsSession(false)
        setIsSessionCompleted(false)
        console.log("Session saved")
    }
    return (
        <>
            {
                patient && (
                    <Box sx={{ p: 4 }}>
                        <Typography variant='h6'>Session Summary : {patient.id} - {patient.name}</Typography>
                        <br />
                        <Typography variant='body2'>Session completed</Typography>
                        <Grid container>
                            <Grid item xs={12} md={12}>
                                <Card sx={{ minWidth: '100%' }}>
                                    <CardContent>
                                        <Typography>Summary</Typography>
                                    </CardContent>
                                </Card>
                            </Grid>

                        </Grid>
                        <br />
                        <Grid container spacing={2}>
                            <Grid item>
                                <Button
                                    variant='contained'
                                    onClick={() => saveSession()}
                                >
                                    Save
                                </Button>
                            </Grid>
                            <Grid item>
                                <Button
                                    sx={{ backgroundColor: "#021526", color: 'white' }}
                                    variant='contained'
                                // onClick={() => saveSession()}
                                >
                                    Share
                                </Button>
                            </Grid>

                        </Grid>


                    </Box>
                )
            }


        </>

    )
}

export default SessionSummary