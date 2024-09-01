import React, { useState } from 'react';
import { Box, Button, Card, CardContent, Grid, Typography, TextField, IconButton } from '@mui/material';
import EditOutlinedIcon from '@mui/icons-material/EditOutlined';
import SaveOutlinedIcon from '@mui/icons-material/SaveOutlined';

let dummySummary = `
**Patient's Name:** Jane Doe
**Session Date:** 2023-07-30
**Session Number:** 2

**Summary of the Session:**

The patient expressed feelings of anxiety, which started in high school during class presentations. These feelings have worsened over time, and now even mundane tasks like grocery shopping trigger anxiety. The patient mentioned that deep breathing helps, but not always. They also tried meditation, but found it difficult to focus. The patient has a supportive family, but they don't fully understand their struggles. They considered joining a support group, which the doctor suggested might be helpful.

The patient smiled and expressed happiness when recalling a recent event where they successfully managed their anxiety during a family gathering. This shows progress in their coping mechanisms. However, there were moments of sadness when discussing their ongoing struggles with anxiety and its impact on daily life. The patient also showed signs of anger when talking about the lack of understanding from some friends and the stigma associated with anxiety.

**Key Points:**

* Anxiety started in high school and worsened over time
* Deep breathing is a coping mechanism, but not always effective
* Meditation is difficult due to focus issues
* Supportive family, but lack of understanding
* Consideration of joining a support group
* Managed anxiety during a family gathering (happy moment)
* Frustration with lack of understanding from friends

**Mood Summary:**

The patient's mood varied throughout the session. They started with a neutral tone, but shifted to anxious and nervous as they discussed their struggles. They expressed frustration and anger when discussing the challenges of managing anxiety and the lack of understanding from friends. However, they also showed hopeful and happy emotions when discussing potential solutions and the doctor's support. The patient ended the session feeling grateful and relieved, showing a mix of emotions including happiness and sadness as they reflected on their progress and ongoing challenges.

**Next Steps:**

The patient will continue to work on managing their anxiety, possibly with the help of a support group. The doctor will provide guidance on coping mechanisms and recommend additional resources if needed. The patient will also explore strategies to improve focus during meditation. Additionally, the patient will be encouraged to engage in activities that previously brought them joy and happiness, to help counteract feelings of sadness and anxiety.

**Additional Notes:**

The patient's emotional expression and tone were closely monitored throughout the session. The detected mood scores and colors provided a visual representation of the patient's emotional state. The doctor will continue to use this data to inform their treatment plan and provide personalized support.
`;

const colorCodes = {
    neutral: 'black',
    happy: 'blue',
    sad: 'red',
    angry: 'orange',
    null: 'black' // default color for null emotion
};

// Function to determine the mood based on keywords
const getMood = (sentence) => {
    if (sentence.includes('anxiety') || sentence.includes('struggles')) return 'sad';
    if (sentence.includes('happiness') || sentence.includes('hopeful') || sentence.includes('grateful') || sentence.includes('joy')) return 'happy';
    if (sentence.includes('anger') || sentence.includes('frustration')) return 'angry';
    return 'neutral';
};

// Function to colorize the summary text
const colorizeSummary = (summary) => {
    const sentences = summary.split('\n');
    return sentences.map((sentence, index) => {
        const mood = getMood(sentence);
        const color = colorCodes[mood];
        return (
            <Typography key={index} variant='body2' sx={{ color }}>
                {sentence}
            </Typography>
        );
    });
};

function SessionSummary({ setIsSession, setIsSessionCompleted, patient, sessionId }) {
    const [isEditing, setIsEditing] = useState(false);
    const [summary, setSummary] = useState(dummySummary);
    const session_id = localStorage.getItem('session_id')
    const saveSession = () => {
        setIsSession(false);
        setIsSessionCompleted(false);
        console.log('Session saved');
    };

    const handleEditToggle = () => {
        if (isEditing) {
            // Save the changes and update dummySummary
            dummySummary = summary; // Update the dummySummary with the new summary
        }
        setIsEditing(!isEditing);
    };

    return (
        <>
            {patient && (
                <Box sx={{ p: 1, width: '100%' }}>
                    <Typography variant='h6'>
                        Session Summary: {patient.id} - {patient.name}
                    </Typography>
                    <br />
                    <Typography variant='body2'>Session completed</Typography>
                    <Grid container>
                        <Grid item xs={12} md={12}>
                            <Card sx={{ minWidth: '100%' }}>
                                <CardContent>
                                    <Grid container justifyContent="space-between" alignItems="center">
                                        <Typography variant='h6'>Summary</Typography>
                                        <IconButton
                                            onClick={handleEditToggle}
                                            sx={{}}
                                            aria-label="edit-summary"
                                            color='#021526'
                                        >
                                            {isEditing ? <SaveOutlinedIcon /> : <EditOutlinedIcon />}
                                        </IconButton>
                                    </Grid>
                                    {isEditing ? (
                                        <TextField
                                            fullWidth
                                            multiline
                                            rows={20}
                                            value={summary}
                                            onChange={(e) => setSummary(e.target.value)}
                                        />
                                    ) : (
                                        colorizeSummary(summary)
                                    )}
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                    <br />
                    <Grid container spacing={2}>
                        <Grid item>
                            {!isEditing && (
                                <Button variant='contained' onClick={saveSession}>
                                    Save
                                </Button>
                            )}
                        </Grid>
                    </Grid>
                </Box>
            )}
        </>
    );
}

export default SessionSummary;
