import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const StreamingController = () => {
    const [isStreaming, setIsStreaming] = useState(false);
    const [summary, setSummary] = useState('');
    const [transcriptions, setTranscriptions] = useState([]);
    const websocketRef = useRef(null);

    const colorCodes = {
        neutral: 'white',
        happy: 'blue',
        sad: 'orange',
        angry: 'red',
        null: 'white' // default color for null emotion
    };

    useEffect(() => {
        if (isStreaming) {
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
    }, [isStreaming]);

    const startStreaming = async () => {
        try {
            const response = await axios.post('http://localhost:8000/start_streaming', {
                interface: 'usb',
                profile_name: 'profile18'
            });
            if (response.data.status === "Streaming started") {
                setIsStreaming(true);
                alert('Streaming started');
            }
        } catch (error) {
            console.error('Error starting streaming:', error);
            alert('Failed to start streaming');
        }
    };

    const stopStreaming = async () => {
        try {
            const response = await axios.post('http://localhost:8000/stop_streaming');
            if (response.data.status === "Streaming stopped") {
                setIsStreaming(false);
                setSummary(response.data.summary);
                alert('Streaming stopped');
            }
        } catch (error) {
            console.error('Error stopping streaming:', error);
            alert('Failed to stop streaming');
        }
    };

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

    return (
        <div>
            <button onClick={startStreaming} disabled={isStreaming}>Start Streaming</button>
            <button onClick={stopStreaming} disabled={!isStreaming}>Stop Streaming</button>
            <div>
                <h2>Real-Time Transcription</h2>
                <p>{renderTranscriptions()}</p>
            </div>
            {summary && (
                <div>
                    <h2>Summary</h2>
                    <p>{summary}</p>
                </div>
            )}
        </div>
    );
};

export default StreamingController;
