// src/App.js
import React from 'react';

import './App.css';
import StreamingController from './components/StreamingController';

function App() {
    return (
        <div className="App">
            <header className="App-header">
                <h1>Streaming Control</h1>
                <StreamingController />
            </header>
        </div>
    );
}

export default App;
