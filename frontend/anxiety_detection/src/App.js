// src/App.js
import React from 'react';

import './App.css';
import StreamingController from './components/StreamingController';
import SignIn from './components/LoginController';
import Home from './pages/Home';
import HomeRoutes from './routes/HomeRoutes';

function App() {
    return (
        <div className="App">
            <header className="App-header">
                
                <HomeRoutes/>
            </header>
        </div>
    );
}

export default App;
