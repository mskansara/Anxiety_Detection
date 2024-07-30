import React from 'react'
import { Routes, Route, BrowserRouter } from 'react-router-dom';
import Home from '../pages/Home';
import Session from '../pages/Session';
import SignIn from '../components/LoginController';

function HomeRoutes() {
    return (
        <>
            <BrowserRouter>
                <Routes>
                    <Route path="/login" exact element={<SignIn />} />
                    <Route path="/" exact element={<Home />} />
                    <Route path="/session" exact element={<Session />} />
                </Routes>
            </BrowserRouter>
        </>
    )
}

export default HomeRoutes