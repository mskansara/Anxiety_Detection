import React from 'react';
import { Navigate } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';

const ProtectedRoute = ({ children, allowedRoles }) => {
    const token = localStorage.getItem('token');
    const userRole = localStorage.getItem('role')

    if (!token) {
        return <Navigate to="/login" />;
    }

    try {
        const decodedToken = jwtDecode(token);


        if (allowedRoles.includes(userRole)) {
            return children;
        } else {
            return <Navigate to="/" />;
        }
    } catch (error) {
        return <Navigate to="/login" />;
    }
};

export default ProtectedRoute;