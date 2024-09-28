import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import App from './App'

const Main: React.FC = () => {
  const [loginID, setLoginId] = useState<string | null>(null);

  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            (
              <App loginID={loginID} />
            )
          }
        />
        <Route
          path="/tar"
          element={<App loginID={loginID} />}
        />
        <Route
          path="*"
          element={<Navigate to="/" />}
        />
      </Routes>
    </Router>
  );
};

export default Main;