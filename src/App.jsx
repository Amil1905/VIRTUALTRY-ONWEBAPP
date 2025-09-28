// src/App.jsx

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import TestComparison from './components/TestComparison';

function App() {
  return (
    <Router>
      <div className="App">
        {/* Navigation Bar */}
        <nav className="bg-gray-800 p-4">
          <div className="max-w-7xl mx-auto flex gap-6">
            <Link to="/" className="text-white hover:text-blue-400 font-semibold">
              Virtual Try-On
            </Link>
          </div>
        </nav>

        {/* Page Routes */}
        <Routes>
          <Route path="/" element={<TestComparison />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;