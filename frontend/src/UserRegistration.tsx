import React, { useState } from 'react';
import axios from 'axios';

const UserRegistration = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [registrationStatus, setRegistrationStatus] = useState<string | null>(null);

  const handleEmailChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setEmail(event.target.value);
  };

  const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPassword(event.target.value);
  };

  const handleRegisterClick = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_BACKEND_URL}/api/register`, { email, password });
      setRegistrationStatus('success');
      console.log('Registration successful!', response.data);
    } catch (error) {
      setRegistrationStatus('error');
      console.error('Failed to register:', error);
    }
  };

  let statusMessage = null;
  if (registrationStatus === 'success') {
    statusMessage = <div>Registration successful! Please check your email to verify your account.</div>;
  } else if (registrationStatus === 'error') {
    statusMessage = <div>Failed to register. Please try again.</div>;
  }

  return (
    <div>
      <input
        type="email"
        value={email}
        onChange={handleEmailChange}
        placeholder="Enter your email"
        className="form-input mb-2"
      />
      <input
        type="password"
        value={password}
        onChange={handlePasswordChange}
        placeholder="Enter your password"
        className="form-input mb-2"
      />
      <button onClick={handleRegisterClick} className="mt-2 bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
        Register
      </button>
      {statusMessage}
    </div>
  );
};

export default UserRegistration;