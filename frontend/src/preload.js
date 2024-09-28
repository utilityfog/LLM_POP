const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  openLinkedInLogin: (url) => ipcRenderer.send('open-linkedin-login', url),
  onLoginSuccess: (callback) => ipcRenderer.on('login-success', callback)
});