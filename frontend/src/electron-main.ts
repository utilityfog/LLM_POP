import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';

function createMainWindow() {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: false,
      nodeIntegration: false,
      sandbox: false
    },
  });

  mainWindow.loadURL('http://localhost:3000'); // URL of your React app
}

app.whenReady().then(createMainWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

// IPC listener to handle LinkedIn login
ipcMain.on('open-linkedin-login', (event, url) => {
  const linkedinWindow = new BrowserWindow({
    width: 600,
    height: 600,
    webPreferences: {
      webSecurity: false,
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: false,
      nodeIntegration: false,
      sandbox: false
    },
  });

  linkedinWindow.loadURL(url);

  const script = `
    (function() {
      const checkLoginStatus = () => {
        try {
          if (window.location.href.includes('/feed')) {
            window.opener.postMessage('linkedin-login-success', '*');
            window.close();
          }
        } catch (e) {
          console.error(e);
        }
      };
      setInterval(checkLoginStatus, 1000);
    })();
  `;

  linkedinWindow.webContents.on('did-finish-load', () => {
    linkedinWindow.webContents.executeJavaScript(`
      document.write(\`
        <html>
          <head>
            <script>${script}</script>
          </head>
          <body>
            <iframe src='https://www.linkedin.com/login' frameborder="0" width="100%" height="100%"></iframe>
          </body>
        </html>
      \`);
    `);
  });

  linkedinWindow.on('closed', () => {
    // linkedinWindow = null;
  });

  ipcMain.on('message', (_event: Electron.IpcMainEvent, message: string) => {
    if (message === 'linkedin-login-success') {
      event.sender.send('login-success', 'linkedin');
    }
  });
});

// Add an empty export to mark this file as a module
export {};