"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
function createMainWindow() {
    const mainWindow = new electron_1.BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: false
        },
    });
    mainWindow.loadURL('http://localhost:3000'); // URL of your React app
}
electron_1.app.whenReady().then(createMainWindow);
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
electron_1.app.on('activate', () => {
    if (electron_1.BrowserWindow.getAllWindows().length === 0) {
        createMainWindow();
    }
});
// IPC listener to handle LinkedIn login
electron_1.ipcMain.on('open-linkedin-login', (event, url) => {
    const linkedinWindow = new electron_1.BrowserWindow({
        width: 600,
        height: 600,
        webPreferences: {
            webSecurity: false,
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
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
    electron_1.ipcMain.on('message', (_event, message) => {
        if (message === 'linkedin-login-success') {
            event.sender.send('login-success', 'linkedin');
        }
    });
});
