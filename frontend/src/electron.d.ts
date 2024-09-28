declare interface Window {
    electron: {
        openLinkedInLogin: (url: string) => void;
        onLoginSuccess: (callback: (event: Event, message: string) => void) => void;
    };
}