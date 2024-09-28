import React, {useEffect, useState, useCallback} from 'react';
import './App.css';
import {fetchEventSource} from "@microsoft/fetch-event-source";
import tar_logo from './tar.png';
import { NotebookRender } from '@nteract/notebook-render';
import { fromJS } from 'immutable';

interface Message {
  message: string;
  isUser: boolean;
  sources?: string[];
  notebook?: object;
}

interface AppProps {
  loginID: string | null;
}

const App: React.FC<AppProps> = ({ loginID }) => {
  const [inputValue, setInputValue] = useState("")
  const [messages, setMessages] = useState<Message[]>([]);

  const setPartialMessage = (chunk: string, sources: string[] = []) => {
    setMessages(prevMessages => {
      const lastMessage = prevMessages[prevMessages.length - 1];
      // Check if the previous message was from the user or if there are no previous messages
      if (prevMessages.length === 0 || lastMessage.isUser) {
        return [...prevMessages, { message: chunk, isUser: false, sources }];
      }
  
      // Otherwise, append chunk to the last message
      return [...prevMessages.slice(0, -1), {
        ...lastMessage,
        message: lastMessage.message + chunk,
        sources: lastMessage.sources ? [...lastMessage.sources, ...sources] : sources,
      }];
    });
  }

  function formatMessageForDisplay(message: string): string {
    // Replace line breaks and indentation from JSON format to HTML format
    return message
      .replace(/\\n/g, '\n') // Replaces escaped newlines with actual newlines
      .replace(/\\t/g, '\t'); // Replaces escaped tabs with actual tabs
  }

  function truncateString(str: string, num: number) {
    if (str.length <= num) {
      return str;
    }
    return str.slice(0, num) + '...';
  }

  function handleReceiveMessage(data: string) {
    let parsedData = JSON.parse(data);
    console.log(data)
    // console.log("YAYY!")

    if (parsedData.answer) {
      // const formattedMessage = JSON.stringify(parsedData.answer.content, null, 2);
      setPartialMessage(formatMessageForDisplay(parsedData.answer.content))
    }

    if (parsedData.docs) {
      //const formattedSources = parsedData.docs.map((doc: any) => JSON.stringify(doc.metadata.source, null, 2));
      let formattedSources = parsedData.docs.map((doc: any) => 
        formatMessageForDisplay(doc.metadata.source)
      ) //.join('\n'); // Join array elements with newline
      let formattedSourcesFinal = ["Context Sources\n"]
      formattedSourcesFinal.push(...formattedSources);
      setPartialMessage("", formattedSourcesFinal)
    }

    if (parsedData.augments) {
      console.log(parsedData.augments);
      // Flatten the array of arrays and map to formatted sources and page content
      var formattedAugmentSources = parsedData.augments.flatMap((augmentArray: any) => 
        augmentArray.map((augment: any) => {
          const formattedSource = formatMessageForDisplay(augment.metadata.source);
          const formattedContent = truncateString(formatMessageForDisplay(augment.page_content), 100); // Truncate to 100 characters
          return `${formattedSource}\n${formattedContent}`;
        })
      )
      let formattedAugmentSourcesFinal = ["Augment Sources\n"]
      formattedAugmentSourcesFinal.push(...formattedAugmentSources);
      // Set the formatted string as the partial message
      setPartialMessage("", formattedAugmentSourcesFinal);
    }
  
  }

  const [currentEventSource, setCurrentEventSource] = useState<EventSource | null>(null);

  const closeCurrentEventSource = useCallback(() => {
    if (currentEventSource) {
      currentEventSource.close();
      setCurrentEventSource(null);
    }
  }, [currentEventSource]);

  // Add a state for session ID
  const [sessionId, setSessionId] = useState<string>();

  useEffect(() => {
    // Generate or retrieve a session ID when the component mounts
    // This example just generates a simple random string. You might want to use a more robust method.
    const generateSessionId = () => {
      return Math.random().toString(36).substring(2, 15);
    };

    setSessionId(generateSessionId());
  }, []);

  // Change this to an array to hold multiple file IDs
  const [uploadedFileIds, setUploadedFileIds] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  // Bool for context fetching
  const [fetchContext, setFetchContext] = useState(true); // Default to true


  const handleSendMessage = async (message: string) => {
    setInputValue("")
    closeCurrentEventSource(); // Close the previous event source if open

    setIsLoading(true); // Start loading

    let endpoint = '/rag/stream'; // Default endpoint
    if (isMode === 'solve') {
      endpoint = '/preprocess/stream';
    } else if (isMode === 'research') {
      endpoint = '/research/stream';
    }

    const messagePayload = {
      message: formatMessageForDisplay(message),
      isUser: true,
      ...(uploadedFileIds ? {file_embedding_keys: uploadedFileIds} : {}) // Include all file IDs that have been uploaded
    };
  
    setMessages(prevMessages => [...prevMessages, messagePayload]);
  
    const bodyPayload = {
      input: {
        question: message,
        file_embedding_keys: uploadedFileIds || {}, // Include the file_embedding_keys if they exist.
        session_id: sessionId || {},
        fetchContext: fetchContext,
      }
    };
  
    console.log("Sending message with payload:", bodyPayload);

    const newEventSource = await fetchEventSource(`${process.env.REACT_APP_BACKEND_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(bodyPayload),
      onmessage(event) {
        if (event.event === "data") {
          handleReceiveMessage(event.data);
        }
        setIsLoading(false); // Stop loading when data is received
      },
      onerror(error) {
        console.error('EventSource failed:', error);
        // Handle the error here, such as updating the UI to inform the user
        setIsLoading(false); // Stop loading when data is received
      },
    });

    setCurrentEventSource(newEventSource as unknown as EventSource);

    if (uploadedFileIds) {
      setUploadedFileIds([]); // Reset the lastUploadedFileId after sending the message
    }
  }

  const handleKeyPress = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      // handleSendMessage(inputValue.trim())
    }
  }

  function formatSource(source: string) {
    // source.split("/").pop() || "";
    //return JSON.stringify(source, null, 2);
    return formatMessageForDisplay(source);
  }

  // Add a state hook for handling file uploads
  const [uploadedFiles, setUploadedFiles] = useState<FileList | null>(null);

  // Function to handle file selections
  function handleFileSelect(event: React.ChangeEvent<HTMLInputElement>) {
    setUploadedFiles(event.target.files);
    // Here you would handle uploading the files, possibly using a separate API endpoint
  }

  const handleUploadFiles = async (uniqueId: string, filename: string) => {
    closeCurrentEventSource(); // Ensure no stream is open before setting a new one for file upload

    setUploadedFileIds(prevIds => [...prevIds, uniqueId]);
  
    // Show some feedback to the user about the file upload
    setMessages(prevMessages => [
      ...prevMessages,
      { message: `File ${filename} uploaded successfully with ID: ${uniqueId}`, isUser: false },
    ]);
  
    console.log(`File uploaded with ID: ${uniqueId}`);
  };

  useEffect(() => {
    // Close event source when the component unmounts
    return () => closeCurrentEventSource();
  // Pass closeCurrentEventSource as a dependency if it's stable and won't change on each render
  }, [closeCurrentEventSource]);

  // Function to handle file upload form submission
  // When handling the file upload, include the unique_id in the message sent to the server
  const handleUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (uploadedFiles) {
      const formData = new FormData();
      Array.from(uploadedFiles).forEach(file => {
        formData.append('files', file);
      });

      let endpoint = '/upload'; // Default endpoint
      if (isMode === 'solve') {
        endpoint = '/solve_jupyter';
      } else if (isMode === 'research') {
        endpoint = '/upload'; // Assuming this is the endpoint for research mode
      }

      // console.log(formData.get('files'));
      const headers = new Headers();
      if (sessionId) {
        headers.append('session-id', sessionId);
      }

      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: headers,
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();

      // If endpoint was /upload, data.files is expected to be an array of file information
      data.files.forEach((fileInfo: { unique_id: string; filename: string; }) => {
        handleUploadFiles(fileInfo.unique_id, fileInfo.filename);
      });

      // If endpoint was /solve_jupyter, data.notebook is returned from server.py as NotebookNode or None
      if (data.solved_notebook) {
        const notebook = data.solved_notebook;
        setMessages(prevMessages => [...prevMessages, { message: "Solved Jupyter Notebook received:", isUser: false, notebook }]);
      }

    }
  };

  // Spinner component
  function Spinner() {
    return (
      <div className="spinner-container">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  const [isMode, setIsMode] = useState('rag');  // Default to 'Chat Mode'

  const downloadNotebook = (notebook: object | undefined) => {
    if (notebook) {
      const notebookJson = JSON.stringify(notebook);
      const blob = new Blob([notebookJson], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'solved_notebook.ipynb';
      link.click();
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <header className="bg-gray-800 text-white text-center p-4">
        <img src={tar_logo} alt="Tar Logo" style={{width: '25%', height: 'auto'}} className="mx-auto" /> {/* Logo will be centered */}
        <h1 style={{ fontFamily: "Origin", color: "#a2fcf5"}} className="text-6xl font-bold">Tar GPT</h1> {/* Increased font size */}
        {/* <h3 style={{ fontFamily: "Origin", color: "#FFFFFF"}} className="text-2xl font-bold">A Research and Reading Assistant that Aids Dyslexic or Neurodivergent Individuals Intuitively Understand the Latest Cutting Edge Technologies Discussed in Recent Research Papers</h3> */}
      </header>
      <main className="flex-grow container mx-auto p-4 flex-col">
        <div className="flex-grow bg-gray-700 shadow overflow-hidden sm:rounded-lg">
          <div className="border-b border-gray-600 p-4">
            {messages.map((msg, index) => (
              <div key={index} 
                  // user-message, llm-response
                  className={`message-container ${msg.isUser ? "bg-gray-800" : "bg-gray-900"}`}>
                <pre>{msg.message}</pre> {/* Use <pre> to preserve whitespace and formatting */}
                {/*  Source */}
                {!msg.isUser && (
                  <div className={"text-xs"}>
                    <hr className="border-b mt-5 mb-5"></hr>
                    {msg.sources?.map((source, index) => (
                      <div>
                        <a
                          target="_blank"
                          download
                          href={`${process.env.REACT_APP_BACKEND_URL}/rag/static/${encodeURI(formatSource(source))}`}
                          rel="noreferrer"
                        >{formatSource(source)}</a>
                      </div>
                    ))}
                  </div>
                )}
                {!msg.isUser && msg.notebook && (
                  <div>
                    <NotebookRender notebook={fromJS(msg.notebook)} />
                    <button
                      className="mt-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                      onClick={() => downloadNotebook(msg.notebook)}
                    >
                      Download Notebook
                    </button>
                  </div>
                )}
              </div>
            ))}
            {isLoading && <Spinner />} {/* Loading spinner displayed when isLoading is true */}
          </div>
          <div className="p-4 bg-gray-800">
            <textarea
              className="form-textarea w-full p-2 border rounded text-white bg-gray-900 border-gray-600 resize-none h-auto"
              placeholder="Enter your message here..."
              onKeyUp={handleKeyPress}
              onChange={(e) => setInputValue(e.target.value)}
              value={inputValue}
            ></textarea>
            <button
              className="mt-2 bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
              onClick={() => handleSendMessage(inputValue.trim())}
            >
              Send
            </button>

            {/* File uploader form */}
            <form onSubmit={handleUpload}>
              <input
                type="file"
                multiple
                onChange={handleFileSelect}
                accept=".png,.jpg,.jpeg,.pdf,.ipynb,.csv,.py,.js" // Accepted file types
                className="form-input mb-2"
              />
              <button
                type="submit"
                className="mt-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              >
                Upload
              </button>
            </form>

            {/* Mode Selector */}
            {/* <label className="switch">
              <input
                type="checkbox"
                checked={isSolveMode}
                onChange={() => setIsSolveMode(!isSolveMode)}
              />
              <span className="slider round"></span>
            </label>
            <span>{isSolveMode ? 'Solve Mode' : 'Chat Mode'}</span> */}
            <select
              value={isMode}
              onChange={(e) => setIsMode(e.target.value)}
              className="bg-gray-800 text-white p-2 rounded"
            >
              <option value="chat">Chat Mode</option>
              <option value="solve">Solve Jupyter Mode</option>
              <option value="research">Research Mode</option>
              <option value="search">Search Mode</option>
            </select>
            <div className="toggle-container">
              <label>
                Fetch Context:
                <input
                  type="checkbox"
                  checked={fetchContext}
                  onChange={() => setFetchContext(!fetchContext)}
                />
              </label>
            </div>
          </div>
        </div>

      </main>
      <footer className="bg-gray-800 text-white text-center p-4 text-xs">
        *AI Agents can make mistakes. Consider checking important information.
        <br/>
        All training data derived from public records
        <br/>
        <br/>
        © 2024 Hydroxide Holdings
      </footer>

    </div>
  );
};

export default App;