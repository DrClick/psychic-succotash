import React, { useState } from 'react';
import './FileUpload.css';

function FileUpload() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload a MIDI file first.");
      return;
    }

    setLoading(true);
    setLogs([]); // Clear logs for a fresh submission
    const taskId = crypto.randomUUID(); // Generate unique task ID

    // Connect to WebSocket for task-specific logs
    const ws = new WebSocket(`ws://localhost:5000/logs/${taskId}`);
    ws.onmessage = (event) => {
      setLogs((prevLogs) => [...prevLogs, event.data]);
    };
    ws.onclose = () => console.log("WebSocket closed.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("task_id", taskId); // Include task ID in the request

    try {
      const response = await fetch("http://localhost:5000/predict/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error during file upload:", error);
    } finally {
      setLoading(false);
      ws.close(); // Close WebSocket after receiving backend response
    }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>MIDI Composer Classification</h1>
      </header>

      <div className="main-layout">
        {/* Left Column: Input and Classification */}
        <div className="left-column">
          <div className="input-section">
            <h2>Upload MIDI File</h2>
            <div className="file-input">
              <input type="file" accept=".mid" onChange={handleFileChange} />
              <button onClick={handleSubmit} disabled={loading}>
                {loading ? "Processing..." : "Upload and Predict"}
              </button>
            </div>
          </div>

          <div className="classification-results">
            <h2>Classification Results</h2>
            {results ? (
              <table>
                <thead>
                  <tr>
                    <th>Classifier</th>
                    <th>Prediction</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>CNN</td>
                    <td>{results.CNN_prediction}</td>
                  </tr>
                  <tr>
                    <td>K-Means</td>
                    <td>{results.KMEANS_prediction}</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <p>Results will appear here after prediction.</p>
            )}
          </div>
        </div>

        {/* Center Column: Main Piano Roll */}
        <div className="center-column">
          {results ? (
            <img
              src={results.file_name}
              alt="Piano Roll"
              className="piano-roll"
            />
          ) : (
            <div className="placeholder">
              <p>Piano Roll will appear here after prediction.</p>
              {/* Logs Section */}
              <div className="logs-section">
                <ul>
                  {logs.map((log, index) => (
                    <li key={index}>{log}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Frames */}
        <div className="right-column">
          <h3>Extracted Frames</h3>
          <div className="frames-container">
            {results
              ? results.frames
                  .map((frame, index) => ({ ...frame, originalIndex: index }))
                  .sort((a, b) => a.indicies[0] - b.indicies[0])
                  .map((frame) => (
                    <div className="frame-card" key={frame.originalIndex}>
                      <img
                        src={frame.file_name}
                        alt={`Frame ${frame.originalIndex + 1}`}
                      />
                      <div>
                        <p>
                          <strong>Frame {frame.originalIndex + 1}</strong>
                        </p>
                        <p>Start: {frame.indicies[0]}</p>
                        <p>End: {frame.indicies[1]}</p>
                      </div>
                    </div>
                  ))
              : (
                <div className="placeholder">
                  <p>Frames will appear here after prediction.</p>
                </div>
              )}
          </div>
        </div>
      </div>
    </div>
    
  );
}

export default FileUpload;
