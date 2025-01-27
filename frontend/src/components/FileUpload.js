

import React, { useState } from 'react';

function FileUpload() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload a MIDI file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:5000/predict/", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResults(data);
  };

  return (
    <div>
      <h1>MIDI Composer Classification</h1>
      <input type="file" accept=".mid" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Upload and Predict</button>

      {results && (
        <div>
          <h2>Results</h2>
          <img src={results.piano_roll_image} alt="Piano Roll" />
          <p><strong>K-Means Prediction:</strong> {results.kmeans_composer}</p>
          <p><strong>CNN Prediction:</strong> {results.cnn_composer}</p>
        </div>
      )}
    </div>
  );
}

export default FileUpload;

