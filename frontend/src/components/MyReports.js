import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const MyReports = () => {
  const [reports, setReports] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    fetch("http://localhost:8000/api/list_reports", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then(res => res.json())
      .then(data => setReports(data))
      .catch(err => alert("Failed to fetch reports: " + err));
  }, []);

  const download = (fileId, filename) => {
    fetch(`http://localhost:8000/api/download_report/${fileId}`)
      .then(res => res.blob())
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      });
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h2>📄 My Reports</h2>
        <button className="back-button" onClick={() => navigate("/dashboard")}>
          ← Back to Dashboard
        </button>
      </div>

      {reports.length === 0 ? (
        <p>No reports found.</p>
      ) : (
        <ul>
          {reports.map(r => (
            <li key={r.file_id} style={{ marginBottom: "1rem" }}>
              <strong>{r.filename}</strong> ({new Date(r.upload_date).toLocaleString()}){" "}
              <button onClick={() => download(r.file_id, r.filename)}>Download</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default MyReports;
