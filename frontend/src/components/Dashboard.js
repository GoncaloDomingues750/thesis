import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./AuthPages.scss";

const Dashboard = () => {
  const [pdbIds, setPdbIds] = useState("");
  const [nBefore, setNBefore] = useState(3);
  const [nInside, setNInside] = useState(4);
  const [selectedModels, setSelectedModels] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const availableModels = [
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN",
    "Gradient Boosting",
    "Extra Trees",
    "Stacking",
  ];

  const toggleModel = (model) => {
    setSelectedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
  };

  const handleSubmit = async () => {
    setLoading(true);
    setResult(null);
    const token = localStorage.getItem("token");

    try {
      const response = await fetch("http://localhost:8000/api/submit_ids", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({
          pdb_ids: pdbIds.split(",").map((id) => id.trim()),
          n_before: nBefore,
          n_inside: nInside,
          models: selectedModels,
        }),
      });

      const data = await response.json();
      setResult(data);

      if (data.pdf_id) {
        const pdfResponse = await fetch(`http://localhost:8000/api/download_report/${data.pdf_id}`);
        const blob = await pdfResponse.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "evaluation_summary.pdf";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (error) {
      alert("Something went wrong: " + error);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("token");
    navigate("/login");
    window.location.reload();
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      setPdbIds(text.trim());
    };
    reader.readAsText(file);
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h2>Protein Secondary Structure Classifier</h2>
      </div>

      <div className="form-group">
        <label>PDB IDs (comma-separated):</label>
        <textarea
          value={pdbIds}
          onChange={(e) => setPdbIds(e.target.value)}
          placeholder="e.g. 1CRN, 2HHB, 4HHB"
        />
      </div>

      <div className="form-group">
        <label>Or Upload PDB ID File:</label>
        <input type="file" accept=".txt" onChange={handleFileUpload} />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>N Before Helix:</label>
          <input
            type="number"
            value={nBefore}
            onChange={(e) => setNBefore(Number(e.target.value))}
            min="0"
          />
        </div>

        <div className="form-group">
          <label>N Inside Helix:</label>
          <input
            type="number"
            value={nInside}
            onChange={(e) => setNInside(Number(e.target.value))}
            min="1"
          />
        </div>
      </div>

      <div className="form-group">
        <label>Select Models to Train:</label>
        <div className="model-grid">
          {availableModels.map((model) => (
            <label key={model} className="checkbox-label">
              <input
                type="checkbox"
                checked={selectedModels.includes(model)}
                onChange={() => toggleModel(model)}
              />
              {model}
            </label>
          ))}
        </div>
      </div>

      <div className="button-row">
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Running models..." : "Submit"}
        </button>

        <button onClick={() => navigate("/my-reports")} className="reports-button">
          My Reports
        </button>

        <button onClick={logout} className="logout-button">
          Logout
        </button>
      </div>



      {result?.metrics && (
        <div className="results-container">
          <h3>Evaluation Results</h3>
          {result.metrics.map((m) => (
            <div key={m.model} className="result-block">
              <h4>{m.model}</h4>
              <table className="report-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-score</th>
                    <th>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(m.classification_report)
                    .filter(([label]) => label !== "accuracy")
                    .map(([label, scores]) => (
                      <tr key={label}>
                        <td>{label}</td>
                        <td>{scores.precision?.toFixed(2) ?? '-'}</td>
                        <td>{scores.recall?.toFixed(2) ?? '-'}</td>
                        <td>{scores["f1-score"]?.toFixed(2) ?? '-'}</td>
                        <td>{scores.support ?? '-'}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              <p>
                <strong>Average F1 Score:</strong> {m.avg_f1.toFixed(4)}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
