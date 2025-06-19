import React, { useState } from "react";
import "./AuthPages.scss";

const Dashboard = () => {
  const [pdbIds, setPdbIds] = useState("");
  const [nBefore, setNBefore] = useState(3);
  const [nInside, setNInside] = useState(4);
  const [selectedModels, setSelectedModels] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
    try {
      const response = await fetch("http://localhost:8000/api/submit_ids", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pdb_ids: pdbIds.split(",").map((id) => id.trim()),
          n_before: nBefore,
          n_inside: nInside,
          models: selectedModels,
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Something went wrong: " + error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <h2>Protein Secondary Structure Classifier</h2>

      <label>PDB IDs (comma-separated):</label>
      <textarea
        value={pdbIds}
        onChange={(e) => setPdbIds(e.target.value)}
        placeholder="e.g. 1CRN, 2HHB, 4HHB"
      />

      <label>N Before Helix:</label>
      <input
        type="number"
        value={nBefore}
        onChange={(e) => setNBefore(Number(e.target.value))}
        min="0"
      />

      <label>N Inside Helix:</label>
      <input
        type="number"
        value={nInside}
        onChange={(e) => setNInside(Number(e.target.value))}
        min="1"
      />

      <label>Select Models to Train:</label>
      {availableModels.map((model) => (
        <label key={model}>
          <input
            type="checkbox"
            checked={selectedModels.includes(model)}
            onChange={() => toggleModel(model)}
          />
          {model}
        </label>
      ))}

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Running models..." : "Submit"}
      </button>

      {result?.metrics && (
        <div className="results-container">
          <h3>Evaluation Results</h3>
          {result.metrics.map((m) => (
            <div key={m.model} className="result-block">
              <h4>{m.model}</h4>
              <pre>{m.classification_report}</pre>
              <p><strong>Average F1 Score:</strong> {m.avg_f1.toFixed(4)}</p>
            </div>
          ))}
          <a
            href="/output_csvs/evaluation_summary.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="download-link"
          >
            📄 Download PDF Report
          </a>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
