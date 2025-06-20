# 🧠 ReasoningProt

A web-based tool for protein secondary structure classification using machine learning. Users can upload or input PDB IDs, select ML models, configure feature windows, and receive downloadable performance reports in PDF format.

---

## 🚀 Features

- 🔐 User registration & login (JWT-based auth)
- 🧬 Input PDB IDs manually or via file
- ⚙️ Select models to train (e.g., Decision Tree, SVM, Random Forest, etc.)
- 📊 See classification results (F1, precision, recall, support)
- 📄 Automatically generate downloadable PDF reports
- 📁 View and download previous reports in **My Reports**

---

## 🐳 How to Run with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/GoncaloDomingues750/thesis.git
   cd thesis
   ```

2. **Build and start the containers**
   ```bash
   cd docker
   docker-compose up --build
   ```

 3. **Run the project after first time build**
     ```bash
     cd docker
     docker-compose up
     ```

4. **Access the application**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Backend API: [http://localhost:8000](http://localhost:8000)

---

## 🧪 Usage Instructions

### 1. Register/Login
- Go to [http://localhost:3000](http://localhost:3000)
- Register a new account or log in with existing credentials

### 2. Run Predictions
- Enter comma-separated PDB IDs (e.g., `1CRN, 2HHB`)
- Or upload a `.txt` file with PDB IDs
- Choose how many amino acids before and inside helices to consider
- Select one or more machine learning models to train
- Click **Submit** and wait for results + automatic PDF download

### 3. My Reports
- Navigate to **My Reports** to see all previously generated PDFs
- Click **Download** to retrieve them

### 4. Logout
- Use the **Logout** button in the bottom-right corner of the dashboard

---

## 🛠️ Tech Stack

- **Frontend**: React + SCSS
- **Backend**: FastAPI + MongoDB + GridFS
- **ML Models**: scikit-learn (SVM, RF, KNN, etc.)
- **Authentication**: JWT (Token stored in localStorage)
- **PDF Reports**: Matplotlib + PdfPages
- **Dockerized** for reproducibility

---

## 📂 Project Structure

```
project/
├── backend/
│   ├── app/
│   │   ├── auth/
│   │   ├── ml/
│   │   ├── pdb_utils.py
│   │   ├── routes.py
│   │   └── main.py
├── frontend/
│   ├── src/
│   ├── public/
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.frontend
│   ├── Dockerfile.backend
```

---

## 📌 Notes

- Make sure Docker is installed and running.
- First-time PDF generation may take time due to ML training.
- The PDB files are fetched live using the PDB API.

---

## 📧 Contact

Feel free to open an issue or contact [goncalo.jose.domingues@gmail.com] if you have questions or feedback.
