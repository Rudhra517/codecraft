# CodeCraft 🚀  

![GitHub stars](https://img.shields.io/github/stars/yourusername/codecraft?style=social)  
![GitHub forks](https://img.shields.io/github/forks/yourusername/codecraft?style=social)  
![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/codecraft)  
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/codecraft?color=red)  
[![Discord](https://img.shields.io/badge/Discord-CodeCraft-blue?logo=discord&logoColor=white)](https://discord.gg/yourdiscord)  
[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/yourusername)  

**CodeCraft** is an advanced **LLM training and deployment framework** that enables developers to **fine-tune, host, and interact with large language models**. The **CodeCraft model** is available on **Hugging Face** (`hf.co/rudra157/codecraft`) and can also be run locally via **Ollama**. The **frontend** is powered by **OpenWebUI**, offering a clean and interactive user interface.

---

![CodeCraft UI Demo](./demo.gif)  

## 🌟 Key Features  

- 🛠 **Train & Deploy Custom LLMs** – Fine-tune models on domain-specific datasets.  
- ⚡ **Seamless Model Hosting** – Supports **Hugging Face (Cloud)** and **Ollama (Local)**.  
- 🖥 **Intuitive Frontend** – Uses **OpenWebUI** for a modern UI.  
- 📄 **Optimized for Code Generation** – Fine-tuned for programming tasks.  
- 🔐 **Secure & Flexible** – Supports **Docker** and **Kubernetes** deployments.  

---

## 🏗 Tech Stack  

| Component     | Technology |
|--------------|------------|
| **Frontend** | OpenWebUI |
| **Backend**  | Python (FastAPI/Flask) |
| **LLM Hosting** | Hugging Face, Ollama (Optional) |
| **Training Framework** | PyTorch / TensorFlow |
| **Deployment** | Docker, Kubernetes (Optional) |

---

## 📦 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/codecraft.git
cd codecraft
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## 🖥️ OpenWebUI Setup  

### ➤ Install OpenWebUI Locally  
```bash
git clone https://github.com/openwebui/openwebui.git
cd openwebui
docker compose up -d
```
- OpenWebUI will be available at: `http://localhost:3000`

### ➤ Run OpenWebUI with Docker  
```bash
docker run -d --name openwebui -p 3000:3000 ghcr.io/openwebui/openwebui:latest
```
- Stop OpenWebUI:  
  ```bash
  docker stop openwebui
  ```
- Restart OpenWebUI:  
  ```bash
  docker start openwebui
  ```

---

## 🧠 Model Inference Options  

### 🔹 **Option 1: Use CodeCraft Model from Hugging Face (Cloud)**  
1. Sign up at [Hugging Face](https://huggingface.co/)  
2. Get your API key and update `.env`:  
```env
HUGGINGFACE_API_KEY=your_api_key_here
USE_OLLAMA=False  # Set to False to use Hugging Face
```

### 🔹 **Option 2: Run CodeCraft Model Locally via Ollama**  
#### ➤ Install Ollama Locally  
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### ➤ Download and Run CodeCraft Model  
```bash
ollama pull hf.co/rudra157/codecraft
ollama run hf.co/rudra157/codecraft
```

#### ➤ Use Ollama in Backend (`.env` Config)  
```env
USE_OLLAMA=True
OLLAMA_MODEL=hf.co/rudra157/codecraft
```

---

## 🐳 **Run Ollama with Docker**  

### ➤ Pull & Run Ollama  
```bash
docker run -d --name ollama -p 11434:11434 ollama/ollama
```

### ➤ Download & Run CodeCraft Model in Ollama (Docker)  
```bash
docker exec -it ollama ollama pull hf.co/rudra157/codecraft
docker exec -it ollama ollama run hf.co/rudra157/codecraft
```

### ➤ Stop Ollama Container  
```bash
docker stop ollama
```

---

## 🚀 Running CodeCraft  

### ➤ Start the Backend  
```bash
python app.py
```

### ➤ Open the Web UI  
```
http://localhost:3000
```

---

## 🔥 Advanced Features  

- **Train Custom Models**: Use `train.py` for fine-tuning LLMs.  
- **API Integration**: Connect with other services via `api.py`.  
- **Multi-Model Support**: Easily switch between Hugging Face and Ollama.  
- **Docker & Kubernetes Deployment**: Run CodeCraft at scale.  

---

## 📅 Roadmap  

✅ Cloud and local inference support  
🔄 More model training optimizations  
📈 Advanced analytics for LLM performance  
🔗 Integration with VS Code & Jupyter  

---

## 💡 Contributing  

We welcome contributions! Submit pull requests or open issues on GitHub.  

---

## 📝 License  

This project is licensed under **MIT License**.  
