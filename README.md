<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=200&section=header&text=ML%20Research%20Engineer&fontSize=50&fontColor=fff&animation=fadeIn&fontAlignY=35&desc=Computer%20Vision%20%7C%20Multi-modal%20AI%20%7C%20Scalable%20Systems&descAlignY=51&descSize=20" width="100%"/>

<a href="https://git.io/typing-svg">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=1000&color=667EEA&center=true&vCenter=true&width=800&lines=Bridging+research+and+real-world+AI+systems;Computer+Vision+%26+Multi-modal+Learning;Distributed+Training+%7C+DeepSpeed+%7C+MoE;LLMs+%26+RAG+Systems+with+grounding+and+evaluation" />
</a>

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)]()
[![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)]()

</div>

---

## 🎯 About Me

**Machine Learning Research Engineer** working at the intersection of **computer vision, multi-modal AI, and scalable machine learning systems**.

My work focuses on **bridging research and deployment** — designing models that are not only performant, but **robust, interpretable, and usable in real-world environments**.

- 🧠 MSc in Machine Learning @ MBZUAI (GPA 3.85)  
- 📊 Strong theoretical grounding: probabilistic inference, mathematical foundations, distributed ML systems  
- ⚡ Systems focus: scaling training (DeepSpeed, MoE, multi-node) and production pipelines  
- 🚀 Founder & builder of deployed AI systems used in real workflows  

---

## 🔬 Research Interests

I’m broadly interested in **human-centered and reliable AI systems**, with current directions including:

- **Multi-modal learning** (vision + language + structured data)
- **Computer vision for real-world environments** (robustness, noise, domain shift)
- **LLM reliability & reasoning integrity** (grounding, hallucination mitigation)
- **Human–AI interaction in high-stakes settings**
- **Scalable training systems** for large models (MoE, distributed optimization)

---

## 💻 Technical Expertise

### 👁️ Computer Vision
- Vision Transformers (ViT, Swin), CNNs, hybrid architectures  
- Segmentation, classification, detection, multi-task learning  
- Representation learning under real-world constraints  

### 🤖 Multi-modal & LLM Systems
- RAG pipelines, retrieval + grounding strategies  
- Vision-Language Models (VLMs)  
- Evaluation of reasoning, safety, and failure modes  

### ⚡ Scalable & Distributed ML
- DeepSpeed, ZeRO, torchrun, multi-node training  
- Mixture of Experts (MoE), model/data parallelism  
- Throughput optimization and large-scale experimentation  

### 🔧 Systems & Deployment
- End-to-end ML pipelines (data → training → deployment → monitoring)  
- FastAPI, Docker, AWS, CI/CD  
- Experiment tracking, reproducibility, dataset versioning  

---

## 🧪 Selected Work & Projects

- **Distributed Training with Mixture of Experts**  
  Designed and trained CNN-based MoE systems using DeepSpeed across multiple GPUs, exploring model parallelism and throughput scaling.

- **Multi-task Computer Vision for Structured Prediction**  
  Built hybrid architectures (e.g., Swin-UNet variants) for joint segmentation and classification under noisy real-world data conditions.

- **Multi-modal AI Systems (Vision + Text)**  
  Developed pipelines integrating visual inputs with contextual information for decision-support systems.

- **LLM Reliability & Grounding**  
  Investigating methods to reduce hallucinations and improve reasoning integrity through retrieval and constraint mechanisms.

---

## ⚡ Current Work

- Designing **multi-modal AI systems** that combine vision, text, and structured data  
- Studying **failure modes in LLM reasoning** and grounding strategies  
- Scaling **training pipelines for large models** using distributed systems  

---

## 💡 Research Engineering Philosophy

```python
class ResearchEngineer:
    def __init__(self):
        self.goal = "Bridge theory, systems, and real-world impact"
    
    def approach(self, problem):
        formulation = self.formalize(problem)          # Define learning objective
        data = self.construct_dataset(formulation)     # Real-world constraints
        
        model = self.train(
            data,
            design="scalable + principled",
            tools=["PyTorch", "DeepSpeed"]
        )
        
        evaluation = self.evaluate(
            model,
            criteria=["performance", "robustness", "failure_modes"]
        )
        
        return self.deploy_if_valid(evaluation)
