# A/B Testing Playground — Frequentist & Bayesian 🚀

> Interactive web app to explore A/B testing, hypothesis testing, Bayesian inference, and power analysis — with simulated data and a friendly UI.

---

## 🔍 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Demo](#demo)  
- [Tech Stack](#tech-stack)  
- [Getting Started (Local)](#getting-started-local)  
- [Docker Deployment](#docker-deployment)  
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🧠 Overview

This project is a hands-on demonstration of A/B testing workflows using simulated data. It combines classical statistics (z-tests, chi-square, Welch’s t-test), **Wilson confidence intervals**, and **Bayesian Beta–Bernoulli inference** to help you:

- Compare two variants (A and B) on conversion or revenue.
- Quantify lift and uncertainty via confidence intervals and posteriors.
- Run power / sample size (MDE) planning.
- Interactively explore changes through a web interface.

Its goal is to bridge **Probability & Statistics theory** and **practical DevOps/data-science workflows** (Python, Streamlit, Docker).

---

## ⚙️ Features

- Simulated conversion and (optional) revenue data  
- Two-proportion **z-test** + **chi-square test**  
- **Wilson score intervals** for better proportion CIs  
- **Welch’s t-test** & confidence interval for revenue differences  
- **Bayesian posterior** of lift (`P(p_B > p_A)`)  
- **Power / Sample Size / MDE** calculators  
- Exportable via container (Docker) or live deploy (Streamlit Cloud)  

---

## 🔗 Demo

After deployment, your app will be available at a stable URL. (e.g. `https://<username>-<repo>.streamlit.app`)  
Use this UI to simulate experiments and experiment with parameters in real time.

---

## 🧰 Tech Stack

- **Python** (≥ 3.8)  
- **Streamlit** for the interactive web app  
- **NumPy, SciPy, Pandas, Matplotlib** for core stats & data  
- **Docker** for containerized deployment  
- (Optional) **ngrok** for ephemeral tunnels during development  

---

## 🚀 Getting Started (Local)

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USER>/A-B-Testing-Playground.git
   cd A-B-Testing-Playground
