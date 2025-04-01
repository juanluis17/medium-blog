# 🚀 Jaeger, Prometheus & Grafana with Docker Compose

This repository contains the **Docker Compose** configuration to deploy an observability stack with:

- **Jaeger**: Distributed tracing for microservices.
- **Prometheus**: Monitoring and metrics collection.
- **Grafana**: Data visualization.

## 📌 Getting Started

### 1️⃣ Prerequisites

Ensure you have the following installed:
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/juanluis17/medium-blog
cd observability/simple
```

### 3️⃣ Deploy the Stack
```bash
docker-compose up -d
```

### 4️⃣ Verify the Services

- **Jaeger UI** → [http://localhost:16686](http://localhost:16686)
- **Prometheus** → [http://localhost:9090](http://localhost:9090)
- **Grafana** → [http://localhost:3000](http://localhost:3000) (User: `admin`, Password: `admin`)

## 📊 Creating Dashboards
You can create **Grafana dashboards** that visualize data from **Prometheus (metrics)** and **Jaeger (traces)**. Use the following data sources in Grafana:

- **Prometheus** → `http://prometheus:9090`
- **Jaeger** → `http://jaeger:16686`

## 📜 Configuration Files

### 🔹 `prometheus.yml`
Defines Prometheus scrape targets, collecting metrics from Jaeger.

### 🔹 `config.yml`
Configures Jaeger to store traces in memory and export metrics to Prometheus.

## 🛑 Stopping the Stack
```bash
docker-compose down
```

## 🛠 Troubleshooting
Check logs if a service is not running correctly:
```bash
docker-compose logs -f <service_name>
```

## ☕ Support This Project
If you found this project useful, consider supporting me:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Me-orange)](buymeacoffee.com/juanluis1702)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Me-blue?logo=kofi)](ko-fi.com/juanluis1702)


Your support helps keep this project maintained and improves future features! 🚀

---

### 📢 Contributions
Feel free to submit issues and pull requests!

### 📝 License
This project is licensed under the **MIT License**.

