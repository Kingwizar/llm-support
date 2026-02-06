# LLM Support Platform – Omnichannel AI Assistant

## Overview

This project implements an **omnichannel conversational AI platform** designed for customer support, internal assistance, and immersive user interaction.  
It combines **Large Language Models (LLM)**, a **Retrieval-Augmented Generation (RAG)** system, web and mobile interfaces, and a **3D conversational agent** powered by Unreal Engine.

The architecture is modular, scalable, and designed to run in controlled environments (datacenter / on-premise), with a strong focus on **security**, **performance**, and **maintainability**.

---

## Global Architecture

The system is composed of several interconnected modules:

- **Backend (FastAPI – Python)**  
  Central orchestration layer and single source of truth
- **LLM Engine (Ollama – Qwen 2.5 14B)**  
  Text generation and reasoning
- **RAG System**  
  Contextual enrichment from indexed documentation
- **Node.js Gateway**  
  Secure intermediary between backend and clients
- **Web Interface (Angular)**  
  Main user interface
- **Mobile Application (Android – Kotlin)**  
  Mobile access to the same features
- **3D Agent (Unreal Engine + MetaHuman)**  
  Immersive vocal interaction
- **Monitoring Stack**  
  Prometheus, Grafana, Loki, DCGM Exporter

---

## Backend – Core of the System

The backend is the **functional core** of the project. It centralizes all business logic and orchestrates interactions between:

- the LLM,
- the RAG system,
- the database,
- the web and mobile clients,
- and the 3D agent.

It exposes a set of **FastAPI routes** handling:
- user queries,
- security and authentication,
- session management,
- document ingestion,
- web search enrichment,
- STT / TTS pipelines.

### Main Files and Directories

#### `main.py`
Main entry point of the backend.  
Acts as the **orchestrator**, exposing all FastAPI routes used by the client applications.

Responsibilities:
- receive user requests and parameters (response depth, RAG usage, web search),
- build structured prompts,
- call the RAG system,
- forward enriched prompts to the LLM,
- return responses to clients.

---

#### `llm/rag_core.py`
Core module of the **Retrieval-Augmented Generation system**.

Responsibilities:
- document indexing,
- contextual search,
- selection of relevant content,
- injection of contextual data into prompts.

It relies on additional helper modules for document loading, preprocessing, and updates.

---

#### LLM Engine – Ollama
The selected model (**Qwen 2.5 – 14B**) runs locally through **Ollama**.

Workflow:
1. Backend builds a structured prompt
2. Prompt is enriched via RAG
3. Prompt is sent to Ollama
4. Generated response is returned through FastAPI routes

---

#### `llm/ai_chat/`
Dedicated folder for **voice interaction**.

Includes:
- **Speech-to-Text (STT)**
- **Text-to-Speech (TTS)**

These components enable real-time voice interaction, especially for the **Unreal Engine MetaHuman agent**, with support for multiple configurable voices using libraries such as **Whisper**.

---

## Node.js Gateway

The Node.js server acts as a **secure intermediary layer** between FastAPI and the client applications.

Responsibilities:
- route management,
- request filtering,
- security enforcement.

Security mechanisms:
- **CSRF protection** (for the Angular web app),
- **token-based authentication**.

For the Android application, CSRF is not applicable; instead, a dedicated token strategy and access rules are used to ensure equivalent security.

---

## Web Interface – Angular

The Angular application is the **main interaction interface**.

Design principles:
- clean and minimal UI,
- consistent with the company’s visual identity,
- user-centric interaction flow.

### Structure

- **Services**
  - handle data exchange with Node.js,
  - manage authentication and security,
  - abstract backend communication.

- **Components**
  - login page,
  - conversation history,
  - chat panel,
  - input bar with options:
    - web search,
    - document upload as contextual sources for the LLM.

---

## Mobile Application – Android (Kotlin)

The Android application provides the **same core features** as the web interface:

- conversational interaction with the LLM,
- file upload,
- web search activation.

The UI and interactions are adapted to mobile usage while maintaining functional parity.

---

## 3D Conversational Agent – Unreal Engine 5

The project includes an experimental **3D conversational agent** for immersive interaction.

### Setup Requirements

1. **Create a MetaHuman**  
   Using Epic Games MetaHuman Creator.

2. **Install NVIDIA Audio2Face Plugin**  
   Used for real-time lip-sync and facial animation driven by audio.

### Reference Tutorials

- Audio2Face + MetaHuman integration (Part 1):  
  https://youtu.be/RCGkqx1qBX0?list=LL

- Audio2Face + Unreal workflow (Part 2):  
  https://youtu.be/-l8kjQUTfQk

### Interaction Pipeline

1. User speaks → **STT**
2. Text processed by **LLM**
3. Response converted to audio → **TTS**
4. Audio streamed to Unreal Engine
5. MetaHuman animates lips and facial expressions in real time

---

## Monitoring and Observability

The platform integrates a full monitoring stack:

- **Prometheus** – metrics collection
- **Grafana** – visualization dashboards
- **Loki** – log aggregation
- **DCGM Exporter** – GPU metrics (usage, memory, temperature)

This setup enables performance analysis, debugging, and resource optimization.

---

## System Startup Procedure

### 1. System Update and GPU Driver Installation

```bash
sudo dnf update -y
sudo dnf install -y oraclelinux-release-el9
sudo dnf config-manager --enable ol9_addons
sudo dnf install -y nvidia-driver nvidia-driver-cuda
sudo reboot
