# MatoshriGPT: AI-Powered Campus Assistant

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Scope](#scope)
- [Methodology](#methodology)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Applications](#applications)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction  
**MatoshriGPT** is an AI-powered chatbot designed to assist students, faculty, and staff by providing instant and accurate information about campus-related queries. The system leverages Natural Language Processing (NLP) and fuzzy matching techniques to ensure efficient interaction.

---

## Problem Statement  
Students and faculty often face challenges in accessing campus-related information such as course details, faculty contacts, event schedules, and administrative procedures. Current communication methods are time-consuming and inefficient. **MatoshriGPT** aims to resolve this by offering instant, accurate responses to frequently asked questions.

---

## Objectives  
- Develop an intelligent chatbot for campus-related queries.  
- Implement a user-friendly web interface.  
- Improve response accuracy using fuzzy matching techniques.  
- Ensure 24/7 availability of campus information.

---

## Scope  
**MatoshriGPT** will serve students, faculty, and administrative staff by providing academic details, event schedules, and general campus information.

---

## Methodology  
1. **Data Collection:** Extracting data from the college website and other sources.  
2. **Data Preprocessing:** Cleaning and structuring the data for better efficiency.  
3. **NLP Techniques:** Using Fuzzy Matching to enhance query recognition.  
4. **Chatbot Development:** Building the system using Flask, HTML, and CSS.  
5. **Deployment:** Hosting on a local or cloud server.

---

## System Architecture  
The chatbot system consists of the following components:

- **Frontend:** Built using HTML, CSS, and JavaScript for user interaction.  
- **Backend:** Flask (Python) to process user queries.  
- **Database:** JSON-based repository for chatbot responses.  
- **AI Logic:** Fuzzy string matching to handle different variations of user queries.

---

## Features  
- Instant responses to campus-related questions.  
- AI-based query understanding for better accuracy.  
- Responsive web-based UI.  
- Easy integration with existing systems.  
- Continuous learning through dataset updates.

---

## Installation  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- Flask  
- FuzzyWuzzy library  

### Setup Instructions  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/matoshrigpt.git
   cd matoshrigpt
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the chatbot server:  
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000`

---

## Usage  
1. Open the web application.  
2. Enter your query in the input field.  
3. Click **"Send"** to get an automated response.  
4. Example queries:  
   - "What are the library hours?"  
   - "Who is the head of the computer department?"  

---

## Results  
The chatbot efficiently responds to campus-related queries, improving accessibility and user experience by reducing response time.

---

## Applications  
- Student and faculty assistance.  
- Admission-related queries.  
- Event scheduling information.  
- Campus facility details.

---

## Conclusion  
**MatoshriGPT** provides an AI-driven solution to bridge communication gaps by offering instant responses to campus-related queries, improving accessibility, and ensuring a better user experience.

---

## References  
- [Matoshri College Official Website](https://www.matoshricollege.edu)  
- Flask Documentation  
- FuzzyWuzzy Documentation  
