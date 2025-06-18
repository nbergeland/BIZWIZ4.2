# BIZWIZ_V4.2 Overview

## Purpose and Scope
BIZWIZ4.2 is a comprehensive location intelligence platform designed specifically for restaurant market analysis and site selection. The system provides two complementary approaches to location analysis: real-time static analysis with live API integration and on-demand dynamic analysis with synthetic data generation capabilities. This document covers the overall system architecture, core applications, and data management infrastructure.

## System Architecture Overview
BIZWIZ4.2 operates as a multi-tiered platform with distinct applications serving different analysis needs. The system architecture consists of two primary analysis pathways supported by a shared configuration management layer.

### Core Application Architecture
<img width="1445" alt="Screenshot 2025-06-17 at 9 05 20 PM" src="https://github.com/user-attachments/assets/4068c7eb-c663-4946-9c0d-0a81117817f3" />

## Data Flow Architecture
The system processes location intelligence through two distinct data flow patterns, each optimized for different use cases and data availability scenarios.
<img width="1377" alt="Screenshot 2025-06-17 at 9 10 39 PM" src="https://github.com/user-attachments/assets/4349ef3b-c6b5-4b21-8e2a-a59ab80bf645" />

## Core Applications
### Static Analysis Dashboard (dash42.py)
The static analysis system provides immediate, real-time location intelligence through direct API integration. This application serves as the primary user interface for live market analysis.
