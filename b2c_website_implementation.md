# PC Lagbe? (B2C) Implementation Plan

## Overview
The consumer-facing portal where direct users come to get the perfect PC build recommendation based on real-time Bangladeshi market prices.

**CRITICAL DESIGN GOAL:** This is **NOT** a chat interface (like ChatGPT or Gemini). It is a highly structured, premium web application with specific forms, sliders, and interactive UI components. The AI happens completely behind the scenes.

## Core Features Needed

### 1. Consumer Authentication
- Email/Password and Google/Facebook Single Sign-On (SSO).
- Guest mode (allow 1 free generation without signup to hook the user).

### 2. Token/Session Based Monetization
- **Pricing Model:** "Sessions" instead of single builds. (e.g., 1 Session = 1 base build + 5 tweak regenerations).
- **Packages:**
  - 1 Session = 50 BDT
  - Pro Builder Pack (3 Sessions) = 100 BDT
  - Ultimate Pack (10 Sessions) = 250 BDT
- **Payment Gateway:** bKash, Nagad, or SSLCommerz direct integration for micro-transactions.

### 3. The Interactive PC Builder (Pre-Build UI)
Before generating, the user fills out a highly visual, structured form (not a text box):
- **Budget Slider:** Set exact minimum and maximum budget (e.g., 60,000৳ - 65,000৳).
- **Purpose Selection:** Gaming, Video Editing, Office, Programming, etc.
- **Brand Preferences (Toggles/Checkboxes):**
  - CPU: Intel Only, AMD Only, or "Best for Budget"
  - GPU: Nvidia, Radeon, Intel, or "Best for Budget"
- **Peripherals & Add-ons:**
  - "Include Monitor?" (Yes/No)
  - "Must have heavy RGB lighting?" (Yes/No)
  - Size preference (Standard ATX vs. Compact Mini-ITX)

### 4. The "Magic" Loading Screen
- Engaging animations showing the AI thinking, analyzing market data, checking thermal compatibilities, and matching prices (essential for perceived value while the API runs).

### 5. Results Page & Interactive Editing (Post-Build UI)
The generated build is presented as a sleek list or grid of components with live vendor links.
- **Visual Breakdown:** Final Score (0-10) with AI reasoning, plus total price.
- **Component Level Editing:** Instead of typing "change the GPU", the user clicks an "Edit" or "Swap" button next to a specific component (e.g., the RTX 4060).
  - The UI instantly opens a modal showing 3 alternative GPUs recommended by the AI that fit the remaining budget and compatibility matrix.
  - **Manual Database Search:** Inside the same modal, there is an option to "Search All Components". This opens a compact, fast-loading version of the entire component database, allowing users to manually search and select their personal choice (e.g., forcing a specific Asus ROG motherboard).
  - **Budget Warning:** If a user manually selects a component from the search that pushes the total price over their initial max budget, a non-intrusive warning notification appears (e.g., *"Warning: This selection puts you 5,000৳ over budget"*), but it still allows them to add it.
  - The user clicks their choice (AI recommended or manual), and the whole build recalculates.
- **Token Deduction:** Editing a component deducts 1 "tweak" from their session balance.

### 6. Gamification & Sharing
- "Share Build" generating a unique URL (`pclagbe.com/build/xyz123`).
- Export to PDF or beautiful image for social media (Facebook groups like PC Builders of Bangladesh).

---

## Session & State Management (This App's Responsibility)

> **Important:** The Zenfa AI engine is **stateless per-request**. All session tracking and state persistence is handled by this B2C app backend, NOT the engine.

### What This App Manages:
- **Session Tracking:** Store the current build, remaining tweak count, and user's preferences in this app's own Redis/database — keyed by a session ID.
- **Tweak Requests:** When a user edits a component, the app constructs a new `BuildRequest` (with the updated component catalog / locked components) and sends it as a fresh API call to the Zenfa engine's `/internal/build` endpoint.
- **Token/Credit Deduction:** The app tracks the user's purchased session balance and deducts on each generation or tweak. The engine is unaware of credits.
- **Build History:** Store past builds in the app's database for the user's profile page, sharing, and analytics.
