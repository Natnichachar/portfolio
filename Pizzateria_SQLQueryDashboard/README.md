# 🍕 Pizzateria – SQL Query Database Project

Pizzateria is a **Streamlit-based web application** that simulates an online pizza ordering system backed by a **SQLite relational database**.  
The project focuses on **SQL querying, database design, and frontend–backend integration** using Python.

---

## 📌 Project Overview

This project demonstrates how SQL databases can be integrated into a real-world application scenario:

- User authentication (login & signup)
- Dynamic pizza ordering system
- Membership tiers with automatic upgrades
- Discount logic based on total spending
- Customer reviews & ratings
- Interactive UI with animations and image carousels

---

## 🛠 Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Database:** SQLite (`PizzaDatabase.db`)  
- **Data Handling:** Pandas  
- **UI Enhancements:**  
  - Lottie animations  
  - Image carousel (`st_ant_carousel`)  

---

## 📂 Project Structure
Pizzateria_SQLQueryDatabaseProject/
│
├── firstPage.py # Login & signup page
├── mainPage.py # Main dashboard (ordering, membership, reviews)
├── PizzaDatabase.db # SQLite database
│
├── images/ # UI images & tier badges
├── json/ # Lottie animation files
│
└── README.md


---

## 🧩 Database Design

The SQLite database contains multiple relational tables, including:

- **customers**
  - User authentication
  - Membership tier
  - Total spending
- **menu**
  - Item names & prices
- **orders**
  - Pizza flavor tracking
- **comments**
  - Ratings and customer reviews
- **tierPromotion**
  - Membership discounts and promotions

SQL queries are used extensively for:
- Authentication checks
- Dynamic price calculation
- Tier-based discount application
- Membership upgrades

---

## 🚀 Key Features

### 🔐 Authentication System
- Login & signup using stored customer credentials
- Username uniqueness validation

### 🍕 Order System
- Select pizza flavor, size, side dishes, and drinks
- Real-time price calculation
- Order summary before confirmation

### 💳 Membership Tier Logic
| Tier   | Requirement (Total Spending) | Benefit |
|------|------------------------------|---------|
| Bronze | Default                     | No discount |
| Silver | ≥ 1,000 THB                 | 5% discount |
| Gold   | ≥ 2,000 THB                 | 10% discount |

Membership tier updates automatically after each order.

### ⭐ Reviews & Ratings
- Displays top-rated customer reviews
- Uses SQL joins between `customers` and `comments`

---
