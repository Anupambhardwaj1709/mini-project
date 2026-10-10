# Smart Library Management System with AI

Flask + React + SQLite web app with AI book recommendations, natural-language search and demand prediction.

## What is inside
| File | Purpose |
|---|---|
| `app.py` | Flask REST API: login (JWT), books, students, borrow / return with Rs 5 per late day fine, analytics |
| `ml.py` | TF-IDF + cosine similarity: recommendations, semantic search, similar books, 30-day demand forecast (pure Python) |
| `static/index.html` | React frontend (student and librarian screens). It needs no Node or npm build |
| `sample_books.csv`, `sample_students.csv` | Templates for importing your real data |

## Run it (Windows)
```
cd smart-library-system
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000. The first start creates the librarian login `admin@library.com` / `admin123`.
Change it from the first login, or set `ADMIN_EMAIL` and `ADMIN_PASSWORD` before the first run.
The browser needs internet once per session to load React, Tailwind and fonts from CDNs.

## Use your real data
1. Log in as the librarian, open **Books**, and import a CSV (`title, author, category, description, isbn, copies`).
   A good description makes recommendations and search much better.
2. Open **Students** and import a CSV (`name, email, roll_no, password`). A blank password becomes the roll number.
   Students can also register themselves.

## API endpoints
`POST /api/auth/login`, `POST /api/auth/register`, `GET/POST /api/books`, `POST /api/books/import`,
`POST /api/borrow`, `POST /api/return`, `GET /api/my-loans`, `GET /api/recommendations`,
`GET /api/search?q=`, `GET /api/analytics`, `GET /api/demand`

## Rules
Loan period 14 days, maximum 3 books per student, fine Rs 5 per late day. Change `LOAN_DAYS`, `MAX_ACTIVE_LOANS` and `FINE_PER_DAY` at the top of `app.py`.

## Moving to MySQL later
Set `DATABASE_URL` (for example `mysql+pymysql://user:pass@localhost/library`) and install the driver. The code uses SQLAlchemy, so nothing else changes.
