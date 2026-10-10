"""Smart Library Management System with AI - Flask backend.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import base64
import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, g, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

import ml

BASE = os.path.dirname(os.path.abspath(__file__))
FINE_PER_DAY = 5      # rupees per late day
LOAN_DAYS = 14
MAX_ACTIVE_LOANS = 3

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///" + os.path.join(BASE, "library.db"))
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


def now():
    return datetime.now()


def load_secret():
    env = os.environ.get("SECRET_KEY")
    if env:
        return env.encode()
    path = os.path.join(BASE, ".secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip().encode()


SECRET = load_secret()


# ----------------------------------------------------------------- models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False)
    roll_no = db.Column(db.String(40), unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), default="student", nullable=False)
    created_on = db.Column(db.DateTime, default=now)


class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(250), nullable=False)
    author = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(80), default="General")
    description = db.Column(db.Text, default="")
    isbn = db.Column(db.String(30), default="")
    total_copies = db.Column(db.Integer, default=1)
    available_copies = db.Column(db.Integer, default=1)
    added_on = db.Column(db.DateTime, default=now)


class Loan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey("book.id"), nullable=False)
    issued_on = db.Column(db.DateTime, default=now, nullable=False)
    due_on = db.Column(db.DateTime, nullable=False)
    returned_on = db.Column(db.DateTime)
    fine = db.Column(db.Integer, default=0)
    user = db.relationship("User")
    book = db.relationship("Book")


class View(db.Model):
    """A student opening a book's details page. Used as a weak signal for recommendations."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey("book.id"), nullable=False)
    viewed_on = db.Column(db.DateTime, default=now)


# ----------------------------------------------------------------- auth (JWT, HS256, standard library only)
def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(message):
    return _b64(hmac.new(SECRET, message.encode(), hashlib.sha256).digest())


def make_token(user):
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps({"sub": user.id, "role": user.role, "exp": int(time.time()) + 12 * 3600}).encode())
    return f"{head}.{body}.{_sign(head + '.' + body)}"


def read_token(token):
    try:
        head, body, sig = token.split(".")
        if not hmac.compare_digest(sig, _sign(head + "." + body)):
            return None
        data = json.loads(_unb64(body))
        return data if data["exp"] >= time.time() else None
    except Exception:
        return None


def auth(role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            header = request.headers.get("Authorization", "")
            data = read_token(header[7:]) if header.startswith("Bearer ") else None
            user = db.session.get(User, data["sub"]) if data else None
            if not user:
                return err("Please log in again.", 401)
            if role and user.role != role:
                return err("Only the librarian can do this.", 403)
            g.user = user
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ----------------------------------------------------------------- helpers
def err(message, code=400):
    return jsonify(error=message), code


def body():
    return request.get_json(silent=True) or {}


def text(value):
    return str(value or "").strip()


def to_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def user_dict(u):
    return dict(id=u.id, name=u.name, email=u.email, roll_no=u.roll_no, role=u.role)


def book_dict(b):
    return dict(id=b.id, title=b.title, author=b.author, category=b.category, description=b.description,
                isbn=b.isbn, total=b.total_copies, available=b.available_copies)


def loan_dict(loan):
    end = loan.returned_on or now()
    late = max(0, (end.date() - loan.due_on.date()).days)
    returned = loan.returned_on is not None
    return dict(
        id=loan.id, book_id=loan.book_id, title=loan.book.title, author=loan.book.author,
        user_id=loan.user_id, student=loan.user.name, roll_no=loan.user.roll_no,
        issued_on=loan.issued_on.date().isoformat(), due_on=loan.due_on.date().isoformat(),
        returned_on=loan.returned_on.date().isoformat() if returned else None,
        days_left=None if returned else (loan.due_on.date() - now().date()).days,
        overdue=(not returned) and late > 0, days_late=late,
        fine=loan.fine if returned else late * FINE_PER_DAY,
    )


def ml_books():
    return [dict(id=b.id, title=b.title, author=b.author, category=b.category, description=b.description)
            for b in Book.query.all()]


def read_csv(raw):
    reader = csv.DictReader(io.StringIO(raw.lstrip("\ufeff")))
    for row in reader:
        yield {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}


def add_student(name, email, roll_no, password):
    email = email.lower()
    if not name or "@" not in email or not roll_no:
        return "Name, a valid email and roll number are required."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if User.query.filter((User.email == email) | (User.roll_no == roll_no)).first():
        return "A student with this email or roll number already exists."
    db.session.add(User(name=name, email=email, roll_no=roll_no,
                        password_hash=generate_password_hash(password), role="student"))
    return None


# ----------------------------------------------------------------- pages
@app.get("/")
def index():
    return app.send_static_file("index.html")


# ----------------------------------------------------------------- auth routes
@app.post("/api/auth/register")
def register():
    d = body()
    problem = add_student(text(d.get("name")), text(d.get("email")), text(d.get("roll_no")), text(d.get("password")))
    if problem:
        return err(problem)
    db.session.commit()
    user = User.query.filter_by(email=text(d.get("email")).lower()).first()
    return jsonify(token=make_token(user), user=user_dict(user)), 201


@app.post("/api/auth/login")
def login():
    d = body()
    who = text(d.get("email"))
    user = User.query.filter((User.email == who.lower()) | (User.roll_no == who)).first()
    if not user or not check_password_hash(user.password_hash, text(d.get("password"))):
        return err("Email, roll number or password is wrong.", 401)
    return jsonify(token=make_token(user), user=user_dict(user))


@app.get("/api/auth/me")
@auth()
def me():
    return jsonify(user=user_dict(g.user))


@app.post("/api/auth/password")
@auth()
def change_password():
    d = body()
    if not check_password_hash(g.user.password_hash, text(d.get("old"))):
        return err("Current password is wrong.")
    if len(text(d.get("new"))) < 6:
        return err("New password must be at least 6 characters.")
    g.user.password_hash = generate_password_hash(text(d.get("new")))
    db.session.commit()
    return jsonify(ok=True)


# ----------------------------------------------------------------- books
@app.get("/api/books")
@auth()
def list_books():
    q = text(request.args.get("q")).lower()
    category = text(request.args.get("category"))
    books = Book.query.order_by(Book.title).all()
    out = []
    for b in books:
        if category and b.category != category:
            continue
        if q and q not in f"{b.title} {b.author} {b.category} {b.isbn}".lower():
            continue
        out.append(book_dict(b))
    return jsonify(books=out)


@app.get("/api/categories")
@auth()
def categories():
    rows = sorted({b.category for b in Book.query.all() if b.category})
    return jsonify(categories=rows)


@app.get("/api/books/<int:book_id>")
@auth()
def book_detail(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return err("Book not found.", 404)
    if g.user.role == "student":
        db.session.add(View(user_id=g.user.id, book_id=book.id))
        db.session.commit()
    similar = []
    for other_id, score in ml.Index(ml_books()).similar(book.id):
        other = db.session.get(Book, other_id)
        similar.append(dict(book_dict(other), match=round(score * 100)))
    return jsonify(book=book_dict(book), similar=similar)


def apply_book_fields(book, d):
    title, author = text(d.get("title")), text(d.get("author"))
    if not title or not author:
        return "Title and author are required."
    copies = max(1, to_int(d.get("copies", d.get("total")), 1))
    taken = book.total_copies - book.available_copies if book.id else 0
    if copies < taken:
        return f"{taken} copies are currently issued, so the total cannot be lower than that."
    book.title, book.author = title, author
    book.category = text(d.get("category")) or "General"
    book.description = text(d.get("description"))
    book.isbn = text(d.get("isbn"))
    book.total_copies = copies
    book.available_copies = copies - taken
    return None


@app.post("/api/books")
@auth("admin")
def add_book():
    book = Book()
    problem = apply_book_fields(book, body())
    if problem:
        return err(problem)
    db.session.add(book)
    db.session.commit()
    return jsonify(book=book_dict(book)), 201


@app.put("/api/books/<int:book_id>")
@auth("admin")
def edit_book(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return err("Book not found.", 404)
    problem = apply_book_fields(book, body())
    if problem:
        return err(problem)
    db.session.commit()
    return jsonify(book=book_dict(book))


@app.delete("/api/books/<int:book_id>")
@auth("admin")
def delete_book(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return err("Book not found.", 404)
    if Loan.query.filter_by(book_id=book.id, returned_on=None).first():
        return err("This book is currently issued. Return it before deleting.")
    View.query.filter_by(book_id=book.id).delete()
    Loan.query.filter_by(book_id=book.id).delete()
    db.session.delete(book)
    db.session.commit()
    return jsonify(ok=True)


@app.post("/api/books/import")
@auth("admin")
def import_books():
    """CSV columns: title, author, category, description, isbn, copies."""
    added = skipped = 0
    existing = {(b.title.lower(), b.author.lower()) for b in Book.query.all()}
    for row in read_csv(text(body().get("csv"))):
        key = (row.get("title", "").lower(), row.get("author", "").lower())
        book = Book()
        if key in existing or apply_book_fields(book, row):
            skipped += 1
            continue
        db.session.add(book)
        existing.add(key)
        added += 1
    db.session.commit()
    return jsonify(added=added, skipped=skipped)


# ----------------------------------------------------------------- students
@app.get("/api/students")
@auth("admin")
def list_students():
    active = Counter(l.user_id for l in Loan.query.filter_by(returned_on=None).all())
    rows = [dict(user_dict(u), active_loans=active.get(u.id, 0))
            for u in User.query.filter_by(role="student").order_by(User.name).all()]
    return jsonify(students=rows)


@app.post("/api/students")
@auth("admin")
def create_student():
    d = body()
    problem = add_student(text(d.get("name")), text(d.get("email")), text(d.get("roll_no")),
                          text(d.get("password")) or text(d.get("roll_no")))
    if problem:
        return err(problem)
    db.session.commit()
    return jsonify(ok=True), 201


@app.post("/api/students/import")
@auth("admin")
def import_students():
    """CSV columns: name, email, roll_no, password (password defaults to the roll number)."""
    added = skipped = 0
    for row in read_csv(text(body().get("csv"))):
        problem = add_student(row.get("name", ""), row.get("email", ""), row.get("roll_no", ""),
                              row.get("password") or row.get("roll_no", ""))
        if problem:
            skipped += 1
        else:
            db.session.commit()
            added += 1
    return jsonify(added=added, skipped=skipped)


# ----------------------------------------------------------------- borrowing
@app.post("/api/borrow")
@auth()
def borrow():
    d = body()
    book = db.session.get(Book, to_int(d.get("book_id"), 0))
    if not book:
        return err("Book not found.", 404)
    student = g.user
    if g.user.role == "admin":
        who = text(d.get("roll_no"))
        student = User.query.filter(((User.roll_no == who) | (User.email == who.lower())) & (User.role == "student")).first() if who else None
        if not student:
            return err("No student found with that roll number.")
    active = Loan.query.filter_by(user_id=student.id, returned_on=None)
    if active.count() >= MAX_ACTIVE_LOANS:
        return err(f"{student.name} already has {MAX_ACTIVE_LOANS} books issued. Return one first.")
    if active.filter_by(book_id=book.id).first():
        return err("This book is already issued to this student.")
    if book.available_copies < 1:
        return err("All copies are out right now.")
    issued = now()
    loan = Loan(user_id=student.id, book_id=book.id, issued_on=issued, due_on=issued + timedelta(days=LOAN_DAYS))
    book.available_copies -= 1
    db.session.add(loan)
    db.session.commit()
    return jsonify(loan=loan_dict(loan)), 201


@app.post("/api/return")
@auth()
def return_book():
    loan = db.session.get(Loan, to_int(body().get("loan_id"), 0))
    if not loan or loan.returned_on:
        return err("This loan was not found or is already returned.", 404)
    if g.user.role != "admin" and loan.user_id != g.user.id:
        return err("This is not your loan.", 403)
    end = now()
    late = max(0, (end.date() - loan.due_on.date()).days)
    loan.fine = late * FINE_PER_DAY
    loan.returned_on = end
    loan.book.available_copies += 1
    db.session.commit()
    return jsonify(loan=loan_dict(loan))


@app.get("/api/my-loans")
@auth()
def my_loans():
    loans = Loan.query.filter_by(user_id=g.user.id).order_by(Loan.issued_on.desc()).all()
    return jsonify(loans=[loan_dict(l) for l in loans])


@app.get("/api/loans")
@auth("admin")
def all_loans():
    query = Loan.query.order_by(Loan.issued_on.desc())
    if request.args.get("active"):
        query = query.filter_by(returned_on=None)
    return jsonify(loans=[loan_dict(l) for l in query.limit(300).all()])


# ----------------------------------------------------------------- AI endpoints
@app.get("/api/recommendations")
@auth()
def recommendations():
    profile, borrowed = Counter(), set()
    for loan in Loan.query.filter_by(user_id=g.user.id).all():
        profile[loan.book_id] += 3
        borrowed.add(loan.book_id)
    views = Counter(v.book_id for v in View.query.filter_by(user_id=g.user.id).all())
    for book_id, n in views.items():
        profile[book_id] += min(n, 3)
    index = ml.Index(ml_books())
    picks = index.recommend(dict(profile), exclude=borrowed)
    out = []
    for book_id, score, source in picks:
        why = "Because you borrowed" if source in borrowed else "Because you looked at"
        out.append(dict(book_dict(db.session.get(Book, book_id)), match=round(score * 100),
                        reason=f"{why} {index.books[source]['title']}"))
    if not out:  # new student: fall back to what other students borrow most
        popular = Counter(l.book_id for l in Loan.query.all())
        ranked = [b for b, _ in popular.most_common(8) if b not in borrowed]
        if not ranked:
            ranked = [b.id for b in Book.query.order_by(Book.added_on.desc()).limit(8).all()]
        for book_id in ranked:
            out.append(dict(book_dict(db.session.get(Book, book_id)), match=None,
                            reason="Popular with other students" if popular else "Recently added"))
    return jsonify(books=out, personalised=bool(picks))


@app.get("/api/search")
@auth()
def semantic_search():
    q = text(request.args.get("q"))
    if not q:
        return jsonify(books=[])
    out = []
    for book_id, score in ml.Index(ml_books()).search(q):
        out.append(dict(book_dict(db.session.get(Book, book_id)), match=min(100, round(score * 100))))
    return jsonify(books=out)


@app.get("/api/demand")
@auth("admin")
def demand():
    books = [dict(id=b.id, title=b.title, category=b.category, total=b.total_copies) for b in Book.query.all()]
    loans = [(l.book_id, l.issued_on) for l in Loan.query.all()]
    return jsonify(forecast=ml.predict_demand(books, loans, now(), LOAN_DAYS))


@app.get("/api/analytics")
@auth("admin")
def analytics():
    books = Book.query.all()
    loans = Loan.query.all()
    active = [l for l in loans if not l.returned_on]
    overdue = [l for l in active if loan_dict(l)["overdue"]]
    by_book = Counter(l.book_id for l in loans)
    titles = {b.id: b.title for b in books}
    cats = {b.id: b.category for b in books}
    by_category = Counter(cats.get(l.book_id, "General") for l in loans)
    today = now().date()
    weekly = []
    for i in range(7, -1, -1):
        start = today - timedelta(days=7 * i + 6)
        end = today - timedelta(days=7 * i)
        count = sum(1 for l in loans if start <= l.issued_on.date() <= end)
        weekly.append(dict(label=start.strftime("%d %b"), count=count))
    return jsonify(
        titles=len(books), copies=sum(b.total_copies for b in books),
        students=User.query.filter_by(role="student").count(),
        active_loans=len(active), overdue=len(overdue),
        fines_collected=sum(l.fine or 0 for l in loans if l.returned_on),
        fines_pending=sum(loan_dict(l)["fine"] for l in overdue),
        top_books=[dict(title=titles[b], loans=n) for b, n in by_book.most_common(5) if b in titles],
        by_category=[dict(category=c, loans=n) for c, n in by_category.most_common(6)],
        weekly=weekly,
    )


# ----------------------------------------------------------------- start up
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role="admin").first():
            email = os.environ.get("ADMIN_EMAIL", "admin@library.com")
            password = os.environ.get("ADMIN_PASSWORD", "admin123")
            db.session.add(User(name="Librarian", email=email, role="admin",
                                password_hash=generate_password_hash(password)))
            db.session.commit()
            print(f"Created librarian login: {email} / {password}  (change it after first login)")


init_db()

if __name__ == "__main__":
    print("Smart Library is running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
