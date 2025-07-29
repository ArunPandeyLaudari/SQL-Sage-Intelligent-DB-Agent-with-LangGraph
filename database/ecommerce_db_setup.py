import sqlite3

# === Connect to SQLite Database ===
conn = sqlite3.connect('final_ecommerce.db')
cursor = conn.cursor()

# === Drop old tables if they exist ===
tables = ["order_coupons","coupons","cart_items","cart","wishlist","reviews","shipping",
          "payments","order_items","orders","product_images","products","categories","users"]
for t in tables:
    cursor.execute(f"DROP TABLE IF EXISTS {t}")

# === Create Tables ===
cursor.execute("""CREATE TABLE users(
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, email TEXT UNIQUE, password TEXT,
    phone TEXT, address TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

cursor.execute("""CREATE TABLE categories(
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL, description TEXT)""")

cursor.execute("""CREATE TABLE products(
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_id INTEGER, name TEXT NOT NULL, description TEXT,
    price REAL NOT NULL, stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(category_id) REFERENCES categories(category_id))""")

cursor.execute("""CREATE TABLE product_images(
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER, image_url TEXT NOT NULL,
    FOREIGN KEY(product_id) REFERENCES products(product_id))""")

cursor.execute("""CREATE TABLE orders(
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'Pending', total_amount REAL,
    FOREIGN KEY(user_id) REFERENCES users(user_id))""")

cursor.execute("""CREATE TABLE order_items(
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER, product_id INTEGER, quantity INTEGER NOT NULL, price REAL NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id))""")

cursor.execute("""CREATE TABLE payments(
    payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER, payment_method TEXT, amount REAL,
    payment_status TEXT DEFAULT 'Pending', paid_at TIMESTAMP NULL,
    FOREIGN KEY(order_id) REFERENCES orders(order_id))""")

cursor.execute("""CREATE TABLE shipping(
    shipping_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER, tracking_number TEXT, shipping_address TEXT,
    shipping_date TIMESTAMP NULL, delivery_date TIMESTAMP NULL,
    status TEXT DEFAULT 'Preparing',
    FOREIGN KEY(order_id) REFERENCES orders(order_id))""")

cursor.execute("""CREATE TABLE reviews(
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, product_id INTEGER, rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id))""")

cursor.execute("""CREATE TABLE wishlist(
    wishlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, product_id INTEGER, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id))""")

cursor.execute("""CREATE TABLE cart(
    cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id))""")

cursor.execute("""CREATE TABLE cart_items(
    cart_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cart_id INTEGER, product_id INTEGER, quantity INTEGER NOT NULL,
    FOREIGN KEY(cart_id) REFERENCES cart(cart_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id))""")

cursor.execute("""CREATE TABLE coupons(
    coupon_id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE, discount_percent INTEGER CHECK(discount_percent BETWEEN 1 AND 100),
    valid_from DATE, valid_to DATE)""")

cursor.execute("""CREATE TABLE order_coupons(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER, coupon_id INTEGER,
    FOREIGN KEY(order_id) REFERENCES orders(order_id),
    FOREIGN KEY(coupon_id) REFERENCES coupons(coupon_id))""")

# === Insert Users (Nepali Names in English) ===
users = [
    ("Arun Pandey","arun@example.com","pass","9801111111","Kathmandu"),
    ("Sita Sharma","sita@example.com","pass","9802222222","Pokhara"),
    ("Ram Thapa","ram@example.com","pass","9803333333","Butwal"),
    ("Krishna Adhikari","krishna@example.com","pass","9804444444","Chitwan"),
    ("Bina Kafle","bina@example.com","pass","9805555555","Lalitpur"),
    ("Binisha Chapagain","binisha@example.com","pass","9806666666","Bhaktapur")
]
cursor.executemany("INSERT INTO users(name,email,password,phone,address) VALUES(?,?,?,?,?)",users)

# === Categories ===
categories = [
    ("Mobiles", "Smartphones and accessories"),
    ("Clothing", "Men and Women fashion"),
    ("Computers", "Laptops and peripherals"),
    ("Books", "Educational and novels"),
    ("Toys", "Kids toys and games")
]
cursor.executemany("INSERT INTO categories(category_name,description) VALUES(?,?)",categories)

# === Products ===
products = [
    (1,"Samsung Galaxy A55","5G smartphone with AMOLED display",550.00,50),
    (1,"iPhone 15","Latest Apple flagship smartphone",999.00,30),
    (2,"Winter Coat","Stylish warm coat for winter",120.00,25),
    (3,"Dell XPS","High-end laptop with great performance",1300.00,10),
    (4,"Nepali Folk Tales","Collection of classic Nepali stories",12.00,100),
    (5,"RC Car","Remote controlled racing car",45.00,40)
]
cursor.executemany("INSERT INTO products(category_id,name,description,price,stock) VALUES(?,?,?,?,?)",products)

# === Product Images ===
images = [(1,"samsung.jpg"),(2,"iphone.jpg"),(3,"coat.jpg"),(4,"dell.jpg"),(5,"folktales.jpg"),(6,"rc_car.jpg")]
cursor.executemany("INSERT INTO product_images(product_id,image_url) VALUES(?,?)",images)

# === Orders ===
orders = [(1,550.00),(2,120.00),(3,1300.00),(4,12.00),(5,45.00),(6,300.00)]
cursor.executemany("INSERT INTO orders(user_id,total_amount) VALUES(?,?)",orders)

# === Order Items ===
items = [(1,1,1,550.00),(2,3,1,120.00),(3,4,1,1300.00),(4,5,1,12.00),(5,6,1,45.00),(6,2,1,300.00)]
cursor.executemany("INSERT INTO order_items(order_id,product_id,quantity,price) VALUES(?,?,?,?)",items)

# === Payments ===
payments = [
    (1,"Esewa",550.00,"Completed"),
    (2,"Khalti",120.00,"Completed"),
    (3,"Cash",1300.00,"Pending"),
    (4,"Card",12.00,"Completed"),
    (5,"Esewa",45.00,"Completed"),
    (6,"Credit Card",300.00,"Completed")
]
cursor.executemany("INSERT INTO payments(order_id,payment_method,amount,payment_status) VALUES(?,?,?,?)",payments)

# === Shipping ===
shipping = [
    (1,"TRK001","Kathmandu","2025-07-01","2025-07-05","Delivered"),
    (2,"TRK002","Pokhara","2025-07-02","2025-07-06","Delivered"),
    (3,"TRK003","Butwal","2025-07-03","2025-07-08","Shipped"),
    (4,"TRK004","Chitwan","2025-07-04","2025-07-09","Preparing"),
    (5,"TRK005","Lalitpur","2025-07-05","2025-07-10","Delivered"),
    (6,"TRK006","Bhaktapur","2025-07-06","2025-07-11","Delivered")
]
cursor.executemany("INSERT INTO shipping(order_id,tracking_number,shipping_address,shipping_date,delivery_date,status) VALUES(?,?,?,?,?,?)",shipping)

# === Reviews ===
reviews = [
    (1,1,5,"Excellent smartphone!"),
    (2,3,4,"Coat is warm and stylish."),
    (3,4,5,"Laptop is powerful."),
    (4,5,3,"Nice story collection."),
    (5,6,4,"RC Car is fun to play."),
    (6,2,5,"iPhone 15 is amazing!")
]
cursor.executemany("INSERT INTO reviews(user_id,product_id,rating,comment) VALUES(?,?,?,?)",reviews)

# === Wishlist ===
wishlist = [(1,2),(2,3),(3,1),(4,5),(5,4),(6,3)]
cursor.executemany("INSERT INTO wishlist(user_id,product_id) VALUES(?,?)",wishlist)

# === Cart ===
cart = [(1,),(2,),(3,),(4,),(5,),(6,)]
cursor.executemany("INSERT INTO cart(user_id) VALUES(?)",cart)

# === Cart Items ===
cart_items = [(1,3,1),(2,4,2),(3,2,1),(4,5,1),(5,1,1),(6,6,1)]
cursor.executemany("INSERT INTO cart_items(cart_id,product_id,quantity) VALUES(?,?,?)",cart_items)

# === Coupons ===
coupons = [
    ("NEPAL10",10,"2025-07-01","2025-12-31"),
    ("FEST20",20,"2025-07-01","2025-12-31"),
    ("WELCOME5",5,"2025-07-01","2025-12-31"),
    ("SALE15",15,"2025-07-01","2025-12-31"),
    ("NEW30",30,"2025-07-01","2025-12-31")
]
cursor.executemany("INSERT INTO coupons(code,discount_percent,valid_from,valid_to) VALUES(?,?,?,?)",coupons)

# === Order Coupons ===
order_coupons = [(1,1),(2,2),(3,3),(4,4),(5,5),(6,1)]
cursor.executemany("INSERT INTO order_coupons(order_id,coupon_id) VALUES(?,?)",order_coupons)

# === Commit & Close ===
conn.commit()
conn.close()

print("✅ Final E-commerce Database created successfully with 15 tables, Nepali users, and realistic data!")
