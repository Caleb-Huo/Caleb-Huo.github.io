#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tuesday Nov 21st, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "SQL via python"
#' ---
#' 
## ----setup, include=FALSE---------------------------------------
library(reticulate)
#use_python("/usr/local/bin/python3.10")
use_python("/Users/zhuo/anaconda3/envs/py311/bin/python")

#' 
#' 
#' Background
#' ===
#' 
#' - What is SQL?
#'   -  It's a programming language used for managing and manipulating relational databases.
#' 
#' - Advantage:  
#'   - Flexibility and Simplicity: The tabular structure makes it easy to understand and navigate.
#'   - Data Accuracy: Constraints and relations ensure accuracy and consistency of data.
#'   - Scalability: They can handle large amounts of data and are scalable.
#'   - Security: Features robust security measures, including access controls and permissions.
#'   - Data Relationships: Easily handles data relationships, making it ideal for complex queries and data analysis.
#'   
#' - Why we learn SQL?
#'   - Ubiquitous demand: SQL skills are in high demand across a wide range of industries, including technology, finance, healthcare, retail, and more.
#'   - Diverse Roles: Proficiency in SQL is a key requirement for many job roles such as database administrators, data analysts, software engineers, business analysts, and more.
#' 
#' 
#' Type of SQL database
#' ===
#' 
#' - SQL (Relational) Databases:
#'   - MySQL: One of the most popular open-source relational database management systems.
#'   - PostgreSQL
#'   - Microsoft SQL Server
#'   - Oracle Database
#'   - SQLite
#'   - others
#'   
#' - NoSQL (Non-Relational) Databases:
#'   - MongoDB: A leading NoSQL database that stores data in flexible, JSON-like documents.
#'   - others
#'   
#' - key difference:
#'   - SQL: structured table based data model
#'   - noSQL: flexible data models
#' 
#' Get started
#' ===
#' 
#' - object: use python to access mysql
#' 
#' - pre-requisit:
#'   - mysql
#'   - mysql-connector
#' 
## ## pip install mysql

## ## pip install mysql-connector

## ## pip install mysql-connector-python

## import mysql.connector

#' 
#' - trouble shooting
#' 
#' https://chat.openai.com/share/18cd0931-81e6-4be4-b3a1-40a73e5232d3
#' 
#' 
#' create a free database (1):
#' ===
#' 
#' - https://www.clever-cloud.com/
#' - https://api.clever-cloud.com/v2/sessions/login
#' 
#' ![](../figure/1_create.png){width=80%}
#' 
#' create a free database: (2)
#' ===
#' 
#' select MySQL
#' 
#' - use the free version
#' 
#' ![](../figure/2_freePlan.png){width=80%}
#' 
#' - session information (to be used for python connection)
#' 
#' ![](../figure/3_sql_info.png){width=80%}
#' 
#' session information
#' ===
#' 
#' - These login methods belong to to the instructor.
#' - **Try to create your own account**.
#' - Since this is a free database, the DB (database) name is fixed.
#' 
## MYSQL_ADDON_HOST="bpujmkvwzwgpe3ppgpeg-mysql.services.clever-cloud.com"

## MYSQL_ADDON_DB="bpujmkvwzwgpe3ppgpeg"

## MYSQL_ADDON_USER="uryjoqt0ohxsfzsn"

## MYSQL_ADDON_PORT="3306"

## MYSQL_ADDON_PASSWORD='hCkQynjDSNXpzUbpYUjL'

## MYSQL_ADDON_URI="mysql://uryjoqt0ohxsfzsn:hCkQynjDSNXpzUbpYUjL@bpujmkvwzwgpe3ppgpeg-mysql.services.clever-cloud.com:3306/bpujmkvwzwgpe3ppgpeg"

#' 
#' login
#' ===
#' 
## import mysql.connector

## 

## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## print(myconn)

## 

## mycursor = myconn.cursor() ## an object to interact with the SQL

## print(myconn)

#' 
#' 
#' 
#' 
#' 
#' Create a database
#' ===
#' 
#' - Create a database
#'   - This doesn't work in our class demonstration
#'   - Since we're using the free version
#'   - The dataset name is given
#' 
#' ```
#' myconn = mysql.connector.connect(
#'   host=MYSQL_ADDON_HOST,
#'   user=MYSQL_ADDON_USER,
#'   password=MYSQL_ADDON_PASSWORD,
#'   port=MYSQL_ADDON_PORT,
#'   database=MYSQL_ADDON_DB
#' )
#' mycursor = myconn.cursor()
#' mycursor.execute("CREATE DATABASE mydatabase")
#' ```
#' 
#' 
#' Show existing database
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## mycursor.execute("SHOW DATABASES")

## for db_name in mycursor:

##    print(db_name)

#' 
#' 
#' Create a table
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## 

## mycursor.execute("CREATE TABLE customers (name VARCHAR(255), address VARCHAR(255))")

#' 
#' - table name: customers
#'   - variable: name (type character)
#'   - variable: address (type character)
#' 
#' 
#' Show existing tables
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("SHOW TABLES")

## for table_name in mycursor:

##    print(table_name)

#' 
#' 
#' Delete a table (1)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("DROP TABLE customers")

## mycursor.execute("SHOW TABLES")

## for table_name in mycursor:

##    print(table_name)

#' 
#' 
#' Delete a table (2)
#' ===
#' 
#' - delete a table only if that exists
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## sql = "DROP TABLE IF EXISTS customers"

## mycursor.execute(sql)

## mycursor.execute("SHOW TABLES")

## for table_name in mycursor:

##    print(table_name)

#' 
#' 
#' Create a table with auto increasing ID as key
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")

## mycursor.execute("SHOW TABLES")

## for table_name in mycursor:

##    print(table_name)

#' 
#' 
#' 
#' Add an entry into the database
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## # sql = "INSERT INTO customers (name, address) VALUES ('John', 'Highway21')"

## # mycursor.execute(sql)

## sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"

## val = ("John", "Highway 21")

## mycursor.execute(sql, val)

## myconn.commit() ## commit changes to the database

## print(mycursor.rowcount, "record inserted.")

#' 
#' Print all content of a table
#' ===
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("SELECT * FROM customers")

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' Add multiple entries into the database
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## 

## sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"

## val = [

##   ('Peter', 'Lowstreet 4'),

##   ('Amy', 'Apple st 652'),

##   ('Hannah', 'Mountain 21'),

##   ('Michael', 'Valley 345'),

##   ('Sandy', 'Ocean blvd 2'),

##   ('Betty', 'Green Grass 1'),

##   ('Richard', 'Sky st 331'),

##   ('Susan', 'One way 98'),

##   ('Vicky', 'Yellow Garden 2'),

##   ('Ben', 'Park Lane 38'),

##   ('William', 'Central st 954'),

##   ('Chuck', 'Main Road 989'),

##   ('Viola', 'Sideway 1633')

## ]

## 

## mycursor.executemany(sql, val)

## 

## myconn.commit()

## 

## print(mycursor.rowcount, "was inserted.")

#' 
#' 
#' Print all content of a table
#' ===
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("SELECT * FROM customers")

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' Print head content of a table
#' ===
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("SELECT * FROM customers LIMIT 5")

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' Print head content of a table with offset
#' ===
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("SELECT * FROM customers LIMIT 5 OFFSET 2")

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' 
#' 
#' Select certain columns in a table
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("SELECT address, name FROM customers")

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' filter a table (1)
#' ===
#' 
#' - filter by a specific item
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## # sql = "SELECT * FROM customers WHERE address = 'Yellow Garden 2'"

## # mycursor.execute(sql)

## sql = "SELECT * FROM customers WHERE address = %s"

## adr = ("Yellow Garden 2", )

## mycursor.execute(sql, adr)

## 

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' filter a table (2)
#' ===
#' 
#' - filter by a pattern
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## sql = "SELECT * FROM customers WHERE address LIKE '%way%'"

## mycursor.execute(sql)

## 

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' Sort a table 
#' ===
#' 
#' - ascending order (default)
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## sql = "SELECT * FROM customers ORDER BY name"

## #sql = "SELECT * FROM customers ORDER BY name DESC" ## descending order

## mycursor.execute(sql)

## 

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' 
#' 
#' Delete a record in a table (1)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## sql = "DELETE FROM customers WHERE address = 'Mountain 21'"

## mycursor.execute(sql)

## myconn.commit()

## print(mycursor.rowcount, "record(s) deleted")

#' 
#' 
#' Delete a record in a table (2)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## sql = "DELETE FROM customers WHERE address = %s"

## adr = ("Yellow Garden 2", )

## mycursor.execute(sql, adr)

## myconn.commit()

## print(mycursor.rowcount, "record(s) deleted")

#' 
#' 
#' 
#' Update a record/entry (1)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## 

## sql = "UPDATE customers SET address = 'Canyon 123' WHERE address = 'Valley 345'"

## 

## mycursor.execute(sql)

## 

## myconn.commit()

## 

## print(mycursor.rowcount, "record(s) affected")

#' 
#' 
#' 
#' Update a record/entry (2)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## 

## sql = "UPDATE customers SET address = %s WHERE address = %s"

## val = ("Valley 345", "Canyon 123")

## mycursor.execute(sql, val)

## 

## myconn.commit()

## print(mycursor.rowcount, "record(s) affected")

#' 
#' merge two tables (create left table)
#' ===
#' 
#' - left table:
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## sql = "DROP TABLE IF EXISTS users"

## mycursor.execute(sql)

## 

## mycursor.execute("CREATE TABLE users (id INT(255), name VARCHAR(255), fav INT(255))")

## sql = "INSERT INTO users (id, name, fav) VALUES (%s, %s, %s)"

## val = [

##     (1, 'John', 154),

##     (2, 'Peter', 154),

##     (3, 'Amy', 155),

##     (4, 'Hannah', 167),

##     (5, 'Michael', 189)

## ]

## 

## mycursor.executemany(sql, val)

## myconn.commit()

## print(mycursor.rowcount, "was inserted.")

#' 
#' --- 
#' 
## mycursor.execute("SELECT * FROM users LIMIT 5")

## myresult = mycursor.fetchall()

## 

## for x in myresult:

##   print(x)

## 

#' 
#' 
#' 
#' merge two tables (create right table)
#' ===
#' 
#' - right table:
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## 

## mycursor = myconn.cursor()

## sql = "DROP TABLE IF EXISTS products"

## mycursor.execute(sql)

## 

## mycursor.execute("CREATE TABLE products (id INT(255), name VARCHAR(255))")

## 

## sql = "INSERT INTO products (id, name) VALUES (%s, %s)"

## val = [

##     (154, 'Chocolate Heaven'),

##     (155, 'Tasty Lemons'),

##     (156, 'Vanilla Dreams')

## ]

## 

## mycursor.executemany(sql, val)

## myconn.commit()

## print(mycursor.rowcount, "was inserted.")

#' 
#' --- 
#' 
## mycursor.execute("SELECT * FROM products LIMIT 5")

## myresult = mycursor.fetchall()

## 

## for x in myresult:

##   print(x)

## 

#' 
#' merge (inner) (1)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## 

## sql = "SELECT * \

##   FROM users \

##   INNER JOIN products ON users.fav = products.id"

## mycursor.execute(sql)

## 

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' 
#' 
#' merge (inner) (2)
#' ===
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## 

## sql = "SELECT \

##   users.name AS user, \

##   products.name AS favorite \

##   FROM users \

##   INNER JOIN products ON users.fav = products.id"

## mycursor.execute(sql)

## 

## myresult = mycursor.fetchall()

## for x in myresult:

##   print(x)

#' 
#' 
#' Reference
#' ===
#' 
#' 
#' https://www.w3schools.com/python/python_mysql_create_table.asp
#' 
#' 
#' 
## myconn = mysql.connector.connect(

##   host=MYSQL_ADDON_HOST,

##   user=MYSQL_ADDON_USER,

##   password=MYSQL_ADDON_PASSWORD,

##   port=MYSQL_ADDON_PORT,

##   database=MYSQL_ADDON_DB

## )

## mycursor = myconn.cursor()

## mycursor.execute("DROP TABLE customers")

## mycursor.execute("SHOW TABLES")

## #for table_name in mycursor:

## #   print(table_name)

#' 
